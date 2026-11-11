"""Reproducible numerical comparison; all writes stay under this new project."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from power_predict.api.app import create_app
from power_predict.config import Settings
from power_predict.data.loaders import load_daily, load_future_weather
from power_predict.domain import TARGETS
from power_predict.features.pipeline import build_feature_frame
from power_predict.forecasting.service import ForecastService
from power_predict.models.training import train
from power_predict.storage import create_run_directory, write_json
from tests.asgi_client import request


def compare_frames(original: pd.DataFrame, refactored: pd.DataFrame) -> dict:
    pd.testing.assert_frame_equal(original, refactored, check_exact=True)
    numeric = original.select_dtypes(include="number").columns.tolist()
    differences = {}
    for column in numeric:
        delta = np.abs(original[column].to_numpy() - refactored[column].to_numpy())
        differences[column] = dict(max_abs_difference=float(delta.max()), mean_abs_difference=float(delta.mean()))
    return dict(rows=len(original), exactly_equal=True, columns=differences)


def original_job(source: Path, model_dir: Path, destination: Path, task: str) -> float:
    job = dict(source=str(source), destination=str(destination), task=task,
               data_dir=str(ROOT / "data"), model_dir=str(model_dir),
               weather=str(ROOT / "data/demo_future_weather.csv"), lgb_params={"num_threads": 1})
    path = destination.parent / f"job_{task}_{destination.name}.json"
    write_json(path, job)
    environment = dict(os.environ, OMP_NUM_THREADS="1", PYTHONWARNINGS="ignore", PYTHONUTF8="1")
    started = time.perf_counter()
    with (destination.parent / f"original_{task}.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/original_runner.py"), str(path)],
                       cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def compare_suite(settings: Settings, original_directory: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(features={}, forecasts={}, api={})
    weather = load_future_weather(settings.resolve().future_weather_csv)
    for target in TARGETS:
        daily = load_daily(settings.resolve().data_files[target], target)
        features = build_feature_frame(daily, lags=settings.lags, roll_windows=settings.roll_windows,
                                       log_transform_target=target if settings.log_transform[target] else None)
        reference = joblib.load(original_directory / f"features_{target}.pkl")
        pd.testing.assert_frame_equal(reference, features, check_exact=True)
        report["features"][target] = dict(exactly_equal=True, rows=len(features), columns=len(features.columns))
    service, app = ForecastService(settings), create_app(settings)
    for count in (18, 7):
        actual = service.predict(weather.head(count), include_total=True)
        actual.to_csv(destination / f"forecast_{count}d.csv", index=False, encoding="utf-8-sig")
        reference = pd.read_csv(original_directory / f"forecast_{count}d.csv", parse_dates=["date"])
        comparable = actual.copy()
        comparable["date"] = comparable["date"].astype(reference["date"].dtype)
        report["forecasts"][f"{count}d_weather"] = compare_frames(reference, comparable)
        payload = dict(anchor_date="2026-07-31", weather=weather.head(count).assign(
            date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")).to_dict("records"))
        for target in TARGETS:
            status, response = request(app, "POST", f"/predict/{target}", payload)
            reference_response = json.loads((original_directory / f"api_{target}_{count}d.json").read_text(encoding="utf-8"))
            assert status == 200 and response == reference_response, (target, count, status)
            report["api"][f"{target}_{count}d_weather"] = dict(status=status, exactly_equal=True)
            write_json(destination / f"api_{target}_{count}d.json", response)
    return report


def historical_response_comparison(source: Path, settings: Settings) -> dict:
    weather = load_future_weather(settings.resolve().future_weather_csv)
    service, result = ForecastService(settings), {}
    for filename, target in [("resp_industry_18.json", "industry"), ("resp_resident.json", "resident")]:
        saved = json.loads((source / "outputs" / filename).read_text(encoding="utf-8"))
        count = saved.get("weather_days_received", 18)
        current = service.predict(weather.head(count), targets=[target])
        values = np.array([row[TARGETS[target]] for row in saved["forecast"]])
        delta = np.abs(current[TARGETS[target]].to_numpy() - values)
        result[filename] = dict(weather_days=count, rows=len(current), max_abs_difference=float(delta.max()),
                               mean_abs_difference=float(delta.mean()), exactly_equal=bool(np.all(delta == 0)))
    return result


def verify_source(source: Path) -> dict:
    manifest = json.loads((ROOT / "reports/source_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, relative
    return dict(files=len(manifest["sha256"]), unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "Power-Predict")
    parser.add_argument("--train", action="store_true", help="完整重训原版和重构版各36个模型")
    args = parser.parse_args()
    destination = create_run_directory(ROOT / "reports", "comparison")
    report = dict(environment={package: importlib.metadata.version(package) for package in
                              ["numpy", "pandas", "lightgbm", "scikit-learn", "fastapi", "joblib", "chinese-calendar"]},
                  python=sys.version, forecast_start="2026-08-01", forecast_end="2026-08-18",
                  source_before=verify_source(args.source), path=str(destination))
    settings = Settings().resolve()
    print("Comparing existing model bundles...", flush=True)
    elapsed = original_job(args.source, settings.model_dir, destination / "original_baseline", "predict")
    report["existing_models"] = compare_suite(settings, destination / "original_baseline", destination / "refactored_baseline")
    report["existing_models"]["original_worker_seconds"] = elapsed
    report["historical_saved_responses"] = historical_response_comparison(args.source, settings)
    if args.train:
        print("Training original: 2 targets x 18 horizons, up to 1000 rounds, one execution thread...", flush=True)
        old_directory = destination / "original_retrained"
        old_seconds = original_job(args.source, old_directory / "models", old_directory, "train")
        print("Training refactored: same data, features, labels, and parameters...", flush=True)
        started = time.perf_counter()
        trained = train(replace(settings, lgb_params=dict(settings.lgb_params, num_threads=1)))
        new_seconds = time.perf_counter() - started
        old_meta = json.loads((old_directory / "models/meta.json").read_text(encoding="utf-8"))
        assert old_meta["metrics"] == trained.metrics, "retrained validation metrics differ"
        report["retrained"] = compare_suite(replace(settings, model_dir=trained.model_dir), old_directory,
                                             destination / "refactored_retrained")
        report["retrained"].update(metrics_exactly_equal=True, original_seconds=old_seconds,
                                    refactored_seconds=new_seconds, model_dir=str(trained.model_dir),
                                    execution_threads=1, n_estimators=1000, horizon=18,
                                    summary={target: {key: float(np.mean([m[key] for m in horizons.values()]))
                                                      for key in ("mae", "mape", "rmse")}
                                             for target, horizons in trained.metrics.items()})
    report["source_after"] = verify_source(args.source)
    write_json(destination / "comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("Report:", destination / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
