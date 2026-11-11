"""Execute original modules in a separate interpreter with isolated output paths.

This script is only a comparison harness, never imported by the refactored app.
"""

import json
import sys
from pathlib import Path

import joblib


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    source = Path(job["source"])
    destination = Path(job["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    import config
    from dataset import load_daily, load_future_weather
    from features import build_feature_frame
    from predict import predict_from_weather

    config.DATA_FILES = {target: str(Path(job["data_dir"]) / f"{target}_daily.csv") for target in config.TARGETS}
    cfg = config.Config(project_dir=destination, model_dir=Path(job["model_dir"]), out_dir=destination,
                        future_weather_csv=Path(job["weather"]))
    cfg.lgb_params.update(job.get("lgb_params", {}))
    if job["task"] == "train":
        from train import train
        train(cfg)
    cfg = config.resolve_paths(cfg)
    for target in config.TARGETS:
        daily = load_daily(cfg.data_files[target], target)
        frame = build_feature_frame(daily, lags=cfg.lags, roll_windows=cfg.roll_windows,
                                    log_transform_target=target if cfg.log_transform[target] else None)
        joblib.dump(frame, destination / f"features_{target}.pkl")
    weather = load_future_weather(job["weather"])
    for count in (18, 7):
        result = predict_from_weather(weather.head(count), cfg=cfg, include_total=True)
        result.to_csv(destination / f"forecast_{count}d.csv", index=False, encoding="utf-8-sig")
    import api
    api.CFG_RT = cfg
    for count in (18, 7):
        payload = dict(anchor_date="2026-07-31", weather=weather.head(count).assign(
            date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")).to_dict("records"))
        request = api.PredictRequest.model_validate(payload)
        for target in config.TARGETS:
            response = api._run_predict(target, request)
            (destination / f"api_{target}_{count}d.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
