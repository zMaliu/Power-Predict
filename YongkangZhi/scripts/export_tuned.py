"""Assemble targets already selected on development data; never select on test."""

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from power_predict.config import Settings
from power_predict.forecasting.service import predict_file
from power_predict.storage import create_run_directory, write_json


def export(runs):
    targets, sources = {}, {}
    signature = None
    for path in runs:
        path = path.resolve()
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        config = json.loads((path / "settings.json").read_text(encoding="utf-8"))["settings"]
        current = {key: config[key] for key in ("horizon", "lags", "roll_windows", "data_dir")}
        if signature is not None and signature != current:
            raise ValueError("各目标的日期长度、特征或数据目录配置不一致")
        signature = current
        for target, result in report["targets"].items():
            if target in targets:
                raise ValueError(f"重复目标: {target}")
            if result["selected"] not in ("default", "candidate"):
                raise ValueError("未知开发集选择结果")
            targets[target] = dict(source=path, selection=result["selected"])
            sources[target] = dict(report=str(path / "report.json"), selected=result["selected"])
    if not targets:
        raise ValueError("至少提供一个完成的调参目录")
    destination = create_run_directory(ROOT / "models/runs", "tuned")
    for target, info in targets.items():
        shutil.copytree(info["source"] / "selected_models" / target, destination / target)
    write_json(destination / "meta.json", dict(contract="anchor_direct_v1", sources=sources,
               selection_rule="copied development selections unchanged; no holdout reselection",
               signature=signature))
    settings = replace(Settings(), model_dir=destination, horizon=signature["horizon"],
                       lags=tuple(signature["lags"]), roll_windows=tuple(signature["roll_windows"]),
                       data_dir=Path(signature["data_dir"]))
    prediction = predict_file(settings, targets=list(targets), include_total=len(targets) == 2)
    write_json(destination / "prediction.json", dict(output=str(prediction.output_path)))
    print("Exported models:", destination)
    print("Prediction:", prediction.output_path)
    print(prediction.forecast.to_string(index=False))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    export(parser.parse_args().runs)
