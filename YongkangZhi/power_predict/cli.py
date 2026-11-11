"""Thin command-line adapters. All business logic lives in application services."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .domain import TARGETS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="日用电量18天预测")
    parser.add_argument("--project-dir", type=Path, default=Settings().project_dir)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--training-dir", type=Path, default=Path("models/runs"))
    parser.add_argument("--weather", type=Path)
    parser.add_argument("--threads", type=int, help="LightGBM执行线程数；未传则沿用库默认行为")
    parser.add_argument("--targets", action="append", choices=list(TARGETS), help="可重复指定；默认两个目标")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="完整训练，输出到独立模型运行目录")
    prediction = commands.add_parser("predict", help="从CSV预测并保存到独立输出目录")
    prediction.add_argument("--include-total", action="store_true")
    commands.add_parser("run", help="先训练，再使用本次模型预测")
    tuning = commands.add_parser("tune", help="Optuna搜索、严格时间回测及独立测试对比")
    tuning.add_argument("--trials", type=int, default=30)
    tuning.add_argument("--holdout-days", type=int, default=90)
    tuning.add_argument("--folds", type=int, default=3)
    tuning.add_argument("--cv-days", type=int, default=54)
    tuning.add_argument("--cv-spacing", type=int, default=120)
    tuning.add_argument("--max-estimators", type=int, default=3000)
    tuning.add_argument("--search-horizons", type=int, nargs="+", default=[1, 7, 14, 18])
    tuning.add_argument("--skip-legacy", action="store_true", help="省略原策略对照臂")
    server = commands.add_parser("serve", help="启动HTTP服务")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--target", choices=[*TARGETS, "all"], default=None)
    conversion = commands.add_parser("convert", help="显式转换原始JSON，写入独立导入目录")
    conversion.add_argument("--electricity", type=Path, required=True)
    conversion.add_argument("--weather-json", type=Path, required=True)
    conversion.add_argument("--extra-weather", type=Path)
    args = parser.parse_args(argv)
    args.targets = args.targets or list(TARGETS)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings(project_dir=args.project_dir, data_dir=args.data_dir, model_dir=args.model_dir,
                        out_dir=args.output_dir, training_dir=args.training_dir, future_weather_csv=args.weather)
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads须大于0")
        settings = replace(settings, lgb_params=dict(settings.lgb_params, num_threads=args.threads))
    if args.command in ("train", "run"):
        from .models.training import train
        from .models.metrics import summarize
        training = train(settings, args.targets)
        print(summarize(training.metrics).to_string(index=False))
        print(f"模型目录: {training.model_dir}")
        print(f"指标文件: {training.meta_path}")
        settings = replace(settings, model_dir=training.model_dir)
    if args.command in ("predict", "run"):
        from .forecasting.service import predict_file
        result = predict_file(settings, args.targets, getattr(args, "include_total", False))
        print(result.forecast.to_string(index=False))
        print(f"预测文件: {result.output_path}")
    if args.command == "serve":
        import uvicorn
        from .api.app import create_app
        uvicorn.run(create_app(settings, args.target), host=args.host, port=args.port)
    if args.command == "tune":
        from .tuning import SearchOptions, tune
        options = SearchOptions(trials=args.trials, folds=args.folds, holdout_days=args.holdout_days,
                                cv_days=args.cv_days, cv_spacing=args.cv_spacing,
                                max_estimators=args.max_estimators,
                                search_horizons=tuple(args.search_horizons), compare_legacy=not args.skip_legacy)
        destination = tune(settings, args.targets, options)
        print(f"调参与独立测试报告: {destination / 'report.json'}")
        print(f"开发集选择后重训的模型: {destination / 'selected_models'}")
    if args.command == "convert":
        from .data.conversion import convert
        destination = convert(args.electricity, args.weather_json, args.extra_weather, settings.resolve().data_dir)
        print(f"导入目录: {destination}；使用 --data-dir 指向该目录后再训练")
    return 0
