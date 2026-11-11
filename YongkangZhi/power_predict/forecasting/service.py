"""Prediction use case: orchestrates data, features, and strategy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..data.loaders import load_daily, load_future_weather
from ..domain import TARGETS
from ..models.repository import ModelRepository
from ..storage import create_run_directory, write_json
from .strategies import direct_predict, hybrid_predict, recursive_predict
from .weather import fill_missing_weather

STRATEGIES = {"recursive": recursive_predict, "direct": direct_predict, "hybrid": hybrid_predict}


def build_merged(history: pd.DataFrame, future: pd.DataFrame, target: str, settings: Settings) -> pd.DataFrame:
    need = max(settings.lags) + max(settings.roll_windows) + 5
    hist = history.sort_values("date").tail(need).copy()
    future = future.copy()
    future[target] = np.nan
    return pd.concat([hist, future], ignore_index=True).sort_values("date").reset_index(drop=True)


class ForecastService:
    def __init__(self, settings: Settings | None = None):
        self.settings = (settings or Settings()).resolve()
        self.repository = ModelRepository(self.settings.model_dir)

    def predict(self, future_weather: pd.DataFrame, targets: list[str] | None = None,
                include_total: bool = False) -> pd.DataFrame:
        want = list(TARGETS) if targets is None else targets
        if not want or any(t not in TARGETS for t in want):
            raise ValueError("targets必须包含至少一个已知目标")
        if future_weather.empty or "date" not in future_weather:
            raise ValueError("weather不能为空且必须包含date")
        fw = future_weather.copy()
        fw["date"] = pd.to_datetime(fw["date"]).dt.normalize()
        fw = fw.sort_values("date").reset_index(drop=True)
        if fw["date"].isna().any() or fw["date"].duplicated().any():
            raise ValueError("weather日期不能为空或重复")
        missing = self.repository.missing(want, self.settings.horizon)
        if missing:
            raise FileNotFoundError("缺少模型文件: " + ", ".join(missing))
        histories = {target: load_daily(self.settings.data_files[target], target) for target in want}
        fw = fill_missing_weather(fw, histories[want[0]], self.settings)
        future_dates = sorted(fw["date"].tolist())
        result = pd.DataFrame({"date": fw["date"]})
        for target in want:
            spec = self.repository.load_spec(target)
            if spec.get("contract") == "anchor_direct_v1":
                from .aligned import predict_aligned
                values = predict_aligned(target, histories[target], fw, self.settings, self.repository)
                result[TARGETS[target]] = np.round(values, 2)
                continue
            if spec.get("contract") not in (None, "legacy"):
                raise ValueError(f"未知模型时间契约: {spec['contract']}")
            strategy = self.settings.predict_strategy.get(target, "recursive")
            if strategy not in STRATEGIES:
                raise ValueError(f"未知预测策略: {strategy}")
            merged = build_merged(histories[target], fw, target, self.settings)
            values = STRATEGIES[strategy](target, merged, future_dates, self.settings)
            result[TARGETS[target]] = np.round(values, 2)
        if include_total and all(TARGETS[t] in result for t in TARGETS):
            result["全社会用电"] = np.round(result[TARGETS["industry"]] + result[TARGETS["resident"]], 2)
        return result


@dataclass(frozen=True)
class PredictionResult:
    forecast: pd.DataFrame
    output_path: Path


def predict_file(settings: Settings | None = None, targets: list[str] | None = None,
                 include_total: bool = False) -> PredictionResult:
    service = ForecastService(settings)
    weather = load_future_weather(service.settings.future_weather_csv)
    result = service.predict(weather, targets=targets, include_total=include_total)
    destination = create_run_directory(service.settings.out_dir, "predict")
    output = destination / f"forecast_{result['date'].min().date()}_{result['date'].max().date()}.csv"
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        result.to_csv(stream, index=False)
    write_json(destination / "settings.json", asdict(service.settings))
    return PredictionResult(result, output)
