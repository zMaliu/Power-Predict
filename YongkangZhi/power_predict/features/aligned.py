"""Forecast-date weather with target history strictly frozen at the origin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from .pipeline import build_feature_frame, feature_columns

CONTRACT = "anchor_direct_v1"


def validate_daily(frame: pd.DataFrame, target: str) -> None:
    dates = pd.to_datetime(frame["date"])
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("时间对齐模型要求日期有效、递增且不重复")
    if len(dates) < 2 or not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("时间对齐模型要求连续自然日，请先处理历史缺日")
    if target not in frame or not np.isfinite(frame[target].to_numpy(dtype=float)).all():
        raise ValueError("训练历史目标必须完整且为有限数值")
    if (frame[target] < 0).any():
        raise ValueError("用电量不能为负")


def aligned_frame(daily: pd.DataFrame, target: str, horizon: int, settings: Settings) -> pd.DataFrame:
    """Row d predicts y(d), using target values no later than d-horizon.

    Weather at d and preceding days is assumed supplied by the weather forecast.
    This retrospective dataset only contains realized weather, not forecast vintages.
    """
    if horizon < 1:
        raise ValueError("horizon须大于0")
    use_log = settings.log_transform.get(target, False)
    frame = build_feature_frame(daily, settings.lags, settings.roll_windows,
                                target if use_log else None)
    values = frame[target]
    for lag in set(settings.lags) | {7, 30}:
        frame[f"{target}_lag{lag}"] = values.shift(horizon + lag - 1)
    for window in settings.roll_windows:
        frame[f"{target}_r{window}"] = values.shift(horizon).rolling(window).mean()
    frame[f"{target}_diff7"] = values.shift(horizon) - values.shift(horizon + 7)
    return frame


def aligned_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    return feature_columns(frame)
