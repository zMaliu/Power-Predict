"""Inference adapter for the explicit anchor_direct_v1 model contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..features.aligned import aligned_frame, validate_daily
from ..models.repository import ModelRepository


def predict_aligned(target, history, future, settings, repository: ModelRepository):
    validate_daily(history, target)
    expected = pd.date_range(history["date"].iloc[-1] + pd.Timedelta(days=1),
                             periods=settings.horizon)
    if not pd.DatetimeIndex(future["date"]).equals(expected):
        raise ValueError("时间对齐模型要求从历史末日次日开始的连续horizon天天气")
    spec = repository.load_spec(target)
    if spec["horizon"] != settings.horizon:
        raise ValueError("预测长度与模型训练长度不一致")
    settings = replace(settings, lags=tuple(spec["lags"]),
                       roll_windows=tuple(spec["roll_windows"]),
                       log_transform=dict(settings.log_transform, **{target: spec["log_transform"]}))
    future = future.copy()
    future[target] = np.nan
    merged = pd.concat([history, future], ignore_index=True)
    values = []
    for h in range(1, settings.horizon + 1):
        frame = aligned_frame(merged, target, h, settings)
        row = frame.loc[frame["date"].eq(expected[h - 1]), spec["cols"]].copy()
        vocab = spec.get("categories_by_horizon", {}).get(str(h), spec.get("categories", {}))
        for column, categories in vocab.items():
            row[column] = pd.Categorical(row[column].where(row[column].isin(categories)), categories=categories)
        prediction = float(repository.load_model(target, h).predict(row)[0])
        if spec["log_transform"]:
            prediction = float(np.expm1(prediction))
        values.append(prediction)
    if not np.isfinite(values).all():
        raise ValueError("模型预测包含非有限值")
    return values
