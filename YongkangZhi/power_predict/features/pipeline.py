"""Single feature pipeline shared by training and inference."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..domain import TARGETS
from .calendar import add_calendar
from .weather import add_weather_derived
from .target import add_target_lags


def build_feature_frame(df: pd.DataFrame, lags=(1, 2, 3, 5, 7, 14, 21), roll_windows=(7, 14), log_transform_target: str | None = None) -> pd.DataFrame:
    """历史日表 -> 完整特征帧（含目标列，供训练切窗用）。
    log_transform_target: 若指定，对该目标列先做 log1p 再算 lag，保证训练和推理一致。"""
    df = df.sort_values("date").reset_index(drop=True).copy()
    if log_transform_target and log_transform_target in df.columns:
        df[log_transform_target] = np.log1p(df[log_transform_target])
    df = add_calendar(df)
    df = add_weather_derived(df)
    df = add_target_lags(df, lags, roll_windows)
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """返回（特征列, 类别特征列）。剔除 date 与表中存在的目标列。"""
    drop = {"date"} | {t for t in TARGETS if t in df.columns}
    feats = [c for c in df.columns if c not in drop]
    cat = [c for c in feats if str(df[c].dtype) == "category"]
    return feats, cat
