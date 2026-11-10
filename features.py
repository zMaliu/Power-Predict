"""特征工程：日历、节假日、目标滞后/滚动、气象（含统计滞后）。所有特征对"未来 18 天"推理同样可计算。"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from config import TARGETS, WEATHER_KEYS

try:
    import chinese_calendar as _cc

    _HAS_CC = True
except ImportError:
    _HAS_CC = False


def _holiday_flags(date) -> tuple[int, int, int]:
    """返回 (is_workday, is_statutory_holiday, is_adjusted_workday)。
    chinese_calendar 覆盖不到的日期回退到普通周末判断。"""
    d = date.date() if isinstance(date, pd.Timestamp) else date
    if _HAS_CC:
        try:
            is_wd = int(_cc.is_workday(d))
            detail = _cc.get_holiday_detail(d)
            is_stat = int(detail[0] and detail[1] is not None)  # 是假日且有名字才算法定节假日
            is_adj = int(d.weekday() >= 5 and _cc.is_workday(d))  # 周末但调休上班
            return is_wd, is_stat, is_adj
        except (NotImplementedError, NotImplementedError):
            pass
    # 回退：纯周末判断
    is_wd = int(d.weekday() < 5)
    return is_wd, 0, 0


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["dom"] = d.dt.day
    df["month"] = d.dt.month
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["doy"] = d.dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.0)
    # 节假日特征（chinese_calendar，含调休）
    flags = df["date"].apply(_holiday_flags)
    df["is_workday"] = flags.str[0]
    df["is_stat_holiday"] = flags.str[1]  # 法定节假日（国庆/春节等），不含普通周末
    df["is_adj_workday"] = flags.str[2]  # 调休上班的周末
    # 季节分段：居民用电有明显季节跳变
    df["season"] = df["month"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:2, 8:2, 9:3, 10:3, 11:0})
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    return df


def add_weather_derived(df: pd.DataFrame, lags=(1, 2), roll=(7,)) -> pd.DataFrame:
    if "tem" in df.columns:
        df["hdd"] = np.clip(15 - df["tem"], 0, None)   # 制热度日
        df["cdd"] = np.clip(df["tem"] - 26, 0, None)   # 制冷度日
        # 温度分桶：居民用电对温度敏感区间不同
        df["tem_low"] = np.clip(10 - df["tem"], 0, None)    # 低温取暖
        df["tem_mid"] = np.clip(df["tem"] - 10, 0, None) * np.clip(26 - df["tem"], 0, None)  # 舒适区间
        df["tem_high"] = np.clip(df["tem"] - 26, 0, None)   # 高温制冷
        # 温度变化率
        df["tem_diff1"] = df["tem"].diff(1)
        df["tem_diff3"] = df["tem"].diff(3)
        # 累积效应：连续高温/低温天数
        df["hot_streak"] = (df["tem"] > 28).astype(int).groupby((df["tem"] <= 28).cumsum()).cumsum()
        df["cold_streak"] = (df["tem"] < 20).astype(int).groupby((df["tem"] >= 20).cumsum()).cumsum()
    if "rhum" in df.columns:
        df["rhum_high"] = (df["rhum"] > 85).astype(int)  # 高湿闷热
    if "precip" in df.columns:
        df["has_rain"] = (df["precip"] > 0).astype(int)
        # 连续降雨天数
        df["rain_streak"] = (df["precip"] > 0).astype(int).groupby((df["precip"] <= 0).cumsum()).cumsum()
        # 强降雨标记
        df["heavy_rain"] = (df["precip"] > 10).astype(int)
    for c in WEATHER_KEYS:
        if c not in df.columns or c == "weather_type":
            continue
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
        for w in roll:
            df[f"{c}_r{w}"] = df[c].shift(1).rolling(w).mean()
    if "weather_type" in df.columns:
        df["weather_type"] = df["weather_type"].astype("category")
    return df


def add_target_lags(df: pd.DataFrame, lags: tuple[int, ...], roll_windows: tuple[int, ...]) -> pd.DataFrame:
    for t in TARGETS:
        if t not in df.columns:
            continue
        for L in lags:
            df[f"{t}_lag{L}"] = df[t].shift(L)
        for w in roll_windows:
            df[f"{t}_r{w}"] = df[t].shift(1).rolling(w).mean()
        # 同比特征：上周同日、上月同日
        df[f"{t}_lag7"] = df[t].shift(7)
        df[f"{t}_lag30"] = df[t].shift(30)
        # 与上周同日的差值
        df[f"{t}_diff7"] = df[t].shift(1) - df[t].shift(8)
    return df


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
