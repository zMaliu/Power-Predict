"""Weather derived features, preserving the baseline calculation order."""
import numpy as np
import pandas as pd
from ..domain import WEATHER_KEYS


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
