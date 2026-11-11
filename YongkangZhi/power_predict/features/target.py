"""Target lag and rolling features."""
import pandas as pd
from ..domain import TARGETS


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
