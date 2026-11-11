"""Calendar feature computation with Chinese holiday fallback."""
import numpy as np
import pandas as pd
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
