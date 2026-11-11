"""Baseline weather completion. Does not access models or write files."""
import numpy as np
import pandas as pd


def fill_missing_weather(fw: pd.DataFrame, history: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    把 future_weather 补齐到 cfg.horizon 天。
    缺失日期：按历史同星期几的均值补 tem/tem_max/tem_min/rhum/precip/wind_speed；
    weather_type 用历史同星期众数或默认 '多云'。
    """
    fw = fw.copy()
    fw["date"] = pd.to_datetime(fw["date"]).dt.normalize()
    fw = fw.sort_values("date").reset_index(drop=True)

    if len(fw) >= cfg.horizon:
        return fw.head(cfg.horizon)

    # 需要补的日期：从 fw 最后一天往后续，或从 anchor 后第 1 天开始
    anchor = fw["date"].min() - pd.Timedelta(days=1)
    all_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=cfg.horizon, freq="D")
    existing = set(fw["date"].tolist())
    missing_dates = [d for d in all_dates if d not in existing]

    if not missing_dates:
        return fw

    # 历史同星期均值
    hist = history.copy()
    hist["dow"] = hist["date"].dt.dayofweek
    num_cols = ["tem", "tem_max", "tem_min", "rhum", "precip", "wind_speed"]
    dow_mean = hist.groupby("dow")[num_cols].mean()
    overall_mean = hist[num_cols].mean()

    # weather_type 众数
    if "weather_type" in hist.columns:
        dow_mode = hist.groupby("dow")["weather_type"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "多云")
    else:
        dow_mode = pd.Series(dtype=str)

    rows = []
    for d in missing_dates:
        dow = d.dayofweek
        row = {"date": d}
        for c in num_cols:
            v = dow_mean.loc[dow, c] if dow in dow_mean.index else overall_mean.get(c, np.nan)
            row[c] = round(float(v), 1) if pd.notna(v) else None
        row["weather_type"] = dow_mode.get(dow, "多云") if len(dow_mode) else "多云"
        rows.append(row)

    fill_df = pd.DataFrame(rows)
    out = pd.concat([fw, fill_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out.head(cfg.horizon)
