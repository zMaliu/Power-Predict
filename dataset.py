"""数据加载：两张「每天一行」表（全行业 / 城乡居民），各自含 date + 目标 + 气象。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import CFG, COLS, TARGETS, WEATHER_KEYS, resolve_paths


def load_daily(csv_path, target_key: str) -> pd.DataFrame:
    """
    读取单目标训练表。
    target_key: 'industry' 或 'resident'，对应表内日电量列（见 config.COLS）。
    """
    if target_key not in TARGETS:
        raise ValueError(f"未知目标: {target_key}")

    df = pd.read_csv(csv_path)
    # 日期 + 气象列统一命名
    rename = {src: key for key, src in COLS.items() if src in df.columns and key != target_key}
    df = df.rename(columns=rename)
    # 该表的目标列 -> 内部名 industry / resident
    src_target = COLS[target_key]
    if src_target in df.columns:
        df = df.rename(columns={src_target: target_key})
    elif target_key not in df.columns:
        raise ValueError(f"表 {csv_path} 缺少目标列 {src_target!r}")

    keep = ["date", target_key] + [k for k in WEATHER_KEYS if k in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for k in [target_key] + [w for w in WEATHER_KEYS if w != "weather_type"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def load_weather_json(path) -> pd.DataFrame:
    raw = json.loads(open(path, encoding="utf-8").read())
    data = raw["data"] if isinstance(raw, dict) else raw
    w = pd.DataFrame(data)
    w["date"] = pd.to_datetime(w["ddataTime"]).dt.normalize()
    out = pd.DataFrame({"date": w["date"]})
    m = {
        "tem": "tem",
        "tem_max": "temMax",
        "tem_min": "temMin",
        "rhum": "rhum",
        "precip": "prediction",
        "wind_speed": "windSpeed",
        "weather_type": "weatherType",
    }
    for k, src in m.items():
        if src in w.columns:
            out[k] = w[src]
    agg = {c: ("first" if c == "weather_type" else "mean") for c in out.columns if c != "date"}
    return out.groupby("date", as_index=False).agg(agg).sort_values("date").reset_index(drop=True)


def load_future_weather(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {src: key for key, src in COLS.items() if src in df.columns and key in WEATHER_KEYS + ("date",)}
    df = df.rename(columns=rename)
    keep = ["date"] + [k for k in WEATHER_KEYS if k in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def make_demo(cfg=None) -> None:
    """生成两张演示训练表 + 未来天气。"""
    cfg = resolve_paths(cfg or CFG)
    data_dir = cfg.data_files["industry"].parent
    data_dir.mkdir(parents=True, exist_ok=True)

    wpath = next(cfg.project_dir.glob("*weather*.json"))
    w = load_weather_json(wpath)
    rng = np.random.default_rng(cfg.seed)
    n = len(w)
    doy = w["date"].dt.dayofyear.values
    dow = w["date"].dt.dayofweek.values
    base = 2600 + 500 * np.sin(2 * np.pi * (doy - 120) / 365.0) + 120 * (dow >= 5)
    heat = 12 * np.clip(w["tem"].values - 26, 0, None) - 8 * np.clip(15 - w["tem"].values, 0, None)
    industry = (base + heat + rng.normal(0, 60, n)).round(2)
    resident = (
        900
        + 260 * np.sin(2 * np.pi * (doy - 150) / 365.0)
        + 8 * np.clip(w["tem"].values - 26, 0, None)
        + 60 * (dow >= 5)
        + rng.normal(0, 35, n)
    ).round(2)

    weather_cols = [c for c in WEATHER_KEYS if c in w.columns]
    base_df = w[["date"] + weather_cols].copy()

    ind = base_df.copy()
    ind[COLS["industry"]] = industry
    ind.to_csv(cfg.data_files["industry"], index=False, encoding="utf-8-sig")

    res = base_df.copy()
    res[COLS["resident"]] = resident
    res.to_csv(cfg.data_files["resident"], index=False, encoding="utf-8-sig")

    fut = w.tail(cfg.horizon).copy()
    fut["date"] = pd.date_range(w["date"].max() + pd.Timedelta(days=1), periods=cfg.horizon, freq="D")
    fut.to_csv(cfg.future_weather_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    cfg = resolve_paths(CFG)
    make_demo(cfg)
    for t in TARGETS:
        d = load_daily(cfg.data_files[t], t)
        print(f"{t}: {d.shape}, {d['date'].min().date()} -> {d['date'].max().date()}")
        print(d.head(2).to_string(), "\n")
