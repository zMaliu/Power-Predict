"""CSV/JSON adapters. They never create or replace input data."""
from __future__ import annotations
import json
import pandas as pd
from ..domain import COLS, TARGETS, WEATHER_KEYS


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
    from pathlib import Path
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
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
