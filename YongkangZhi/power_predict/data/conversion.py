"""Explicit raw-JSON import; writes only to a newly allocated import directory."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ..storage import create_run_directory, write_json
CODE_MAP = {"MX0000": "industry", "MY0000": "resident"}
WEATHER_MAP = {"tem": "tem", "temMax": "tem_max", "temMin": "tem_min", "rhum": "rhum",
               "prediction": "precip", "windSpeed": "wind_speed", "weatherType": "weather_type"}
WEATHER_COLS = ["tem", "tem_max", "tem_min", "rhum", "precip", "wind_speed", "weather_type"]
TARGET_COLS = {"industry": "industry_kwh", "resident": "resident_kwh"}


def load_elec(path: str) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw)
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["apTotal"] = pd.to_numeric(df["apTotal"], errors="coerce")
    df["sumTotal"] = pd.to_numeric(df["sumTotal"], errors="coerce")
    return df


def load_weather(path: str, extra_path: str | None = None) -> pd.DataFrame:
    """加载天气数据，可选合并补充数据，并填充缺失值。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = raw["data"] if isinstance(raw, dict) else raw
    w = pd.DataFrame(data)
    w["date"] = pd.to_datetime(w["ddataTime"]).dt.normalize()
    out = pd.DataFrame({"date": w["date"]})
    for src, dst in WEATHER_MAP.items():
        if src in w.columns:
            out[dst] = w[src]

    # 合并补充天气数据（缺失的 7 天）
    if extra_path and Path(extra_path).exists():
        raw2 = json.loads(Path(extra_path).read_text(encoding="utf-8"))
        data2 = raw2["data"] if isinstance(raw2, dict) else raw2
        w2 = pd.DataFrame(data2)
        w2["date"] = pd.to_datetime(w2["ddataTime"]).dt.normalize()
        extra = pd.DataFrame({"date": w2["date"]})
        for src, dst in WEATHER_MAP.items():
            if src in w2.columns:
                extra[dst] = w2[src]
        # 只补充主表中缺失的日期
        existing = set(out["date"])
        extra_new = extra[~extra["date"].isin(existing)]
        out = pd.concat([out, extra_new], ignore_index=True)
        print(f"  补充天气: {len(extra_new)} 天")

    # 同一日期若多行（多站点），数值取均值、天气现象取首个
    agg = {c: ("first" if c == "weather_type" else "mean") for c in out.columns if c != "date"}
    out = out.groupby("date", as_index=False).agg(agg).sort_values("date").reset_index(drop=True)

    # 填充缺失值：tem 用 (tem_max+tem_min)/2 估算，其余线性插值
    if "tem" in out.columns and "tem_max" in out.columns and "tem_min" in out.columns:
        tem_est = (out["tem_max"] + out["tem_min"]) / 2
        out["tem"] = out["tem"].where(out["tem"].notna(), tem_est)
    for c in ["rhum", "wind_speed", "tem_max", "tem_min", "precip"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].interpolate(method="linear", limit_direction="both")

    # 统一取整：与主天气文件保持一致（tem_max/tem_min 为整数）
    for c in ["tem_max", "tem_min"]:
        if c in out.columns:
            out[c] = out[c].round(0)

    return out


def split_and_merge(elec: pd.DataFrame, weather: pd.DataFrame, out_dir: Path) -> dict:
    """按 codeClsVal 拆两张表，各自合并天气，落盘 CSV。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for code, key in CODE_MAP.items():
        sub = elec[elec["codeClsVal"] == code][["date", "apTotal"]].copy()
        sub = sub.rename(columns={"apTotal": TARGET_COLS[key]})
        sub = sub.sort_values("date").reset_index(drop=True)

        merged = sub.merge(weather, on="date", how="left")
        # 列顺序：date, 目标, 天气
        cols = ["date", TARGET_COLS[key]] + [c for c in WEATHER_COLS if c in merged.columns]
        merged = merged[cols]
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

        out_path = out_dir / f"{key}_daily.csv"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        paths[key] = out_path
    return paths


def convert(electricity: Path, weather: Path, extra_weather: Path | None, data_dir: Path) -> Path:
    electricity_frame = load_elec(str(electricity))
    weather_frame = load_weather(str(weather), str(extra_weather) if extra_weather else None)
    directory = create_run_directory(data_dir / "imports", "import")
    paths = split_and_merge(electricity_frame, weather_frame, directory)
    summary = {}
    for target, path in paths.items():
        frame = pd.read_csv(path)
        summary[target] = dict(rows=len(frame), first_date=frame["date"].min(),
                               last_date=frame["date"].max(), missing=frame.isna().sum().to_dict())
    write_json(directory / "data_report.json", summary)
    return directory
