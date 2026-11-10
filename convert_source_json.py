"""
将溯源提供的两个 JSON 转成训练用的两张表：
  - data/industry_daily.csv   全行业用电 + 天气
  - data/resident_daily.csv   城乡居民用电 + 天气

同时输出数据质量报告到 outputs/data_report.txt。

用法:
  python convert_source_json.py
  # 或指定路径
  python convert_source_json.py --elec <path> --weather <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 溯源文件默认路径
DEFAULT_ELEC = r"c:\Users\test\Documents\xwechat_files\wxid_xav1t8junvtd22_31eb\temp\RWTemp\2026-08\7a9d7b9c40c304ccc20af7b73d41c3ad\electricity_2025-08-01_2026-07-31(1).json"
DEFAULT_WEATHER = r"c:\Users\test\Documents\xwechat_files\wxid_xav1t8junvtd22_31eb\temp\RWTemp\2026-08\7a9d7b9c40c304ccc20af7b73d41c3ad\weather.json"
DEFAULT_WEATHER_EXTRA = r"c:\Users\test\Documents\xwechat_files\wxid_xav1t8junvtd22_31eb\temp\RWTemp\2026-08\7a9d7b9c40c304ccc20af7b73d41c3ad\250801-05260418260419(1).txt"

# 用电分类编码 -> 内部目标名
CODE_MAP = {
    "MX0000": "industry",   # A、全行业用电合计
    "MY0000": "resident",   # B、城乡居民生活用电合计
}
CODE_NAME = {
    "MX0000": "A、全行业用电合计",
    "MY0000": "B、城乡居民生活用电合计",
}

# 天气字段映射: 源字段 -> 内部统一名
WEATHER_MAP = {
    "tem": "tem",
    "temMax": "tem_max",
    "temMin": "tem_min",
    "rhum": "rhum",
    "prediction": "precip",      # 降水量
    "windSpeed": "wind_speed",
    "weatherType": "weather_type",
}

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


def report(elec: pd.DataFrame, weather: pd.DataFrame, paths: dict, out_dir: Path) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("数据转换报告")
    lines.append("=" * 70)

    # 用电
    lines.append("\n[用电数据]")
    lines.append(f"记录数: {len(elec)}")
    lines.append(f"日期范围: {elec['date'].min().date()} ~ {elec['date'].max().date()}")
    lines.append(f"唯一日期数: {elec['date'].nunique()}")
    for code, key in CODE_MAP.items():
        sub = elec[elec["codeClsVal"] == code]
        lines.append(f"  {code} ({CODE_NAME[code]}): {len(sub)} 行, "
                     f"apTotal min={sub['apTotal'].min():.2f} max={sub['apTotal'].max():.2f} "
                     f"mean={sub['apTotal'].mean():.2f}")

    # sumTotal 校验
    piv = elec.pivot_table(index="date", columns="codeClsVal", values="apTotal", aggfunc="first")
    st = elec.groupby("date")["sumTotal"].first()
    if "MX0000" in piv and "MY0000" in piv:
        diff = (piv["MX0000"] + piv["MY0000"] - st).abs()
        lines.append(f"\n[全社会校验] sumTotal vs (全行业+城乡居民)")
        lines.append(f"  最大差异: {diff.max():.4f}")
        lines.append(f"  不一致天数(>0.01): {(diff > 0.01).sum()}")

    # 天气
    lines.append("\n[天气数据]")
    lines.append(f"记录数: {len(weather)}")
    lines.append(f"日期范围: {weather['date'].min().date()} ~ {weather['date'].max().date()}")
    lines.append(f"唯一日期数: {weather['date'].nunique()}")
    for c in WEATHER_COLS:
        if c in weather.columns:
            s = pd.to_numeric(weather[c], errors="coerce") if c != "weather_type" else weather[c]
            if c == "weather_type":
                lines.append(f"  {c}: {sorted(weather[c].dropna().unique())}")
            else:
                lines.append(f"  {c}: min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f} null={s.isna().sum()}")

    # 缺日
    start = datetime(2025, 8, 1)
    end = datetime(2026, 7, 31)
    all_days = set()
    d = start
    while d <= end:
        all_days.add(d.date())
        d += timedelta(days=1)
    e_days = set(elec["date"].dt.date)
    w_days = set(weather["date"].dt.date)
    lines.append("\n[缺日检查] (期望 2025-08-01 ~ 2026-07-31 共 365 天)")
    lines.append(f"  用电缺日: {len(all_days - e_days)} 天 -> {sorted(all_days - e_days)}")
    lines.append(f"  天气缺日: {len(all_days - w_days)} 天 -> {sorted(all_days - w_days)}")
    lines.append(f"  用电有/天气无: {len(e_days - w_days)} 天 -> {sorted(e_days - w_days)}")
    lines.append(f"  天气有/用电无: {len(w_days - e_days)} 天 -> {sorted(w_days - e_days)}")

    # 输出文件
    lines.append("\n[输出文件]")
    for key, p in paths.items():
        df = pd.read_csv(p)
        miss_w = df[df.iloc[:, 2:].isna().any(axis=1)]  # 天气列缺失的行
        lines.append(f"  {p.name}: {len(df)} 行, {list(df.columns)}")
        lines.append(f"    天气缺失行: {len(miss_w)} 天 -> {miss_w['date'].tolist() if len(miss_w) else '无'}")

    text = "\n".join(lines)
    rep_path = out_dir / "data_report.txt"
    rep_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n报告已保存: {rep_path}")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elec", default=DEFAULT_ELEC, help="用电 JSON 路径")
    ap.add_argument("--weather", default=DEFAULT_WEATHER, help="天气 JSON 路径")
    ap.add_argument("--weather-extra", default=DEFAULT_WEATHER_EXTRA, help="补充天气 JSON 路径")
    ap.add_argument("--out", default="data", help="输出目录")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(__file__).resolve().parent / "outputs"
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取用电: {args.elec}")
    elec = load_elec(args.elec)
    print(f"读取天气: {args.weather}")
    weather = load_weather(args.weather, args.weather_extra)

    paths = split_and_merge(elec, weather, out_dir)
    print(f"\n已生成:")
    for key, p in paths.items():
        print(f"  {key}: {p}")

    report(elec, weather, paths, rep_dir)


if __name__ == "__main__":
    main()
