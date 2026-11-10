"""HTTP 预测服务：供本机调试 / Docker 部署 / Dify HTTP 节点调用。

环境变量 FORECAST_TARGET：
- industry  → 只提供 POST /predict/industry（端口如 8001）
- resident  → 只提供 POST /predict/resident（端口如 8002）
- 不设/all → 两个接口都提供（本机联调）

入参相同：anchor_date（可选）+ weather（任意天数，不足 18 天自动补估）。
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import CFG, TARGETS, WEATHER_KEYS, resolve_paths
from predict import predict_from_weather

FORECAST_TARGET = os.getenv("FORECAST_TARGET", "all").strip().lower()
if FORECAST_TARGET not in ("industry", "resident", "all", ""):
    FORECAST_TARGET = "all"
if FORECAST_TARGET == "":
    FORECAST_TARGET = "all"

app = FastAPI(title="用电量18天预测", version="1.2.0")
CFG_RT = resolve_paths(CFG)
TZ = ZoneInfo("Asia/Shanghai")


class WeatherDay(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD，须为锚定日后的未来日")
    tem: float | None = None
    tem_max: float | None = None
    tem_min: float | None = None
    rhum: float | None = None
    precip: float | None = None
    wind_speed: float | None = None
    weather_type: str | None = None


class PredictRequest(BaseModel):
    weather: list[WeatherDay] = Field(..., description="未来天气，任意天数，不足 18 天自动补估")
    anchor_date: str | None = Field(
        default=None,
        description="锚定日 YYYY-MM-DD（通常=当天）。不传则用服务器上海时区当天。"
        "预测区间为 anchor+1 … anchor+18。",
    )
    strict_dates: bool = Field(
        default=False,
        description="为 true 时校验 weather 日期必须严格等于锚定日后连续 18 天；默认 false 允许缺天",
    )


def _today_cn() -> date:
    return datetime.now(TZ).date()


def _parse_day(s: str, field: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise HTTPException(400, f"{field} 日期格式须为 YYYY-MM-DD，收到: {s!r}") from e


def _weather_to_df(rows: list[WeatherDay]) -> pd.DataFrame:
    data = [r.model_dump() for r in rows]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in WEATHER_KEYS:
        if c not in df.columns:
            df[c] = None
        if c != "weather_type":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def _expected_forecast_dates(anchor: date, horizon: int) -> list[date]:
    return [anchor + timedelta(days=i) for i in range(1, horizon + 1)]


def _active_targets() -> list[str]:
    if FORECAST_TARGET in TARGETS:
        return [FORECAST_TARGET]
    return list(TARGETS.keys())


def _run_predict(target: str, req: PredictRequest) -> dict[str, Any]:
    if target not in _active_targets():
        raise HTTPException(404, f"本实例未启用目标 {target}（FORECAST_TARGET={FORECAST_TARGET}）")

    if not req.weather:
        raise HTTPException(400, "weather 不能为空")

    anchor = _parse_day(req.anchor_date, "anchor_date") if req.anchor_date else _today_cn()
    fw = _weather_to_df(req.weather)
    got = [d.date() for d in fw["date"].tolist()]

    if req.strict_dates:
        expected = _expected_forecast_dates(anchor, CFG_RT.horizon)
        if got != expected:
            raise HTTPException(
                400,
                "weather 日期须为锚定日后连续 "
                f"{CFG_RT.horizon} 天。"
                f" anchor_date={anchor.isoformat()},"
                f" 期望={expected[0].isoformat()}~{expected[-1].isoformat()},"
                f" 实际={got[0].isoformat()}~{got[-1].isoformat()}。",
            )

    try:
        df = predict_from_weather(fw, targets=[target], cfg=CFG_RT, include_total=False)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"预测失败: {e}") from e

    forecast = []
    for _, row in df.iterrows():
        item: dict[str, Any] = {"date": row["date"].strftime("%Y-%m-%d")}
        for col in df.columns:
            if col == "date":
                continue
            item[col] = float(row[col]) if pd.notna(row[col]) else None
        forecast.append(item)

    return {
        "ok": True,
        "target": target,
        "target_name": TARGETS[target],
        "anchor_date": anchor.isoformat(),
        "forecast_start": forecast[0]["date"] if forecast else None,
        "forecast_end": forecast[-1]["date"] if forecast else None,
        "horizon": CFG_RT.horizon,
        "weather_days_received": len(req.weather),
        "weather_days_filled": max(0, CFG_RT.horizon - len(req.weather)),
        "forecast": forecast,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    missing = []
    for t in _active_targets():
        tdir = CFG_RT.model_dir / t
        if not (tdir / "feature_spec.pkl").exists():
            missing.append(f"{t}/feature_spec.pkl")
        for h in range(1, CFG_RT.horizon + 1):
            if not (tdir / f"h{h:02d}.pkl").exists():
                missing.append(f"{t}/h{h:02d}.pkl")
        if not CFG_RT.data_files[t].exists():
            missing.append(str(CFG_RT.data_files[t]))
    ok = not missing
    today = _today_cn()
    return {
        "ok": ok,
        "server_today": today.isoformat(),
        "timezone": "Asia/Shanghai",
        "horizon": CFG_RT.horizon,
        "forecast_target": FORECAST_TARGET,
        "forecast_rule": "以 anchor_date（默认当天）为锚定，预测其后连续 18 天；天气不足自动补估",
        "endpoints": {
            t: f"POST /predict/{t}" for t in _active_targets()
        },
        "targets": _active_targets(),
        "strategies": {t: CFG_RT.predict_strategy.get(t) for t in _active_targets()},
        "missing": missing[:20],
    }


if FORECAST_TARGET in ("industry", "all"):

    @app.post("/predict/industry")
    def predict_industry(req: PredictRequest) -> dict[str, Any]:
        """全行业用电 18 天预测。"""
        return _run_predict("industry", req)


if FORECAST_TARGET in ("resident", "all"):

    @app.post("/predict/resident")
    def predict_resident(req: PredictRequest) -> dict[str, Any]:
        """城乡居民用电 18 天预测。"""
        return _run_predict("resident", req)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "power-forecast",
        "forecast_target": FORECAST_TARGET,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {t: f"POST /predict/{t}" for t in _active_targets()},
    }
