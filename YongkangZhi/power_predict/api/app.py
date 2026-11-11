"""Application factory. Settings and service are injected for isolated testing."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..config import Settings
from ..domain import TARGETS, WEATHER_KEYS
from ..forecasting.service import ForecastService
from .schemas import PredictRequest

logger = logging.getLogger(__name__)
TZ = ZoneInfo("Asia/Shanghai")


def create_app(settings: Settings | None = None, target: str | None = None) -> FastAPI:
    service = ForecastService(settings)
    settings = service.settings
    target = (target if target is not None else os.getenv("FORECAST_TARGET", "all")).strip().lower()
    target = target if target in TARGETS else "all"
    active = [target] if target in TARGETS else list(TARGETS)
    app = FastAPI(title="用电量18天预测", version="2.0.0")
    app.state.forecast_service = service

    def run_prediction(wanted: str, request: PredictRequest) -> dict:
        if not request.weather:
            raise HTTPException(400, "weather不能为空")
        try:
            anchor = date.fromisoformat(request.anchor_date) if request.anchor_date else datetime.now(TZ).date()
            frame = pd.DataFrame([row.model_dump() for row in request.weather])
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, "日期格式须为YYYY-MM-DD") from exc
        frame = frame.sort_values("date").reset_index(drop=True)
        for column in WEATHER_KEYS:
            if column not in frame:
                frame[column] = None
            if column != "weather_type":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if request.strict_dates:
            expected = [anchor + timedelta(days=h) for h in range(1, settings.horizon + 1)]
            if frame["date"].dt.date.tolist() != expected:
                raise HTTPException(400, f"weather日期须为锚定日后连续{settings.horizon}天")
        try:
            result = service.predict(frame, targets=[wanted])
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Prediction failed for %s", wanted)
            raise HTTPException(500, "预测失败") from exc
        forecast = []
        for _, row in result.iterrows():
            item = {"date": row["date"].strftime("%Y-%m-%d")}
            item.update({column: float(row[column]) if pd.notna(row[column]) else None
                         for column in result.columns if column != "date"})
            forecast.append(item)
        return dict(ok=True, target=wanted, target_name=TARGETS[wanted], anchor_date=anchor.isoformat(),
                    forecast_start=forecast[0]["date"], forecast_end=forecast[-1]["date"],
                    horizon=settings.horizon, weather_days_received=len(request.weather),
                    weather_days_filled=max(0, settings.horizon - len(request.weather)), forecast=forecast)

    @app.get("/health")
    def health():
        missing = service.repository.missing(active, settings.horizon)
        missing += [str(settings.data_files[t]) for t in active if not settings.data_files[t].is_file()]
        return dict(ok=not missing, server_today=datetime.now(TZ).date().isoformat(), timezone="Asia/Shanghai",
                    horizon=settings.horizon, forecast_target=target, targets=active,
                    forecast_rule="以weather起始日期预测；strict_dates=true时校验anchor+1至anchor+horizon",
                    endpoints={t: f"POST /predict/{t}" for t in active},
                    strategies={t: settings.predict_strategy.get(t) for t in active}, missing=missing[:20])

    if "industry" in active:
        @app.post("/predict/industry")
        def industry(request: PredictRequest):
            return run_prediction("industry", request)

    if "resident" in active:
        @app.post("/predict/resident")
        def resident(request: PredictRequest):
            return run_prediction("resident", request)

    @app.get("/")
    def root():
        return dict(service="power-forecast", forecast_target=target, docs="/docs", health="/health",
                    endpoints={t: f"POST /predict/{t}" for t in active})

    return app
