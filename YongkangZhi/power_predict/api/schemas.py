"""HTTP-only request models; forecasting does not depend on FastAPI."""

from pydantic import BaseModel, Field


class WeatherDay(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    tem: float | None = None
    tem_max: float | None = None
    tem_min: float | None = None
    rhum: float | None = None
    precip: float | None = None
    wind_speed: float | None = None
    weather_type: str | None = None


class PredictRequest(BaseModel):
    weather: list[WeatherDay]
    anchor_date: str | None = None
    strict_dates: bool = False
