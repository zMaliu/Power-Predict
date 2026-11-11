"""Explicit, copyable settings. Resolving paths never creates directories."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .domain import TARGETS


@dataclass(frozen=True)
class Settings:
    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Path("data")
    model_dir: Path = Path("models/baseline")
    training_dir: Path = Path("models/runs")
    out_dir: Path = Path("outputs")
    future_weather_csv: Path | None = None
    lags: tuple[int, ...] = (1, 2, 3, 5, 7, 14, 21)
    roll_windows: tuple[int, ...] = (7, 14)
    horizon: int = 18
    val_days: int = 30
    seed: int = 42
    predict_strategy: dict[str, str] = field(
        default_factory=lambda: {"industry": "recursive", "resident": "hybrid"}
    )
    log_transform: dict[str, bool] = field(
        default_factory=lambda: {"industry": False, "resident": True}
    )
    lgb_params: dict = field(default_factory=lambda: dict(
        objective="regression", metric="l2", learning_rate=0.05, num_leaves=63,
        max_depth=-1, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
        verbose=-1, n_estimators=1000, early_stopping_rounds=100,
    ))

    @property
    def data_files(self) -> dict[str, Path]:
        return {target: self.data_dir / f"{target}_daily.csv" for target in TARGETS}

    def resolve(self) -> Settings:
        base = Path(self.project_dir).resolve()

        def absolute(path: Path) -> Path:
            path = Path(path)
            return path if path.is_absolute() else base / path

        if self.horizon < 1 or self.val_days <= self.horizon:
            raise ValueError("horizon须为正数，val_days须大于horizon以留出验证标签")
        if not self.lags or not self.roll_windows or min(*self.lags, *self.roll_windows) < 1:
            raise ValueError("lags和roll_windows必须是非空正整数序列")
        data_dir = absolute(self.data_dir)
        return replace(
            self, project_dir=base, data_dir=data_dir,
            model_dir=absolute(self.model_dir), training_dir=absolute(self.training_dir),
            out_dir=absolute(self.out_dir),
            future_weather_csv=absolute(self.future_weather_csv) if self.future_weather_csv else data_dir / "demo_future_weather.csv",
            lgb_params=dict(self.lgb_params), predict_strategy=dict(self.predict_strategy),
            log_transform=dict(self.log_transform),
        )
