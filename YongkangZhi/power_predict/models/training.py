"""Training use case. Preserves baseline labels, split, and final fitting semantics."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..config import Settings
from ..data.loaders import load_daily
from ..domain import TARGETS
from ..features.pipeline import build_feature_frame, feature_columns
from ..storage import create_run_directory, write_json
from .metrics import regression_metrics
from .repository import ModelRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict
    model_dir: Path
    meta_path: Path


def fit_regressor(X_tr, y_tr, X_va, y_va, cat_cols, params):
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=cat_cols or "auto")
    return model


def train_target(target: str, settings: Settings, repository: ModelRepository) -> tuple[dict, dict]:
    csv_path = settings.data_files[target]
    daily = load_daily(csv_path, target)
    use_log = settings.log_transform.get(target, False)
    feat = build_feature_frame(daily, lags=settings.lags, roll_windows=settings.roll_windows,
                               log_transform_target=target if use_log else None)
    cols, cat_cols = feature_columns(feat)
    last_date = daily["date"].max()
    val_cut = last_date - pd.Timedelta(days=settings.val_days)
    metrics = {}
    for h in range(1, settings.horizon + 1):
        d = feat.copy()
        d["y"] = d[target].shift(-h)
        d = d.dropna(subset=["y"]).dropna(subset=cols)
        if len(d) < settings.val_days + 30:
            raise ValueError(f"样本不足: {target}, h={h}, rows={len(d)}")
        tr, va = d[d["date"] <= val_cut], d[d["date"] > val_cut]
        if tr.empty or va.empty:
            raise ValueError(f"训练/验证集为空: {target}, h={h}")
        model = fit_regressor(tr[cols], tr["y"], va[cols], va["y"], cat_cols, dict(settings.lgb_params))
        predicted = model.predict(va[cols])
        actual = va["y"]
        if use_log:
            predicted, actual = np.expm1(predicted), np.expm1(actual)
        metrics[f"h{h}"] = regression_metrics(actual, predicted)
        final = fit_regressor(d[cols], d["y"], va[cols], va["y"], cat_cols,
                              dict(settings.lgb_params, early_stopping_rounds=None))
        repository.save_model(target, h, final)
        logger.info("trained target=%s horizon=%s mape=%.6f", target, h, metrics[f"h{h}"]["mape"])
    repository.save_spec(target, dict(cols=cols, cat_cols=cat_cols, target=target,
                                     data_file=str(csv_path), log_transform=use_log))
    return metrics, dict(data_file=str(csv_path), last_train_date=str(last_date.date()), val_cut=str(val_cut.date()))


def train(settings: Settings | None = None, targets: list[str] | None = None) -> TrainingResult:
    settings = (settings or Settings()).resolve()
    want = list(TARGETS) if targets is None else targets
    if not want or any(target not in TARGETS for target in want):
        raise ValueError("targets必须包含至少一个已知目标")
    for target in want:
        if not settings.data_files[target].is_file():
            raise FileNotFoundError(f"缺少历史数据: {settings.data_files[target]}")
    destination = create_run_directory(settings.training_dir, "train")
    repository = ModelRepository(destination)
    write_json(destination / "settings.json", asdict(settings))
    metrics, metadata = {}, {}
    for target in want:
        metrics[target], metadata[target] = train_target(target, settings, repository)
    meta_path = destination / "meta.json"
    write_json(meta_path, dict(horizon=settings.horizon, targets=metadata, metrics=metrics))
    return TrainingResult(metrics, destination, meta_path)
