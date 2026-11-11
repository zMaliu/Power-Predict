"""Regression metrics in the original electricity units."""

import numpy as np
import pandas as pd

from ..domain import TARGETS


def regression_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return dict(
        mae=float(np.mean(np.abs(predicted - actual))),
        mape=float(np.mean(np.abs((predicted - actual) / np.clip(np.abs(actual), 1e-6, None))) * 100),
        rmse=float(np.sqrt(np.mean((predicted - actual) ** 2))),
        n_val=int(len(actual)),
    )


def summarize(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([
        dict(target=t, name=TARGETS[t],
             mae_mean=round(float(np.mean([m["mae"] for m in horizons.values()])), 3),
             mape_mean=round(float(np.mean([m["mape"] for m in horizons.values()])), 3))
        for t, horizons in metrics.items()
    ])
