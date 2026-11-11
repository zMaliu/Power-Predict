"""Leakage-free fitting: label-date split, inner early stopping, prefix refit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from ..features.aligned import aligned_columns, aligned_frame


@dataclass
class AlignedData:
    frame: pd.DataFrame
    columns: list[str]
    categorical: list[str]
    target: str
    use_log: bool


def prepare(daily, target, horizon, settings):
    frame = aligned_frame(daily, target, horizon, settings)
    columns, categorical = aligned_columns(frame)
    return AlignedData(frame, columns, categorical, target,
                       settings.log_transform.get(target, False))


def split_at(data: AlignedData, cutoff, validation_days: int):
    frame = data.frame
    eligible = frame[data.target].notna() & frame[data.columns].notna().all(axis=1)
    cutoff = pd.Timestamp(cutoff)
    boundary = cutoff - pd.Timedelta(days=validation_days)
    train = frame.loc[eligible & frame["date"].le(boundary)]
    validation = frame.loc[eligible & frame["date"].gt(boundary) & frame["date"].le(cutoff)]
    if len(train) < 30 or len(validation) < 2:
        raise ValueError("历史不足，无法构造内部早停窗口")
    return train, validation


def with_categories(frame, columns, categories):
    result = frame[columns].copy()
    for column, values in categories.items():
        known = result[column].where(result[column].isin(values))
        result[column] = pd.Categorical(known, categories=values)
    return result


def fit_at(data: AlignedData, cutoff, params: dict, validation_days=30):
    train, validation = split_at(data, cutoff, validation_days)
    # Learn categorical vocabularies only from the prefix available at this origin.
    prefix = pd.concat([train, validation])
    categories = {c: sorted(prefix[c].dropna().unique().tolist()) for c in data.categorical}
    params = dict(params)
    patience = params.pop("early_stopping_rounds", None)
    model = LGBMRegressor(**params)
    callbacks = [early_stopping(patience, verbose=False)] if patience else []
    model.fit(with_categories(train, data.columns, categories), train[data.target],
              eval_set=[(with_categories(validation, data.columns, categories), validation[data.target])],
              categorical_feature=data.categorical or "auto", callbacks=callbacks)
    rounds = int(model.best_iteration_ or params.get("n_estimators", 1000))
    final = LGBMRegressor(**dict(params, n_estimators=rounds))
    final.fit(with_categories(prefix, data.columns, categories), prefix[data.target],
              categorical_feature=data.categorical or "auto")
    audit = dict(cutoff=str(pd.Timestamp(cutoff).date()),
                 inner_train_label_max=str(train["date"].max().date()),
                 early_stop_label_min=str(validation["date"].min().date()),
                 final_train_label_max=str(prefix["date"].max().date()),
                 n_train=len(prefix), best_iteration=rounds)
    return final, categories, audit


def predict_rows(data, model, categories, dates):
    rows = data.frame.set_index("date").loc[pd.DatetimeIndex(dates)].reset_index()
    result = model.predict(with_categories(rows, data.columns, categories))
    return np.expm1(result) if data.use_log else result
