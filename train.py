"""训练：两张表分别训两个模型，各 18 个 horizon。"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import CFG, TARGETS, resolve_paths
from dataset import load_daily, make_demo
from features import build_feature_frame, feature_columns


def _fit_one(X_tr, y_tr, X_va, y_va, cat_cols, params):
    model = LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=cat_cols or "auto",
    )
    return model


def train_one_target(target: str, cfg) -> tuple[dict, str, str]:
    """单表 -> 单模型目录。"""
    csv_path = cfg.data_files[target]
    if not csv_path.exists():
        make_demo(cfg)

    daily = load_daily(csv_path, target)
    use_log = cfg.log_transform.get(target, False)
    feat = build_feature_frame(daily, lags=cfg.lags, roll_windows=cfg.roll_windows, log_transform_target=target if use_log else None)
    cols, cat_cols = feature_columns(feat)

    last_date = daily["date"].max()
    val_cut = last_date - pd.Timedelta(days=cfg.val_days)
    tdir = cfg.model_dir / target
    tdir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict] = {}

    for h in range(1, cfg.horizon + 1):
        d = feat.copy()
        d["y"] = d[target].shift(-h)
        d = d.dropna(subset=["y"])
        d = d.dropna(subset=[c for c in cols if c in d.columns])
        if len(d) < cfg.val_days + 30:
            raise RuntimeError(f"样本不足（{target} h={h} 只有 {len(d)} 行）")

        tr = d[d["date"] <= val_cut]
        va = d[d["date"] > val_cut]
        model = _fit_one(tr[cols], tr["y"], va[cols], va["y"], cat_cols, dict(cfg.lgb_params))

        pred = model.predict(va[cols])
        # 还原到原始尺度计算指标
        if use_log:
            pred_orig = np.expm1(pred)
            y_orig = np.expm1(va["y"])
        else:
            pred_orig = pred
            y_orig = va["y"]

        metrics[f"h{h}"] = dict(
            mae=float(np.mean(np.abs(pred_orig - y_orig))),
            mape=float(np.mean(np.abs((pred_orig - y_orig) / np.clip(np.abs(y_orig), 1e-6, None))) * 100),
            rmse=float(np.sqrt(np.mean((pred_orig - y_orig) ** 2))),
            n_val=int(len(va)),
        )

        final = _fit_one(
            d[cols], d["y"], va[cols], va["y"], cat_cols, dict(cfg.lgb_params, early_stopping_rounds=None)
        )
        joblib.dump(final, tdir / f"h{h:02d}.pkl")

    joblib.dump(dict(cols=cols, cat_cols=cat_cols, target=target, data_file=str(csv_path), log_transform=use_log), tdir / "feature_spec.pkl")
    return metrics, str(last_date.date()), str(val_cut.date())


def train(cfg=None) -> dict:
    cfg = resolve_paths(cfg or CFG)
    all_metrics: dict[str, dict] = {}
    meta_targets = {}

    for target in TARGETS:
        m, last, val_cut = train_one_target(target, cfg)
        all_metrics[target] = m
        meta_targets[target] = dict(
            data_file=str(cfg.data_files[target]),
            last_train_date=last,
            val_cut=val_cut,
        )

    (cfg.model_dir / "meta.json").write_text(
        json.dumps(dict(horizon=cfg.horizon, targets=meta_targets, metrics=all_metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return all_metrics


def _summary(metrics: dict) -> pd.DataFrame:
    rows = []
    for t, hs in metrics.items():
        ma = np.mean([v["mae"] for v in hs.values()])
        mp = np.mean([v["mape"] for v in hs.values()])
        rows.append(dict(target=t, name=TARGETS[t], mae_mean=round(ma, 3), mape_mean=round(mp, 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    m = train()
    print(_summary(m).to_string(index=False))
    print(f"\n模型已保存到: {resolve_paths().model_dir}")
