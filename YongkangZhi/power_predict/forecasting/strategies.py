"""Forecast strategies, independent of HTTP and command-line interfaces."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..features.pipeline import build_feature_frame
from ..models.repository import ModelRepository


def recursive_predict(target: str, merged: pd.DataFrame, future_dates: list, cfg) -> list[float]:
    """递归预测：每天预测完后把预测值回填，作为后续日期的 lag 输入。"""
    repository = ModelRepository(cfg.model_dir)
    spec = repository.load_spec(target)
    use_cols = spec["cols"]
    use_log = spec.get("log_transform", False)
    preds = []

    for i, fd in enumerate(future_dates):
        feat = build_feature_frame(merged, lags=cfg.lags, roll_windows=cfg.roll_windows, log_transform_target=target if use_log else None)
        row = feat[feat["date"] == fd]
        if row.empty:
            raise RuntimeError(f"特征帧中未找到日期 {fd}")
        X = row[use_cols].iloc[[0]]

        h = i + 1
        model = repository.load_model(target, h)
        pred = float(model.predict(X)[0])
        if use_log:
            pred = float(np.expm1(pred))
        preds.append(pred)

        # 合并表保存原始尺度；特征流水线在下一轮统一执行一次log1p。
        merged.loc[merged["date"] == fd, target] = pred

    return preds


def direct_predict(target: str, merged: pd.DataFrame, future_dates: list, cfg) -> list[float]:
    """各horizon使用对应未来日期特征，不回填未来目标值。"""
    repository = ModelRepository(cfg.model_dir)
    spec = repository.load_spec(target)
    use_cols = spec["cols"]
    use_log = spec.get("log_transform", False)

    feat = build_feature_frame(merged, lags=cfg.lags, roll_windows=cfg.roll_windows, log_transform_target=target if use_log else None)
    inf = feat[feat["date"].isin(set(future_dates))].sort_values("date").reset_index(drop=True)

    preds = []
    for i in range(len(future_dates)):
        h = i + 1
        model = repository.load_model(target, h)
        pred = float(model.predict(inf.iloc[[i]][use_cols])[0])
        if use_log:
            pred = float(np.expm1(pred))
        preds.append(pred)
    return preds


def hybrid_predict(target: str, merged: pd.DataFrame, future_dates: list, cfg, split_day: int = 7) -> list[float]:
    """
    混合预测：前 split_day 天直接预测（未来目标lag可能缺失），
    后段用「历史同期 + 天气修正」外推。
    外推方式：最近历史同星期均值，经近期校准和天气规则修正。
    """
    repository = ModelRepository(cfg.model_dir)
    spec = repository.load_spec(target)
    use_cols = spec["cols"]
    use_log = spec.get("log_transform", False)

    # 前 split_day 天：直接预测
    feat = build_feature_frame(merged, lags=cfg.lags, roll_windows=cfg.roll_windows, log_transform_target=target if use_log else None)
    inf = feat[feat["date"].isin(set(future_dates))].sort_values("date").reset_index(drop=True)

    preds = []
    for i in range(min(split_day, len(future_dates))):
        h = i + 1
        model = repository.load_model(target, h)
        pred = float(model.predict(inf.iloc[[i]][use_cols])[0])
        if use_log:
            pred = float(np.expm1(pred))
        preds.append(pred)

    # 后段：历史同期 + 天气修正，同时参考前段预测水平
    if len(future_dates) > split_day:
        # 历史数据
        hist_mask = merged["date"] < future_dates[0]
        hist = merged[hist_mask].copy()

        # 计算历史日期的星期、温度
        hist["dow"] = hist["date"].dt.dayofweek
        hist["tem"] = hist["tem"]

        # 历史基准：同星期几的均值
        dow_mean = hist.groupby("dow")[target].mean()
        overall_mean = hist[target].mean()

        # 前段预测均值，作为近期水平参考
        front_mean = np.mean(preds[:split_day])
        # 历史近期（前30天）均值，作为基准校准
        recent_hist_mean = hist.tail(30)[target].mean()
        # 校准系数：如果前段预测比历史近期高，说明近期偏热/偏高，后段也相应上调
        calibration = front_mean / recent_hist_mean if recent_hist_mean > 0 else 1.0
        # 限制校准幅度，避免过度调整
        calibration = np.clip(calibration, 0.9, 1.1)

        # 历史温度-用电弹性：高温日 vs 普通日
        hot_mask = hist["tem_max"] >= 34
        normal_mean = hist[~hot_mask][target].mean()
        if hot_mask.sum() >= 5:
            hot_mean = hist[hot_mask][target].mean()
            heat_boost = hot_mean / normal_mean if normal_mean > 0 else 1.15
        else:
            heat_boost = 1.15  # 默认高温加成 15%

        # 历史暴雨弹性：暴雨日 vs 普通日
        rain_mask = hist["precip"] > 10
        if rain_mask.sum() >= 3:
            rain_mean = hist[rain_mask][target].mean()
            rain_damp = rain_mean / normal_mean if normal_mean > 0 else 0.88
        else:
            rain_damp = 0.88  # 默认暴雨减成 12%

        for i in range(split_day, len(future_dates)):
            fd = future_dates[i]
            dow = pd.Timestamp(fd).dayofweek

            # 基准：同星期几均值，再按近期水平校准
            base = dow_mean.get(dow, overall_mean) * calibration

            # 天气修正：从 inf 中取该日天气
            w = inf[inf["date"] == fd].iloc[0]
            tem_max = w.get("tem_max", 30)
            precip = w.get("precip", 0)

            # 高温修正
            if tem_max >= 34:
                base = base * heat_boost
            elif tem_max >= 32:
                base = base * (1 + (heat_boost - 1) * 0.5)

            # 暴雨修正：考虑连续暴雨的累积效应
            if precip > 10:
                # 检查前一日是否也暴雨
                if i > 0:
                    prev_w = inf[inf["date"] == future_dates[i-1]].iloc[0]
                    prev_precip = prev_w.get("precip", 0)
                    if prev_precip > 10:
                        # 连续暴雨，累积效应更强
                        base = base * rain_damp * 0.95
                    else:
                        base = base * rain_damp
                else:
                    base = base * rain_damp
            elif precip > 5:
                base = base * 0.95  # 中雨轻微下调

            preds.append(base)

    return preds
