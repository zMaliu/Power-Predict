"""预测：支持两种策略——递归回填 lag / 直接多 horizon 独立预测，按 config 配置。

支持任意天数天气输入：不足 18 天时，缺失日用「历史同星期均值」补估，仍输出完整 18 天。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import CFG, TARGETS, resolve_paths
from dataset import load_daily, load_future_weather, make_demo
from features import build_feature_frame, feature_columns


def _build_merged(history: pd.DataFrame, future_weather: pd.DataFrame, target: str, cfg) -> pd.DataFrame:
    need = max(cfg.lags) + max(cfg.roll_windows) + 5
    hist = history.sort_values("date").tail(need).copy()
    fw = future_weather.copy()
    fw[target] = np.nan
    merged = pd.concat([hist, fw], ignore_index=True).sort_values("date").reset_index(drop=True)
    return merged


def _fill_missing_weather(fw: pd.DataFrame, history: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    把 future_weather 补齐到 cfg.horizon 天。
    缺失日期：按历史同星期几的均值补 tem/tem_max/tem_min/rhum/precip/wind_speed；
    weather_type 用历史同星期众数或默认 '多云'。
    """
    fw = fw.copy()
    fw["date"] = pd.to_datetime(fw["date"]).dt.normalize()
    fw = fw.sort_values("date").reset_index(drop=True)

    if len(fw) >= cfg.horizon:
        return fw.head(cfg.horizon)

    # 需要补的日期：从 fw 最后一天往后续，或从 anchor 后第 1 天开始
    anchor = fw["date"].min() - pd.Timedelta(days=1)
    all_dates = pd.date_range(anchor + pd.Timedelta(days=1), periods=cfg.horizon, freq="D")
    existing = set(fw["date"].tolist())
    missing_dates = [d for d in all_dates if d not in existing]

    if not missing_dates:
        return fw

    # 历史同星期均值
    hist = history.copy()
    hist["dow"] = hist["date"].dt.dayofweek
    num_cols = ["tem", "tem_max", "tem_min", "rhum", "precip", "wind_speed"]
    dow_mean = hist.groupby("dow")[num_cols].mean()
    overall_mean = hist[num_cols].mean()

    # weather_type 众数
    if "weather_type" in hist.columns:
        dow_mode = hist.groupby("dow")["weather_type"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "多云")
    else:
        dow_mode = pd.Series(dtype=str)

    rows = []
    for d in missing_dates:
        dow = d.dayofweek
        row = {"date": d}
        for c in num_cols:
            v = dow_mean.loc[dow, c] if dow in dow_mean.index else overall_mean.get(c, np.nan)
            row[c] = round(float(v), 1) if pd.notna(v) else None
        row["weather_type"] = dow_mode.get(dow, "多云") if len(dow_mode) else "多云"
        rows.append(row)

    fill_df = pd.DataFrame(rows)
    out = pd.concat([fw, fill_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out.head(cfg.horizon)


def _recursive_predict(target: str, merged: pd.DataFrame, future_dates: list, cfg) -> list[float]:
    """递归预测：每天预测完后把预测值回填，作为后续日期的 lag 输入。"""
    tdir = cfg.model_dir / target
    spec = joblib.load(tdir / "feature_spec.pkl")
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
        model = joblib.load(tdir / f"h{h:02d}.pkl")
        pred = float(model.predict(X)[0])
        if use_log:
            pred = float(np.expm1(pred))
        preds.append(pred)

        # 回填时存 log 空间值，保持与训练一致
        merged.loc[merged["date"] == fd, target] = np.log1p(pred) if use_log else pred

    return preds


def _direct_predict(target: str, merged: pd.DataFrame, future_dates: list, cfg) -> list[float]:
    """直接预测：每个 horizon 用同一帧（lag 为空），不回填。"""
    tdir = cfg.model_dir / target
    spec = joblib.load(tdir / "feature_spec.pkl")
    use_cols = spec["cols"]
    use_log = spec.get("log_transform", False)

    feat = build_feature_frame(merged, lags=cfg.lags, roll_windows=cfg.roll_windows, log_transform_target=target if use_log else None)
    inf = feat[feat["date"].isin(set(future_dates))].sort_values("date").reset_index(drop=True)

    preds = []
    for i in range(len(future_dates)):
        h = i + 1
        model = joblib.load(tdir / f"h{h:02d}.pkl")
        pred = float(model.predict(inf.iloc[[i]][use_cols])[0])
        if use_log:
            pred = float(np.expm1(pred))
        preds.append(pred)
    return preds


def _hybrid_predict(target: str, merged: pd.DataFrame, future_dates: list, cfg, split_day: int = 7) -> list[float]:
    """
    混合预测：前 split_day 天直接预测（lag 特征完整），
    后段用「历史同期 + 天气修正」外推。
    外推方式：找历史同星期几、相似天气的日子，取均值作为基准，再根据温度偏离修正。
    """
    tdir = cfg.model_dir / target
    spec = joblib.load(tdir / "feature_spec.pkl")
    use_cols = spec["cols"]
    use_log = spec.get("log_transform", False)

    # 前 split_day 天：直接预测
    feat = build_feature_frame(merged, lags=cfg.lags, roll_windows=cfg.roll_windows, log_transform_target=target if use_log else None)
    inf = feat[feat["date"].isin(set(future_dates))].sort_values("date").reset_index(drop=True)

    preds = []
    for i in range(min(split_day, len(future_dates))):
        h = i + 1
        model = joblib.load(tdir / f"h{h:02d}.pkl")
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
        if hot_mask.sum() >= 5:
            hot_mean = hist[hot_mask][target].mean()
            normal_mean = hist[~hot_mask][target].mean()
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


def predict_from_weather(
    future_weather: pd.DataFrame,
    targets: list[str] | None = None,
    cfg=None,
    include_total: bool = False,
) -> pd.DataFrame:
    """
    核心预测：传入未来天气 DataFrame（须含 date + 气象列），返回预测结果。
    历史用电从 data/*.csv 读取。供 CLI / HTTP / Docker 共用。
    天气不足 horizon 天时，自动用历史同星期均值补齐。
    """
    cfg = resolve_paths(cfg or CFG)
    if not Path(cfg.model_dir).exists():
        raise FileNotFoundError("未找到模型目录，请先运行 train.py")

    fw = future_weather.copy()
    fw["date"] = pd.to_datetime(fw["date"]).dt.normalize()
    fw = fw.sort_values("date").reset_index(drop=True)

    want = targets or list(TARGETS.keys())
    for t in want:
        if t not in TARGETS:
            raise ValueError(f"未知目标: {t}")
        if not cfg.data_files[t].exists():
            raise FileNotFoundError(f"缺少历史表: {cfg.data_files[t]}")

    # 补齐天气到 horizon 天（用第一个目标的历史表做参考）
    ref_history = load_daily(cfg.data_files[want[0]], want[0])
    fw = _fill_missing_weather(fw, ref_history, cfg)

    future_dates = sorted(fw["date"].tolist())
    res = pd.DataFrame({"date": fw["date"]})

    for target in want:
        history = load_daily(cfg.data_files[target], target)
        merged = _build_merged(history, fw, target, cfg)
        strategy = cfg.predict_strategy.get(target, "recursive")

        if strategy == "recursive":
            preds = _recursive_predict(target, merged, future_dates, cfg)
        elif strategy == "hybrid":
            preds = _hybrid_predict(target, merged, future_dates, cfg)
        else:
            preds = _direct_predict(target, merged, future_dates, cfg)

        res[TARGETS[target]] = np.round(preds, 2)

    if include_total and TARGETS["industry"] in res.columns and TARGETS["resident"] in res.columns:
        res["全社会用电"] = np.round(res[TARGETS["industry"]] + res[TARGETS["resident"]], 2)

    return res


def predict(cfg=None, future_weather_csv=None) -> pd.DataFrame:
    cfg = resolve_paths(cfg or CFG)
    if not Path(cfg.model_dir).exists():
        raise FileNotFoundError("未找到模型目录，请先运行 train.py")

    need_demo = any(not p.exists() for p in cfg.data_files.values()) or not cfg.future_weather_csv.exists()
    if need_demo:
        make_demo(cfg)

    fw = load_future_weather(future_weather_csv or cfg.future_weather_csv)
    res = predict_from_weather(fw, cfg=cfg, include_total=False)

    out_path = cfg.out_dir / f"forecast_{fw['date'].min().date()}_{fw['date'].max().date()}.csv"
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已输出: {out_path}")
    return res


if __name__ == "__main__":
    print(predict().to_string(index=False))
