"""Reproducible Optuna search with seasonal development folds and a held-out tail.

The legacy pipeline is an explicit comparison arm. It is never a search objective.
No holdout score participates in parameter or model selection.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .data.loaders import load_daily
from .domain import TARGETS
from .features.aligned import CONTRACT, validate_daily
from .models.aligned import fit_at, predict_rows, prepare
from .models.metrics import regression_metrics
from .models.repository import ModelRepository
from .storage import create_run_directory, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchOptions:
    trials: int = 30
    folds: int = 3
    cv_days: int = 54
    cv_spacing: int = 120
    holdout_days: int = 90
    search_horizons: tuple[int, ...] = (1, 7, 14, 18)
    max_estimators: int = 3000
    compare_legacy: bool = True


def make_plan(daily, settings, options):
    if options.trials < 1 or options.folds < 2 or options.max_estimators < 1:
        raise ValueError("trials/max_estimators须为正数，folds至少为2")
    if options.holdout_days < settings.horizon or options.cv_days < settings.horizon:
        raise ValueError("开发和测试窗口至少覆盖完整预测长度")
    if options.cv_spacing < options.cv_days:
        raise ValueError("开发折不能重叠：cv_spacing须不小于cv_days")
    if not options.search_horizons or any(h < 1 or h > settings.horizon for h in options.search_horizons):
        raise ValueError("搜索步长须位于1..horizon")
    test_start = len(daily) - options.holdout_days
    latest_cutoff = test_start - options.cv_days - 1
    cuts = [latest_cutoff - options.cv_spacing * i for i in reversed(range(options.folds))]
    if cuts[0] < settings.val_days + settings.horizon + 65:
        raise ValueError("历史不足，无法构造指定的开发折和独立测试集")
    dates = daily["date"]
    folds = []
    for cut in cuts:
        origins = list(range(cut, cut + options.cv_days - settings.horizon + 1, settings.horizon))
        folds.append(dict(cutoff=str(dates.iloc[cut].date()),
                          origins=[str(dates.iloc[i].date()) for i in origins],
                          last_prediction=str(dates.iloc[origins[-1] + settings.horizon].date())))
    test_origins = list(range(test_start - 1, len(daily) - 1, settings.horizon))
    return dict(folds=folds, test=dict(cutoff=str(dates.iloc[test_start - 1].date()),
                origins=[str(dates.iloc[i].date()) for i in test_origins],
                last_prediction=str(dates.iloc[-1].date())),
                holdout_start=str(dates.iloc[test_start].date()),
                holdout_end=str(dates.iloc[-1].date()))


def summarize_rows(rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    result = regression_metrics(frame["actual"], frame["prediction"])
    result["per_horizon"] = {str(h): regression_metrics(group["actual"], group["prediction"])
                             for h, group in frame.groupby("horizon")}
    result["per_origin"] = {str(origin): regression_metrics(group["actual"], group["prediction"])
                            for origin, group in frame.groupby("origin")}
    return result


def evaluate(daily, target, settings, cache, params, fold, horizons):
    cutoff = pd.Timestamp(fold["cutoff"])
    indexed = daily.set_index("date")[target]
    rows, audits = [], {}
    for h in horizons:
        data = cache[h]
        model, categories, audit = fit_at(data, cutoff, params, settings.val_days)
        audits[str(h)] = audit
        origins = [pd.Timestamp(d) for d in fold["origins"]]
        valid = [(o, o + pd.Timedelta(days=h)) for o in origins
                 if o + pd.Timedelta(days=h) <= indexed.index.max()]
        predictions = predict_rows(data, model, categories, [d for _, d in valid])
        for (origin, date), prediction in zip(valid, predictions):
            rows.append(dict(target=target, origin=str(origin.date()), date=str(date.date()),
                             horizon=h, actual=float(indexed.loc[date]), prediction=float(prediction)))
    return rows, audits


def evaluate_cv(daily, target, settings, cache, params, folds, horizons, trial=None):
    rows, audits = [], []
    for i, fold in enumerate(folds):
        values, audit = evaluate(daily, target, settings, cache, params, fold, horizons)
        rows.extend(values)
        audits.append(audit)
        if trial is not None:
            trial.report(summarize_rows(rows)["mape"], i)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    return summarize_rows(rows), rows, audits


def sample_params(trial, base, options):
    depth = trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 7, 8])
    leaves = trial.suggest_int("num_leaves", 8, 64 if depth == -1 else min(64, 2 ** depth))
    return dict(base, learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=leaves, max_depth=depth,
                min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
                feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
                reg_alpha=trial.suggest_categorical("reg_alpha", [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]),
                reg_lambda=trial.suggest_categorical("reg_lambda", [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]),
                n_estimators=options.max_estimators)


def legacy_test(daily, target, settings, fold, destination):
    """Retrain legacy models using only the prefix before the held-out period."""
    from .models.training import train_target
    from .forecasting.service import STRATEGIES, build_merged
    cutoff = pd.Timestamp(fold["cutoff"])
    source = destination / "legacy_data"
    source.mkdir(exist_ok=True)
    daily.loc[daily["date"].le(cutoff)].to_csv(source / f"{target}_daily.csv", index=False)
    repository = ModelRepository(destination / "legacy_models")
    cfg = replace(settings, data_dir=source, model_dir=repository.directory)
    train_target(target, cfg, repository)
    rows = []
    strategy = STRATEGIES[cfg.predict_strategy[target]]
    for value in fold["origins"]:
        origin = pd.Timestamp(value)
        history = daily.loc[daily["date"].le(origin)]
        future = daily.loc[daily["date"].gt(origin)].head(settings.horizon)
        # Incomplete final batches are omitted for all arms by using complete-day plans.
        weather = future.drop(columns=[target])
        merged = build_merged(history, weather, target, cfg)
        predictions = strategy(target, merged, weather["date"].tolist(), cfg)
        for h, ((_, row), prediction) in enumerate(zip(future.iterrows(), predictions), 1):
            rows.append(dict(target=target, origin=str(origin.date()), date=str(row["date"].date()),
                             horizon=h, actual=float(row[target]), prediction=float(prediction)))
    return summarize_rows(rows), rows


def save_bundle_target(daily, target, settings, params, repository):
    audits = {}
    categories = {}
    for h in range(1, settings.horizon + 1):
        data = prepare(daily, target, h, settings)
        model, vocab, audit = fit_at(data, daily["date"].iloc[-1], params, settings.val_days)
        repository.save_model(target, h, model)
        audits[str(h)] = audit
        categories[str(h)] = vocab
    repository.save_spec(target, dict(contract=CONTRACT, cols=data.columns,
                         cat_cols=data.categorical, categories_by_horizon=categories,
                         target=target, log_transform=data.use_log, params=params,
                         horizon=settings.horizon, lags=list(settings.lags),
                         roll_windows=list(settings.roll_windows)))
    return audits


def tune(settings: Settings | None = None, targets=None, options: SearchOptions | None = None) -> Path:
    import optuna

    settings = (settings or Settings()).resolve()
    options = options or SearchOptions()
    targets = list(TARGETS) if targets is None else targets
    if not targets or len(set(targets)) != len(targets) or any(t not in TARGETS for t in targets):
        raise ValueError("目标须为不重复的已知目标")
    histories, plans = {}, {}
    for target in targets:
        daily = load_daily(settings.data_files[target], target)
        validate_daily(daily, target)
        histories[target], plans[target] = daily, make_plan(daily, settings, options)
    destination = create_run_directory(settings.project_dir / "reports", "tuning")
    started = time.perf_counter()
    write_json(destination / "settings.json", dict(settings=asdict(settings), search=asdict(options), plans=plans))
    sources = {str(settings.data_files[t]): hashlib.sha256(settings.data_files[t].read_bytes()).hexdigest()
               for t in targets}
    sources.update({str(p.relative_to(settings.project_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (settings.project_dir / "power_predict").rglob("*.py")})
    write_json(destination / "provenance.json", dict(sources=sources, python=sys.version,
               platform=platform.platform(), packages={p: version(p) for p in
               ("lightgbm", "optuna", "pandas", "numpy", "scikit-learn", "chinese-calendar")}))
    repository = ModelRepository(destination / "selected_models")
    report = dict(contract=CONTRACT, selection_rule="full-horizon development MAPE only; holdout never selects",
                  weather_assumption="realized historical weather as forecast proxy; not operational weather skill",
                  targets={}, plans=plans, model_dir=str(repository.directory))
    all_horizons = tuple(range(1, settings.horizon + 1))
    for target in targets:
        target_started = time.perf_counter()
        daily, plan = histories[target], plans[target]
        out = destination / target
        out.mkdir()
        # Searching never even receives the held-out rows.
        development = daily.loc[daily["date"].lt(pd.Timestamp(plan["holdout_start"]))].copy()
        cache = {h: prepare(development, target, h, settings) for h in all_horizons}
        base = dict(settings.lgb_params, random_state=settings.seed,
                    num_threads=settings.lgb_params.get("num_threads", 1), deterministic=True,
                    force_col_wise=True)
        storage = optuna.storages.RDBStorage("sqlite:///" + (out / "study.sqlite3").as_posix())
        study = optuna.create_study(study_name=target, direction="minimize", storage=storage,
                    sampler=optuna.samplers.TPESampler(seed=settings.seed, n_startup_trials=10),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
        def objective(trial):
            params = sample_params(trial, base, options)
            trial.set_user_attr("effective_params", params)
            metrics, _, _ = evaluate_cv(development, target, settings, cache, params,
                                        plan["folds"], options.search_horizons, trial)
            return metrics["mape"]
        logger.info("search target=%s trials=%d output=%s", target, options.trials, out)
        try:
            study.optimize(objective, n_trials=options.trials, n_jobs=1)
            study.trials_dataframe().to_csv(out / "trials.csv", index=False)
            best_trial, trials = study.best_trial, study.trials
        finally:
            storage.remove_session()
            storage.engine.dispose()
        candidate = best_trial.user_attrs["effective_params"]
        cv = {}
        for label, params in (("default", base), ("candidate", candidate)):
            logger.info("full development evaluation target=%s arm=%s", target, label)
            metrics, rows, audits = evaluate_cv(development, target, settings, cache, params,
                                                plan["folds"], all_horizons)
            cv[label] = metrics
            pd.DataFrame(rows).to_csv(out / f"development_{label}.csv", index=False)
            write_json(out / f"development_{label}_audit.json", audits)
        selected = "candidate" if cv["candidate"]["mape"] < cv["default"]["mape"] else "default"
        selected_params = candidate if selected == "candidate" else base
        # Persist the decision BEFORE any held-out observations are scored.
        write_json(out / "selection.json", dict(selected=selected, params=selected_params,
                   cv=cv, best_trial=best_trial.number, search_score=best_trial.value))
        test_cache = {h: prepare(daily, target, h, settings) for h in all_horizons}
        test = {}
        for label, params in (("default", base), ("candidate", candidate)):
            logger.info("held-out test target=%s arm=%s", target, label)
            rows, audit = evaluate(daily, target, settings, test_cache, params, plan["test"], all_horizons)
            test[label] = summarize_rows(rows)
            pd.DataFrame(rows).sort_values(["origin", "horizon"]).to_csv(out / f"test_{label}.csv", index=False)
            write_json(out / f"test_{label}_audit.json", audit)
        if options.compare_legacy:
            logger.info("held-out test target=%s arm=legacy", target)
            metrics, rows = legacy_test(daily, target, settings, plan["test"], out)
            test["legacy"] = metrics
            pd.DataFrame(rows).to_csv(out / "test_legacy.csv", index=False)
        test["selected"] = test[selected]
        target_report = dict(cv=cv, test=test, selected=selected, best_trial=best_trial.number,
                             best_search_mape=best_trial.value, default_params=base, candidate_params=candidate,
                             completed_trials=sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials),
                             pruned_trials=sum(t.state == optuna.trial.TrialState.PRUNED for t in trials),
                             tuning_mape_change_pp=test["default"]["mape"] - test["candidate"]["mape"],
                             tuning_relative_improvement_pct=100 * (1 - test["candidate"]["mape"] / test["default"]["mape"]))
        logger.info("final refit target=%s selected=%s", target, selected)
        target_report["deployment_audit"] = save_bundle_target(daily, target, settings, selected_params, repository)
        target_report["seconds"] = time.perf_counter() - target_started
        report["targets"][target] = target_report
        write_json(out / "result.json", target_report)
        logger.info("result target=%s default=%.4f candidate=%.4f selected=%s", target,
                    test["default"]["mape"], test["candidate"]["mape"], selected)
    report["seconds"] = time.perf_counter() - started
    write_json(repository.directory / "meta.json", dict(contract=CONTRACT, report=str(destination / "report.json"),
               horizon=settings.horizon, targets=targets, settings=asdict(settings)))
    write_json(destination / "report.json", report)
    return destination
