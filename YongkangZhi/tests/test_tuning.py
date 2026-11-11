import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from power_predict.config import Settings
from power_predict.data.loaders import load_daily, load_future_weather
from power_predict.domain import TARGETS
from power_predict.features.aligned import aligned_frame, validate_daily
from power_predict.forecasting.service import ForecastService
from power_predict.models.aligned import fit_at, predict_rows, prepare, split_at
from power_predict.models.repository import ModelRepository
from power_predict.tuning import SearchOptions, make_plan, save_bundle_target, tune

ROOT = Path(__file__).resolve().parents[1]


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings().resolve()
        self.daily = load_daily(self.settings.data_files["industry"], "industry").head(200)

    def test_future_target_mutations_cannot_change_origin_features(self):
        for target in TARGETS:
            original = load_daily(self.settings.data_files[target], target).head(200)
            changed = original.copy()
            changed.loc[changed.index > 150, target] *= 1000
            for h in (1, 7, 18):
                before = aligned_frame(original, target, h, self.settings)
                after = aligned_frame(changed, target, h, self.settings)
                cols = [c for c in before if c.startswith(target + "_")]
                pd.testing.assert_series_equal(before.loc[150 + h, cols], after.loc[150 + h, cols])
                expected = original.loc[150, target]
                if target == "resident":
                    expected = np.log1p(expected)
                self.assertAlmostEqual(before.loc[150 + h, target + "_lag1"], expected)

    def test_label_date_split_and_inner_early_stop_are_before_origin(self):
        for h in (1, 18):
            data = prepare(self.daily, "industry", h, self.settings)
            cutoff = self.daily.loc[150, "date"]
            train, valid = split_at(data, cutoff, 30)
            self.assertLessEqual(train["date"].max(), cutoff - pd.Timedelta(days=30))
            self.assertGreater(valid["date"].min(), train["date"].max())
            self.assertLessEqual(valid["date"].max(), cutoff)
            params = dict(self.settings.lgb_params, n_estimators=4, early_stopping_rounds=2, num_threads=1)
            _, _, audit = fit_at(data, cutoff, params)
            self.assertLessEqual(pd.Timestamp(audit["final_train_label_max"]), cutoff)

    def test_missing_calendar_day_rejected(self):
        with self.assertRaises(ValueError):
            validate_daily(self.daily.drop(index=50), "industry")

    def test_plans_keep_all_development_labels_before_holdout(self):
        daily = load_daily(self.settings.data_files["industry"], "industry")
        plan = make_plan(daily, self.settings, SearchOptions())
        for fold in plan["folds"]:
            self.assertLess(fold["last_prediction"], plan["holdout_start"])
        self.assertEqual(plan["test"]["cutoff"], "2026-05-02")
        self.assertEqual(len(plan["test"]["origins"]), 5)
        self.assertLess(plan["test"]["origins"][-1], plan["holdout_end"])

    def test_invalid_search_plan_rejected(self):
        daily = load_daily(self.settings.data_files["industry"], "industry")
        for options in (SearchOptions(trials=0), SearchOptions(cv_spacing=1),
                        SearchOptions(search_horizons=(19,)), SearchOptions(holdout_days=3)):
            with self.assertRaises(ValueError):
                make_plan(daily, self.settings, options)

    def test_saved_aligned_model_matches_backtest_and_api_service(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = replace(self.settings, horizon=2, model_dir=Path(temporary),
                          lgb_params=dict(self.settings.lgb_params, n_estimators=5,
                                          early_stopping_rounds=2, num_threads=1))
            daily = load_daily(cfg.data_files["resident"], "resident")
            history, future = daily.iloc[:160], daily.iloc[160:162]
            repository = ModelRepository(cfg.model_dir)
            save_bundle_target(history, "resident", cfg, cfg.lgb_params, repository)
            # Feed a temporary input snapshot to the public service, not future actuals.
            inputs = Path(temporary) / "inputs"
            inputs.mkdir()
            history.to_csv(inputs / "resident_daily.csv", index=False)
            cfg = replace(cfg, data_dir=inputs)
            result = ForecastService(cfg).predict(future.drop(columns="resident"), ["resident"])
            for h in (1, 2):
                data = prepare(daily, "resident", h, cfg)
                spec = repository.load_spec("resident")
                expected = predict_rows(data, repository.load_model("resident", h), spec["categories_by_horizon"][str(h)],
                                        [future.iloc[h - 1]["date"]])[0]
                self.assertAlmostEqual(result.iloc[h - 1][TARGETS["resident"]], round(expected, 2))
            with self.assertRaises(ValueError):
                ForecastService(cfg).predict(future.drop(columns="resident").assign(
                    date=lambda f: f["date"] + pd.Timedelta(days=1)), ["resident"])


class SearchIntegrationTests(unittest.TestCase):
    def test_real_search_outputs_selection_before_test_and_isolated_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            daily = load_daily(ROOT / "data/industry_daily.csv", "industry").head(250)
            daily.to_csv(root / "data/industry_daily.csv", index=False)
            cfg = Settings(project_dir=root, horizon=2,
                           lgb_params=dict(Settings().lgb_params, n_estimators=3,
                                           early_stopping_rounds=2, num_threads=1))
            options = SearchOptions(trials=1, folds=2, cv_days=4, cv_spacing=20,
                                    holdout_days=4, search_horizons=(1, 2), max_estimators=3,
                                    compare_legacy=False)
            first = tune(cfg, ["industry"], options)
            original = (first / "report.json").read_bytes()
            second = tune(cfg, ["industry"], options)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "report.json").read_bytes(), original)
            report = json.loads(original)
            self.assertEqual(report["targets"]["industry"]["test"]["candidate"]["n_val"], 4)
            selection = json.loads((first / "industry/selection.json").read_text(encoding="utf-8"))
            self.assertNotIn("test", selection)
            repository = ModelRepository(first / "selected_models")
            self.assertEqual(repository.missing(["industry"], 2), [])


if __name__ == "__main__":
    unittest.main()
