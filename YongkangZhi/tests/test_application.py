import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from power_predict.api.app import create_app
from power_predict.config import Settings
from power_predict.data.loaders import load_daily, load_future_weather
from power_predict.data.conversion import convert
from power_predict.domain import TARGETS
from power_predict.features.pipeline import build_feature_frame
from power_predict.forecasting.service import ForecastService, build_merged, predict_file
from power_predict.forecasting.strategies import hybrid_predict, recursive_predict
from power_predict.forecasting.weather import fill_missing_weather
from power_predict.models.metrics import regression_metrics
from power_predict.models.repository import ModelRepository
from power_predict.models.training import train
from tests.asgi_client import request

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_path_resolution_has_no_io_or_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = Settings(project_dir=Path(temporary) / "not-created")
            resolved = cfg.resolve()
            self.assertEqual(resolved, resolved.resolve())
            self.assertFalse(resolved.project_dir.exists())
            self.assertFalse(cfg.model_dir.is_absolute())
            resolved.lgb_params["n_estimators"] = 3
            self.assertEqual(cfg.lgb_params["n_estimators"], 1000)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            Settings(horizon=30).resolve()


class FeatureTests(unittest.TestCase):
    def test_lags_do_not_use_current_target(self):
        frame = load_daily(ROOT / "data/industry_daily.csv", "industry").head(50)
        features = build_feature_frame(frame)
        self.assertEqual(features.iloc[30]["industry_lag1"], frame.iloc[29]["industry"])
        self.assertAlmostEqual(features.iloc[30]["industry_r7"], frame.iloc[23:30]["industry"].mean())

    def test_numeric_metrics(self):
        values = regression_metrics([100, 200], [110, 180])
        self.assertEqual(values["mae"], 15)
        self.assertEqual(values["mape"], 10)
        self.assertEqual(values["n_val"], 2)


class ConversionTests(unittest.TestCase):
    def test_explicit_imports_do_not_replace_existing_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "industry_daily.csv"
            original.write_bytes(b"existing training input")
            electricity = root / "electricity.json"
            weather = root / "weather.json"
            electricity.write_text(json.dumps([
                dict(date="2026-01-01", codeClsVal="MX0000", apTotal=100, sumTotal=130),
                dict(date="2026-01-01", codeClsVal="MY0000", apTotal=30, sumTotal=130),
            ]), encoding="utf-8")
            weather.write_text(json.dumps([dict(ddataTime="2026-01-01T08:00:00", tem=20,
                temMax=24, temMin=16, rhum=80, prediction=0, windSpeed=2, weatherType="cloudy")]), encoding="utf-8")
            first = convert(electricity, weather, None, root)
            second = convert(electricity, weather, None, root)
            self.assertNotEqual(first, second)
            self.assertEqual(original.read_bytes(), b"existing training input")
            self.assertEqual(pd.read_csv(first / "industry_daily.csv").iloc[0]["industry_kwh"], 100)
            self.assertTrue((second / "data_report.json").is_file())


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings().resolve()
        cls.weather = load_future_weather(cls.settings.future_weather_csv)

    def test_both_targets_and_total(self):
        result = ForecastService(self.settings).predict(self.weather, include_total=True)
        self.assertEqual(len(result), 18)
        np.testing.assert_allclose(result["全社会用电"], result["全行业用电"] + result["城乡居民用电"], atol=1e-8)

    def test_seven_day_weather_completion(self):
        history = load_daily(self.settings.data_files["industry"], "industry")
        filled = fill_missing_weather(self.weather.head(7), history, self.settings)
        self.assertEqual(len(filled), 18)
        self.assertEqual(filled["date"].nunique(), 18)
        pd.testing.assert_frame_equal(filled.head(7), self.weather.head(7))

    def test_invalid_weather_and_strategy(self):
        service = ForecastService(self.settings)
        for frame in [pd.DataFrame(), pd.concat([self.weather.head(1)] * 2)]:
            with self.assertRaises(ValueError):
                service.predict(frame)
        with self.assertRaises(ValueError):
            service.predict(self.weather, targets=["unknown"])
        bad = replace(self.settings, predict_strategy={"industry": "invalid"})
        with self.assertRaises(ValueError):
            ForecastService(bad).predict(self.weather, targets=["industry"])

    def test_missing_data_does_not_generate_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = replace(self.settings, data_dir=Path(temporary))
            existing = Path(temporary) / "industry_daily.csv"
            existing.write_bytes(b"business data")
            with self.assertRaises(FileNotFoundError):
                ForecastService(cfg).predict(self.weather, targets=["resident"])
            self.assertEqual(existing.read_bytes(), b"business data")
            self.assertEqual(len(list(Path(temporary).iterdir())), 1)

    def test_repeated_prediction_outputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = replace(self.settings, out_dir=Path(temporary))
            first = predict_file(cfg, targets=["industry"])
            content = first.output_path.read_bytes()
            second = predict_file(cfg, targets=["industry"])
            self.assertNotEqual(first.output_path, second.output_path)
            self.assertEqual(first.output_path.read_bytes(), content)
            pd.testing.assert_frame_equal(first.forecast, second.forecast)

    def test_recursive_log_transform_is_applied_once(self):
        cfg = replace(self.settings, horizon=2)
        history = load_daily(cfg.data_files["resident"], "resident").tail(40)
        merged = build_merged(history, self.weather.head(2), "resident", cfg)
        seen = []

        class ConstantModel:
            def predict(self, frame):
                seen.append(frame.iloc[0]["resident_lag1"])
                return np.array([np.log1p(100)])

        with patch.object(ModelRepository, "load_spec", return_value=dict(cols=["resident_lag1"], log_transform=True)), \
             patch.object(ModelRepository, "load_model", return_value=ConstantModel()):
            recursive_predict("resident", merged, self.weather.head(2)["date"].tolist(), cfg)
        self.assertAlmostEqual(seen[1], np.log1p(100))

    def test_hybrid_rain_without_hot_history_has_defined_baseline(self):
        history = load_daily(self.settings.data_files["resident"], "resident").tail(40).copy()
        history["tem_max"], history["precip"] = 30, 20
        merged = build_merged(history, self.weather, "resident", self.settings)

        class ConstantModel:
            def predict(self, frame):
                return np.array([1000])

        with patch.object(ModelRepository, "load_spec", return_value=dict(cols=["tem_max"], log_transform=False)), \
             patch.object(ModelRepository, "load_model", return_value=ConstantModel()):
            values = hybrid_predict("resident", merged, self.weather["date"].tolist(), self.settings)
        self.assertEqual(len(values), 18)
        self.assertTrue(np.isfinite(values).all())


class TrainingTests(unittest.TestCase):
    def test_repeated_real_training_uses_independent_model_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            for target in TARGETS:
                pd.read_csv(ROOT / f"data/{target}_daily.csv").head(140).to_csv(root / f"data/{target}_daily.csv", index=False)
            params = dict(Settings().lgb_params, n_estimators=4, early_stopping_rounds=2, num_threads=1)
            cfg = Settings(project_dir=root, horizon=2, lgb_params=params)
            first = train(cfg)
            original = first.meta_path.read_bytes()
            second = train(cfg)
            self.assertNotEqual(first.model_dir, second.model_dir)
            self.assertEqual(first.meta_path.read_bytes(), original)
            self.assertEqual(set(first.metrics), set(TARGETS))
            for target in TARGETS:
                self.assertEqual(ModelRepository(second.model_dir).missing([target], 2), [])

    def test_missing_inputs_fail_before_creating_training_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                train(Settings(project_dir=root))
            self.assertFalse((root / "models").exists())


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        weather = pd.read_csv(ROOT / "data/demo_future_weather.csv").head(7).to_dict("records")
        cls.payload = dict(anchor_date="2026-07-31", weather=weather)

    def test_health_and_single_target_isolation(self):
        code, health = request(self.app, "GET", "/health")
        self.assertEqual(code, 200)
        self.assertTrue(health["ok"])
        restricted = create_app(target="industry")
        self.assertEqual(request(restricted, "POST", "/predict/resident", self.payload)[0], 404)

    def test_real_prediction_contract_and_weather_fill(self):
        code, response = request(self.app, "POST", "/predict/resident", self.payload)
        self.assertEqual(code, 200)
        self.assertEqual(len(response["forecast"]), 18)
        self.assertEqual(response["weather_days_filled"], 11)
        self.assertEqual(response["forecast_start"], "2026-08-01")

    def test_invalid_and_strict_requests(self):
        for payload in [dict(weather=[]), dict(self.payload, strict_dates=True), dict(self.payload, anchor_date="invalid")]:
            self.assertEqual(request(self.app, "POST", "/predict/industry", payload)[0], 400)
        self.assertEqual(request(self.app, "POST", "/predict/industry", {})[0], 422)

    def test_unavailable_models_return_503(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = create_app(replace(Settings(), model_dir=Path(temporary)))
            self.assertEqual(request(app, "POST", "/predict/industry", self.payload)[0], 503)


if __name__ == "__main__":
    unittest.main()
