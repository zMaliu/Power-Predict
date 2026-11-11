"""Read/write the baseline-compatible joblib model bundle format."""

from pathlib import Path
from typing import Any

import joblib


class ModelRepository:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def spec_path(self, target: str) -> Path:
        return self.directory / target / "feature_spec.pkl"

    def model_path(self, target: str, horizon: int) -> Path:
        return self.directory / target / f"h{horizon:02d}.pkl"

    def load_spec(self, target: str) -> dict:
        return joblib.load(self.spec_path(target))

    def load_model(self, target: str, horizon: int) -> Any:
        return joblib.load(self.model_path(target, horizon))

    def save_model(self, target: str, horizon: int, model: Any) -> None:
        self._save(self.model_path(target, horizon), model)

    def save_spec(self, target: str, spec: dict) -> None:
        self._save(self.spec_path(target), spec)

    @staticmethod
    def _save(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            joblib.dump(value, stream)

    def missing(self, targets: list[str], horizon: int) -> list[str]:
        paths = [path for target in targets for path in
                 [self.spec_path(target), *(self.model_path(target, h) for h in range(1, horizon + 1))]]
        return [str(path) for path in paths if not path.is_file()]
