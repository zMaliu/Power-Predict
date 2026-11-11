"""Run stdlib unittest suite and preserve its log and machine-readable result."""

import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from power_predict.storage import create_run_directory, write_json


if __name__ == "__main__":
    destination = create_run_directory(ROOT / "reports", "tests")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    started = time.perf_counter()
    with (destination / "unittest.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), successful=result.wasSuccessful(),
                  seconds=time.perf_counter() - started)
    write_json(destination / "tests.json", report)
    print(json.dumps(report, indent=2))
    print("Test log:", destination / "unittest.log")
    raise SystemExit(0 if result.wasSuccessful() else 1)
