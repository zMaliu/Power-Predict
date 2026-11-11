"""Run the tune CLI with a preserved UTF-8 console log (all args are forwarded)."""

import contextlib
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from power_predict.cli import main
from power_predict.storage import create_run_directory, write_json


if __name__ == "__main__":
    destination = create_run_directory(ROOT / "reports", "tuning_console")
    write_json(destination / "invocation.json", dict(argv=sys.argv[1:]))
    print("Console log:", destination / "console.log", flush=True)
    # Old library calls also emit repeated deprecation notices. Numerical/runtime
    # warnings remain visible in the log.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    with (destination / "console.log").open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = main(sys.argv[1:])
    logging.shutdown()
    print("Finished. Console log:", destination / "console.log", flush=True)
    raise SystemExit(code)
