"""Start a real local uvicorn process and verify HTTP behavior using urllib."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from power_predict.storage import create_run_directory, write_json


def main():
    destination = create_run_directory(ROOT / "reports", "http")
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
    url = f"http://127.0.0.1:{port}"
    with (destination / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", "power_predict", "serve", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                if process.poll() is not None:
                    raise RuntimeError("HTTP server exited; inspect server.log")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        health = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("HTTP startup timed out")
                    time.sleep(0.1)
            assert health["ok"]
            result = dict(health_ok=True, binding="127.0.0.1", predictions={})
            weather = pd.read_csv(ROOT / "data/demo_future_weather.csv")
            for target in ("industry", "resident"):
                payload = dict(anchor_date="2026-07-31", weather=weather.head(7).to_dict("records"))
                request = urllib.request.Request(url + f"/predict/{target}",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = json.load(response)
                assert body["ok"] and len(body["forecast"]) == 18 and body["weather_days_filled"] == 11
                result["predictions"][target] = dict(status=200, forecast_days=18, weather_days_filled=11)
                write_json(destination / f"response_{target}.json", body)
            write_json(destination / "http.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("HTTP report:", destination / "http.json")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
