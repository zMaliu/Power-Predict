#!/usr/bin/env bash
# 升级 chinese_calendar 并重建预测容器。
# 用法：在项目根目录执行  bash scripts/upgrade_calendar.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] 检查当前节日库覆盖年份（容器内）..."
docker compose exec -T forecast python - <<'PY' || true
import chinese_calendar as c
from chinese_calendar.constants import holidays
ys = sorted({d.year for d in holidays})
print(f"installed chinese_calendar years: {ys[0]} .. {ys[-1]}")
PY

echo "[2/3] 重建镜像（会拉取最新 chinese_calendar）..."
docker compose build --no-cache forecast
docker compose up -d forecast

echo "[3/3] 验证升级结果..."
docker compose exec -T forecast python - <<'PY'
import chinese_calendar as c
from chinese_calendar.constants import holidays
ys = sorted({d.year for d in holidays})
print(f"OK chinese_calendar years: {ys[0]} .. {ys[-1]}")
print("health check: import features")
import features
print("fallback ready:", callable(features._holiday_flags))
PY

echo "完成。若新年份仍未收录，说明上游库尚未发布，预测会自动降级为周末判断。"
