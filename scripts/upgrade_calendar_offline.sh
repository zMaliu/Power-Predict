#!/usr/bin/env bash
# 离线升级 chinese_calendar（局方无外网）
# 有网机：bash scripts/upgrade_calendar_offline.sh download
# 局方机：bash scripts/upgrade_calendar_offline.sh install
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p wheels

cmd="${1:-}"
if [[ "$cmd" == "download" ]]; then
  pip download "chinese_calendar>=1.9" -d wheels -i https://pypi.tuna.tsinghua.edu.cn/simple
  echo "请将 wheels/ 拷到局方项目目录"
  ls -la wheels/chinese_calendar*.whl
elif [[ "$cmd" == "install" ]]; then
  whl=$(ls wheels/chinese_calendar*.whl | head -n 1)
  [[ -n "$whl" ]] || { echo "缺少 wheels/chinese_calendar*.whl"; exit 1; }
  name=$(basename "$whl")
  docker compose cp "$whl" "forecast:/tmp/$name"
  docker compose exec -T forecast pip install --no-index --find-links=/tmp "/tmp/$name"
  docker compose restart forecast
  sleep 3
  docker compose exec -T forecast python -c "from chinese_calendar.constants import holidays; ys=sorted({d.year for d in holidays}); print(f'OK years: {ys[0]} .. {ys[-1]}')"
  echo "完成（不访问外网）"
else
  echo "用法: $0 download|install"
  exit 1
fi
