# 升级 chinese_calendar 并重建预测容器（Windows PowerShell）
# 用法：在项目根目录执行  powershell -File scripts/upgrade_calendar.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "[1/3] 检查当前节日库覆盖年份（容器内）..."
docker compose exec -T forecast python -c "from chinese_calendar.constants import holidays; ys=sorted({d.year for d in holidays}); print(f'installed years: {ys[0]} .. {ys[-1]}')"

Write-Host "[2/3] 重建镜像（会拉取最新 chinese_calendar）..."
docker compose build --no-cache forecast
docker compose up -d forecast

Write-Host "[3/3] 验证升级结果..."
docker compose exec -T forecast python -c "from chinese_calendar.constants import holidays; ys=sorted({d.year for d in holidays}); print(f'OK years: {ys[0]} .. {ys[-1]}')"

Write-Host "完成。若新年份仍未收录，说明上游库尚未发布，预测会自动降级为周末判断。"
