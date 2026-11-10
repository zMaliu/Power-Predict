# 离线升级 chinese_calendar（局方无外网专用）
#
# 【有外网电脑】下载 wheel：
#   powershell -File scripts/upgrade_calendar_offline.ps1 -Download
#   把生成的 wheels\ 目录拷到局方项目里
#
# 【局方服务器】装进正在运行的容器（不访问外网）：
#   powershell -File scripts/upgrade_calendar_offline.ps1 -Install

param(
    [switch]$Download,
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$WheelDir = Join-Path $Root "wheels"
Set-Location $Root

if (-not $Download -and -not $Install) {
    Write-Host "请指定 -Download（有网机）或 -Install（局方离线机）"
    exit 1
}

if ($Download) {
    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
    Get-ChildItem $WheelDir -Filter "chinese_calendar*.whl" -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Host "下载 chinese_calendar wheel 到 $WheelDir ..."
    python -m pip download "chinese_calendar>=1.9" -d $WheelDir --only-binary=:all: -i https://pypi.tuna.tsinghua.edu.cn/simple
    if (-not (Get-ChildItem $WheelDir -Filter "chinese_calendar*.whl")) {
        # 纯 py 包可能只有 none-any，去掉 only-binary 再试
        python -m pip download "chinese_calendar>=1.9" -d $WheelDir -i https://pypi.tuna.tsinghua.edu.cn/simple
    }
    Write-Host "完成。请将 wheels\ 目录拷到局方服务器的项目目录下。"
    Get-ChildItem $WheelDir | Select-Object Name, Length
}

if ($Install) {
    $whls = @(Get-ChildItem $WheelDir -Filter "chinese_calendar*.whl" -ErrorAction SilentlyContinue)
    if ($whls.Count -eq 0) {
        throw "未找到 wheels\chinese_calendar*.whl，请先在有网机器执行 -Download 并拷贝过来"
    }
    $name = $whls[0].Name
    Write-Host "拷贝 $name 到容器并离线安装..."
    docker compose cp "$WheelDir\$name" "forecast:/tmp/$name"
    docker compose exec -T forecast pip install --no-index --find-links=/tmp "/tmp/$name"
    docker compose restart forecast
    Start-Sleep -Seconds 3
    docker compose exec -T forecast python -c "from chinese_calendar.constants import holidays; ys=sorted({d.year for d in holidays}); print(f'OK years: {ys[0]} .. {ys[-1]}')"
    Write-Host "完成（全程不访问外网）。"
}
