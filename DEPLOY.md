# 用电量 18 天预测 — 部署说明

面向：本机联调、Docker 部署到局方服务器、Dify HTTP 节点调用。

**结论：推荐 Docker。** 镜像内自带 LightGBM，局方机器**不必预装** Python 科学计算栈。

---

## 1. 交付清单

```
model_train/
  api.py                 # HTTP 服务
  config.py / features.py / dataset.py / predict.py
  requirements.txt
  Dockerfile
  docker-compose.yml
  models/
    industry/            # h01~h18.pkl + feature_spec.pkl
    resident/
  data/
    industry_daily.csv   # 历史日电量+天气（持续更新）
    resident_daily.csv
```

训练脚本（`train.py`）局方线上可不部署，换模型时用新 `models/` 覆盖即可。

---

## 2. 本机快速验证（Docker，推荐）

前提：已安装 Docker Desktop，且本目录下已有训练好的 `models/`、`data/`。

> 若 `docker.io` 拉基础镜像失败，Dockerfile 已默认使用 DaoCloud 镜像源：  
> `docker.m.daocloud.io/library/python:3.11-slim-bookworm`  
> 局方网络若可直连 Docker Hub，可改回 `python:3.11-slim-bookworm`。

```bash
cd d:\model_train
docker compose up -d --build
```

检查：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @data/sample_predict_request.json
```

期望：health 为 `"ok": true`；predict 返回 18 天结果。

浏览器打开接口文档：http://127.0.0.1:8000/docs

停止：

```bash
docker compose down
```

### 不用 Docker（仅本机有 Python 3.11/3.12 + LightGBM 时）

> 注意：本仓库开发机若是 Python 3.14，部分包可能装不上；**部署以 Docker（镜像内 Python 3.11）为准**。

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 3. 局方服务器部署

1. 把整个项目目录拷到服务器（或至少：代码 + `models/` + `data/` + Docker 文件）。
2. 安装 Docker / Docker Compose（**不要求**装 LightGBM）。
3. 执行：

```bash
docker compose up -d --build
```

4. 防火墙放行 `8000`（或前面加 Nginx 反代到 80/443）。
5. 用 `/health` 确认 `"ok": true`。

换模型：替换宿主机 `models/` 后执行 `docker compose restart`。  
更新历史：替换/追加 `data/*_daily.csv` 后，下次请求自动读到新数据（只读挂载，写文件在宿主机完成）。

---

## 4. HTTP 接口

### `GET /health`

检查模型文件与历史表是否齐全。

### `POST /predict`

**日期约定（重要）：**

| 概念 | 含义 |
|------|------|
| `anchor_date` | 锚定日，通常=**当天**。请求可不传，服务端用上海时区当天 |
| `weather[].date` | **锚定日之后**连续 18 天（anchor+1 … anchor+18），不是锚定日当天 |
| 历史 CSV | 后端应尽量更新到锚定日（至少到 forecast 前一天），否则 lag 不准 |

例：锚定 2026-09-08，则 weather 日期应为 2026-09-09 … 2026-09-26。

**请求体示例：**

```json
{
  "anchor_date": "2026-09-08",
  "strict_dates": true,
  "targets": ["industry", "resident"],
  "include_total": true,
  "weather": [
    {
      "date": "2026-09-09",
      "tem": 27.3,
      "tem_max": 31.0,
      "tem_min": 28.5,
      "rhum": 84.8,
      "precip": 2.5,
      "wind_speed": 1.8,
      "weather_type": "雷阵雨"
    }
  ]
}
```

注意：`weather` **必须正好 18 天**。默认 `strict_dates=true` 会校验日期是否严格等于锚定后连续 18 天；用旧样例联调时可暂时 `"strict_dates": false`。

**响应字段：**

| 字段 | 说明 |
|------|------|
| `anchor_date` | 本次使用的锚定日 |
| `forecast_start` / `forecast_end` | 预测起止日 |
| `forecast[].date` | 预测日 |
| `forecast[].全行业用电` | 万 kWh |
| `forecast[].城乡居民用电` | 万 kWh |
| `forecast[].全社会用电` | 行业+居民（`include_total=true` 时返回） |

只预测一个目标时，把 `targets` 设为 `["industry"]` 或 `["resident"]`。

### Dify

- 节点类型：HTTP Request  
- 方法：`POST`  
- URL：`http://<服务器IP>:8000/predict`  
- Body：JSON；建议传 `anchor_date=当天`，并由后端填入其后 18 天天气。  
- **不必**给时序模型写提示词；提示词仅用于事后把 JSON 解释成人话（可选）。

---

## 5. 后端注意事项

1. **历史表必须更新到预测起始日前一天**，否则 lag 特征不准。建议每次预测前同步最近 ≥40 天。
2. **未来 18 天天气**由后端拉取，字段名与训练一致（见上表）。缺字段会降低精度。
3. 预测策略已写死在配置：全行业 `recursive`，居民 `hybrid`，不要随意改。
4. 容器内已含 LightGBM；宿主机无此库不影响运行。
5. 轻轨等结构变化后：重新训练 → 替换 `models/` → `restart`，接口不变。
6. 单位：万 kWh / 日；结果已四舍五入到 2 位小数。
7. **节日库升级**：
   - **有外网**：`bash scripts/upgrade_calendar.sh` 或 `powershell -File scripts/upgrade_calendar.ps1`
   - **局方无外网（推荐）**：
     1. 在有网电脑：`powershell -File scripts/upgrade_calendar_offline.ps1 -Download`（或 `bash scripts/upgrade_calendar_offline.sh download`）
     2. 把生成的 `wheels/` 拷到局方项目目录
     3. 在局方：`powershell -File scripts/upgrade_calendar_offline.ps1 -Install`（或 `... offline.sh install`）  
     容器内离线安装，**不访问外网**。若新年份尚未发布，预测仍可运行（周末降级）。
   - **首次部署到无外网局方**：建议在有网机器 `docker compose build` 后执行  
     `docker save power-forecast:1.0 -o power-forecast-1.0.tar`，把 tar + 项目拷到局方再 `docker load -i power-forecast-1.0.tar` 后 `docker compose up -d`。

---

## 6. 依赖说明（镜像内已装）

- Python 3.11  
- lightgbm + libgomp  
- pandas / numpy / scikit-learn / joblib  
- chinese_calendar（节假日特征）  
- fastapi / uvicorn  

本机若不用 Docker，需自行 `pip install -r requirements.txt`，且系统需能装 LightGBM。

---

## 7. 验收标准（部署即用）

- [ ] `GET /health` 返回 `"ok": true`  
- [ ] `POST /predict` 用 18 天天气返回 18 行结果  
- [ ] Dify HTTP 节点能调通并拿到 JSON  
- [ ] 局方服务器**未预装** LightGBM 也能跑（依赖 Docker）  
