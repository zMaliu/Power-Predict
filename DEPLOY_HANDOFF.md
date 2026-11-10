# 给后端的部署说明（双模型 / Docker）

## 一、你怎么打包发给后端

在项目根目录 `model_train` 打 zip，**整包发给对方**即可（不要拆成两个互不相干的工程）。

### 必须带上

```
model_train/
  api.py
  config.py
  dataset.py
  features.py
  predict.py
  train.py                 # 可选，线上可不跑
  requirements.txt
  Dockerfile
  docker-compose.yml
  DEPLOY.md
  DEPLOY_HANDOFF.md        # 本文件
  models/
    industry/              # h01~h18.pkl + feature_spec.pkl
    resident/
  data/
    industry_daily.csv
    resident_daily.csv
    sample_predict_request.json
```

### 可不带 / 可删

- `outputs/`
- `__pycache__/`
- 原始溯源 JSON、临时文件

### 打包命令示例（Windows）

在资源管理器里选中上述内容压缩，或：

```powershell
Compress-Archive -Path api.py,config.py,dataset.py,features.py,predict.py,train.py,requirements.txt,Dockerfile,docker-compose.yml,DEPLOY.md,DEPLOY_HANDOFF.md,models,data -DestinationPath power-forecast-deploy.zip -Force
```

---

## 二、告诉后端：怎么部署

### 核心概念（跟他们说清楚）

1. **只构建一个 Docker 镜像**，里面有同一套代码。  
2. **起两个容器**：一个跑全行业，一个跑城乡居民。  
3. **端口由局方自己定**，改 `docker-compose.yml` 左侧宿主机端口即可；容器内始终是 `8000`。  
4. 两个接口的 **请求 Body 完全一样**（`anchor_date` + `weather`），只是 URL/端口不同。

### 默认端口（可改）

| 服务 | 环境变量 | 默认宿主机端口 | 调用地址 |
|------|----------|----------------|----------|
| 全行业 | `FORECAST_TARGET=industry` | **8001** | `POST http://<服务器IP>:8001/predict/industry` |
| 城乡居民 | `FORECAST_TARGET=resident` | **8002** | `POST http://<服务器IP>:8002/predict/resident` |

若 8001/8002 被占用，例如改成 9001/9002，只改 compose 里：

```yaml
ports:
  - "9001:8000"   # 行业
# ...
ports:
  - "9002:8000"   # 居民
```

### 部署步骤（复制给后端）

```bash
# 1. 解压到服务器某目录，例如 /opt/power-forecast
cd /opt/power-forecast

# 2. （可选）改 docker-compose.yml 里的端口

# 3. 构建并启动两个容器
docker compose up -d --build

# 4. 验收
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
```

期望：两边 `"ok": true`，且分别只列出自己的 endpoint。

### 无外网时

在有网机器先 build，再导出镜像：

```bash
docker compose build
docker save power-forecast:1.0 -o power-forecast-1.0.tar
```

把 `power-forecast-1.0.tar` + 项目目录一起拷到局方，然后：

```bash
docker load -i power-forecast-1.0.tar
docker compose up -d
```

（若已 load 镜像，可不再 `--build`。）

---

## 三、HTTP 怎么调（给 Dify / 工作流）

### 入参（两个节点相同）

```json
{
  "anchor_date": "2026-09-13",
  "weather": [
    {
      "date": "2026-09-14",
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

- `anchor_date`：可选；不传则用服务器当天（上海时区）  
- `weather`：任意天数（如 7 天）；不足 18 天服务端自动补估，仍返回 18 天预测  

### 两个并行 HTTP 节点

| 节点 | URL | 方法 |
|------|-----|------|
| 全行业 | `http://<IP>:<行业端口>/predict/industry` | POST |
| 城乡居民 | `http://<IP>:<居民端口>/predict/resident` | POST |

Header：`Content-Type: application/json`  
Body：同上（两边一样）

### 响应形态

行业：

```json
{
  "ok": true,
  "target": "industry",
  "target_name": "全行业用电",
  "forecast": [
    { "date": "2026-09-14", "全行业用电": 3176.08 }
  ]
}
```

居民：

```json
{
  "ok": true,
  "target": "resident",
  "target_name": "城乡居民用电",
  "forecast": [
    { "date": "2026-09-14", "城乡居民用电": 929.14 }
  ]
}
```

全社会加和：在工作流代码节点里按日期把两列相加，或由后端加总。

---

## 四、运维注意

1. `data/industry_daily.csv`、`data/resident_daily.csv` 需持续更新到预测锚定日前一天。  
2. 换模型：替换对应 `models/industry` 或 `models/resident` 后 `docker compose restart`。  
3. 节假日库无外网时不能自动升级；缺年份会降级为周末判断，预测仍可跑。  

---

## 五、你可以用的话术（发给后端）

> 附件是用电 18 天预测 Docker 部署包。  
> 一个镜像、两个容器：行业和居民分开部署。  
> 端口在 `docker-compose.yml` 里改左侧数字即可，默认 8001（行业）、8002（居民）。  
> 启动：`docker compose up -d --build`，用 `/health` 和样例 JSON 验收。  
> Dify 两个 HTTP 节点并行，Body 相同，只调不同端口/路径；天气可不足 18 天。  
> 详细见包内 `DEPLOY_HANDOFF.md`。
