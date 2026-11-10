# 局方服务器部署步骤（你来操作）

> Docker 已装在服务器上。  
> 部署目录按规范：`/opt/apps/<程序名>`  
> **不要**把文件拷进 `/var/lib/docker`。

---

## 一、部署目标

| 服务 | 容器名 | 默认端口 | 接口 |
|------|--------|----------|------|
| 全行业 | `power-forecast-industry` | 8001 | `POST /predict/industry` |
| 城乡居民 | `power-forecast-resident` | 8002 | `POST /predict/resident` |

同一镜像、两个容器；端口若冲突，改 `docker-compose.yml` 左侧数字。

---

## 二、本机先准备好再上传

### 1. 确认本机包完整

在 Windows 的 `d:\model_train` 确认有：

```
api.py / config.py / dataset.py / features.py / predict.py
requirements.txt / Dockerfile / docker-compose.yml
models/industry/  models/resident/
data/industry_daily.csv  data/resident_daily.csv
```

### 2. 打包（本机）

把整个项目打成 zip（可去掉 `outputs/`、`__pycache__`），例如：

`power-forecast-deploy.zip`

### 3. 上传到服务器

用 scp / 堡垒机 / U 盘，把 zip 传到服务器，例如：

```bash
# 示例：上传后放到 /tmp
# 然后在服务器上：
sudo mkdir -p /opt/apps/power-forecast
sudo unzip /tmp/power-forecast-deploy.zip -d /opt/apps/power-forecast
# 若解压多了一层目录，再整理成下面结构
cd /opt/apps/power-forecast
ls
# 应能直接看到 Dockerfile、docker-compose.yml、models、data
```

最终目录建议：

```
/opt/apps/power-forecast/
  Dockerfile
  docker-compose.yml
  api.py ...
  models/
  data/
```

---

## 三、服务器上部署（核心）

```bash
cd /opt/apps/power-forecast

# （可选）改端口：编辑 docker-compose.yml
#   "8001:8000"  → 行业对外端口
#   "8002:8000"  → 居民对外端口

# 构建镜像并启动两个容器
docker compose up -d --build
```

若服务器没有 `docker compose`，试：

```bash
docker-compose up -d --build
```

### 无外网时（镜像在本机构建）

**本机（有网）：**

```bash
cd d:\model_train
docker compose build
docker save power-forecast:1.0 -o power-forecast-1.0.tar
```

把 `power-forecast-1.0.tar` + 项目目录一起拷到服务器 `/opt/apps/power-forecast/`。

**服务器（无外网）：**

```bash
cd /opt/apps/power-forecast
docker load -i power-forecast-1.0.tar
docker compose up -d
# 已有镜像就不必 --build
```

---

## 四、验收

```bash
# 看容器
docker ps | grep power-forecast

# 健康检查
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
```

期望两边 `"ok": true`。

用样例测预测（把样例 JSON 拷到服务器后）：

```bash
curl -X POST http://127.0.0.1:8001/predict/industry \
  -H "Content-Type: application/json" \
  --data-binary @data/sample_predict_request.json

curl -X POST http://127.0.0.1:8002/predict/resident \
  -H "Content-Type: application/json" \
  --data-binary @data/sample_predict_request.json
```

期望：各返回 `ok: true`，`forecast` 长度 18。

---

## 五、常用运维

```bash
cd /opt/apps/power-forecast

# 看日志
docker compose logs -f industry
docker compose logs -f resident

# 重启
docker compose restart

# 停掉
docker compose down

# 更新模型文件后（只换了 models/ 或 data/）
docker compose restart
```

---

## 六、防火墙 / 给 Dify 的地址

1. 让运维放行 **8001、8002**（或你改过的端口）。  
2. Dify HTTP 节点填：

- 行业：`http://<服务器IP>:8001/predict/industry`  
- 居民：`http://<服务器IP>:8002/predict/resident`  

Body 与工作流入参一致：`anchor_date` + `weather`。

---

## 七、对照局方 Docker 说明

| 项 | 路径 / 做法 |
|----|-------------|
| Docker 命令 | `/usr/bin/docker` |
| 建议部署目录 | `/opt/apps/power-forecast` |
| 镜像/容器数据 | 由 Docker 自动写在 `/var/lib/docker`（不要手拷） |
| 正确流程 | 项目放 `/opt/apps/...` → `docker compose build` → 起容器 |

---

## 八、你今天最小操作清单

1. 本机打 zip → 上传服务器  
2. 解压到 `/opt/apps/power-forecast`  
3. `docker compose up -d --build`（无外网则先 `docker load`）  
4. `curl` 两个 `/health` + 一次 `/predict`  
5. 把 IP:端口告诉 Dify / 工作流  

卡住时把 `docker compose logs` 报错发我即可。
