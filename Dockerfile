# 预测服务：内置 LightGBM，局方机器无需预装
# 默认走 DaoCloud 代理，避免 docker.io 直连失败；也可改回 python:3.11-slim-bookworm
FROM docker.m.daocloud.io/library/python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# 节日库单独一行，便于日后只升级 chinese_calendar
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -U "chinese_calendar>=1.9" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 代码（模型与数据通过 volume 挂载）
COPY config.py dataset.py features.py train.py predict.py api.py ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
