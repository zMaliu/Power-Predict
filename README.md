# 用电量 18 天预测框架（LightGBM）

两个 **独立模型**：**全行业**、**城乡居民** —— 各用 **一张训练表**，各训各的、各部署各的。
**全社会 = 两者相加**，由 **后端** 实现。

## 数据：两张表（给溯源同事）

### 表 1：`industry_daily.csv`（全行业）

| 列 | 说明 |
|---|---|
| date | YYYY-MM-DD |
| industry_kwh | 全行业日用电量（万 kWh） |
| tem / tem_max / tem_min / rhum / precip / … | 当日气象 |

### 表 2：`resident_daily.csv`（城乡居民）

| 列 | 说明 |
|---|---|
| date | YYYY-MM-DD |
| resident_kwh | 城乡居民日用电量（万 kWh） |
| tem / tem_max / tem_min / rhum / precip / … | 当日气象（可与表1相同） |

要求：每张表 **365 行左右、每天一行**；两张表 **日期范围对齐**；气象字段 **口径一致**。

> 若他们列名不同，只改 `config.py` 里的 `COLS` 映射，不必改表结构。

## 目录

```
config.py      两张表路径、列名、训练参数
dataset.py     分别加载 industry / resident 表
features.py    特征（每张表只用自己的目标 lag）
train.py       两张表 -> 两个 model 目录
predict.py     各读各的历史表 + 共用未来18天天气
models/
  industry/    全行业模型包（h01~h18 + feature_spec.pkl）
  resident/    城乡居民模型包
data/
  industry_daily.csv
  resident_daily.csv
  demo_future_weather.csv   # 推理用未来天气样例
```

## 命令

```bash
pip install -r requirements.txt
python train.py      # 读两张表，训两个模型
python predict.py    # 两个模型分别预测（不含全社会加总）
python main.py       # 训练 + 预测
```

## 接入真实数据

1. 把溯源整理好的表放到 `data/industry_daily.csv`、`data/resident_daily.csv`
2. 修改 `config.py` 的 `DATA_FILES`、`COLS`（若列名不一致）
3. `python train.py`

## 部署产物（各拷各的）

```
models/industry/   -> 后端「全行业预测服务」
models/resident/   -> 后端「城乡居民预测服务」
```

每个目录含 18 个 `.pkl` + `feature_spec.pkl`；推理还需对应历史表/库 + 未来天气 + `features.py` 逻辑。

## 待确认

- [ ] 两张表最终列名与单位
- [ ] 未来 18 天天气由后端单独提供（与表中气象列一致）
- [ ] 节假日、代表气象站（后续）

## 迭代记录（R1 → R4）

| 版本 | 改动 | 全行业 18 天 | 居民 18 天 |
|------|------|-------------|-----------|
| R1 | 基础直接预测 | 17.26% | 16.71% |
| R2 | 全行业改递归预测 | **4.08%** | 19.96%（更差） |
| R3 | 增强特征 + log 变换 | **4.08%** | 19.76% |
| R4 | 居民改混合策略：h1~7 直接 + h8~18 历史同期外推（带天气修正+近期校准） | **4.08%** | **7.52%** |

**结论**：全行业用递归预测效果极佳；居民用电波动大，直接预测远期 horizon 失效，改用「历史同期 + 天气修正」外推显著降低误差。
