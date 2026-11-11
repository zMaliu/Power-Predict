# YongkangZhi：Power-Predict 分层重构

这是 `Power-Predict` 的独立重构项目。运行所需的历史数据、未来天气样例和已训练模型已复制到本目录；应用运行时不导入或修改原项目。原目录仅在显式执行对照脚本时作为基准实现使用。

本次目标是规范代码结构、建立可测试边界并验证数值一致性。模型仍是 LightGBM：全行业、城乡居民两个独立目标，各训练18个horizon模型；默认旧模型策略仍为全行业递归预测、居民前7天模型预测加后11天规则外推。现已新增独立的时间对齐自动调参流程，见第8节；它使用明确标记的新模型格式，加载后自动选择对应推理逻辑。

## 1. 目录与层级

```text
YongkangZhi/
├── main.py                       # 薄入口，与 python -m power_predict 等价
├── pyproject.toml                # 包元数据、安装入口、依赖声明
├── requirements.txt              # 兼容依赖范围
├── requirements-tested.txt       # 本次验证的直接依赖版本快照
├── Dockerfile / docker-compose.yml
├── power_predict/                # 可导入的业务包
│   ├── __main__.py / cli.py      # 接入层：解析命令、调用用例、打印结果
│   ├── config.py                # 配置层：Settings、参数校验、无副作用的路径解析
│   ├── domain.py                # 公共契约：目标名称、列映射、天气字段
│   ├── storage.py               # 文件边界：创建独立运行目录、写JSON
│   ├── data/
│   │   ├── loaders.py           # 读取历史CSV、未来天气CSV、天气JSON
│   │   └── conversion.py        # 显式原始JSON导入，生成数据质量报告
│   ├── features/
│   │   ├── calendar.py          # 日期、季节、节假日与调休
│   │   ├── weather.py           # 天气衍生、天气滞后与滚动统计
│   │   ├── target.py            # 用电量lag、滚动均值与周变化
│   │   └── pipeline.py          # 训练/推理共用的特征流水线
│   ├── models/
│   │   ├── repository.py        # 模型文件读写，兼容原joblib格式
│   │   ├── metrics.py           # MAE、MAPE、RMSE及汇总
│   │   └── training.py          # 训练用例、时间切分、最终拟合
│   ├── forecasting/
│   │   ├── weather.py           # 不足18天天气的历史星期均值补齐
│   │   ├── strategies.py        # recursive/direct/hybrid策略
│   │   └── service.py           # 预测用例，供CLI和API共用
│   └── api/
│       ├── schemas.py           # HTTP请求校验模型
│       └── app.py               # FastAPI应用工厂与HTTP状态映射
├── data/                        # 业务输入，不是Python包
│   ├── industry_daily.csv
│   ├── resident_daily.csv
│   ├── demo_future_weather.csv
│   └── sample_predict_request.json
├── models/                      # 模型产物，不是Python包
│   ├── baseline/                # 原项目模型副本，默认推理基准
│   │   ├── industry/            # h01.pkl～h18.pkl、feature_spec.pkl
│   │   ├── resident/
│   │   └── meta.json
│   └── runs/train_<时间>_<唯一ID>/  # 每次训练的新模型与指标
├── outputs/predict_<时间>_<唯一ID>/ # 每次CLI预测结果及所用配置
├── scripts/
│   ├── compare.py               # 原版/重构版特征、预测、API与完整重训对比
│   ├── original_runner.py       # 独立子进程执行原代码，隔离模块与输出
│   ├── run_tests.py             # 执行测试，保存测试日志和JSON结果
│   └── http_smoke.py            # 启动真实uvicorn进程做本地HTTP验证
├── tests/
│   ├── test_application.py      # 配置、特征、预测、训练、API与边界测试
│   └── asgi_client.py           # 标准库ASGI测试客户端，无需httpx/pytest
└── reports/
    ├── source_manifest.json    # 原项目文件哈希快照
    ├── comparison_*/           # 逐阶段对照数据、原版日志、comparison.json
    ├── tests_*/                # unittest.log和tests.json
    └── http_*/                 # 真实HTTP服务日志与响应
```

## 2. 依赖方向和职责

```text
CLI / FastAPI
      │
      ├── 训练用例 models.training
      │       ├── data.loaders -> features.pipeline
      │       ├── LGBMRegressor -> models.metrics
      │       └── models.repository -> 独立模型运行目录
      │
      └── 预测用例 forecasting.service
              ├── data.loaders -> forecasting.weather
              ├── forecasting.strategies -> features.pipeline
              └── models.repository -> 读取指定模型目录

config / domain：共享配置和字段契约
storage：仅在需要保存结果时显式调用
```

接入层不负责特征与模型计算；数据加载器不训练模型；特征函数不读写文件；策略不知道HTTP请求结构；模型仓库集中处理joblib文件。所有应用用例都显式接收 `Settings`，不共享可变 `CFG`。

`Settings.resolve()` 返回新的配置，不修改传入配置，也不创建目录。导入模块、创建API实例和健康检查都不会生成模型/输出目录。缺少历史数据时明确失败，不自动用演示数据替换真实输入。

## 3. 数据、模型和超参数

两张历史表各有2038天数据：2021-01-01～2026-07-31。未来天气样例为2026-08-01～2026-08-18。日电量单位按原CSV口径使用，代码不隐式换算。

模型为 `lightgbm.LGBMRegressor`。默认参数在 `power_predict/config.py` 的 `Settings.lgb_params`：学习率0.05、叶子63、最大深度-1、特征采样0.9、行采样0.8、采样频率5、最多1000轮、验证早停100轮。行业使用原始目标；居民对目标及目标滞后使用log1p，预测后expm1还原。

每个horizon标签为 `y(t+h)`；验证边界是历史末日减30天。先拟合验证模型计算指标，再用全部可用样本拟合部署模型，最终拟合关闭早停。该语义保留原代码，未改成Word文档的比值模型或自动调参。

如需程序中配置，使用独立实例：

```python
from dataclasses import replace
from power_predict.config import Settings
from power_predict.models.training import train

settings = Settings()
settings = replace(settings, lgb_params=dict(settings.lgb_params, num_leaves=31))
result = train(settings)
print(result.model_dir, result.meta_path)
```

## 4. 运行方式

在本目录执行，推荐Python 3.11/3.12：

```powershell
cd C:\projects\YongkangZhi
python -m pip install -r requirements.txt

# 直接使用复制的baseline模型预测，默认输出两个目标
python -m power_predict predict

# 预测并加总全社会
python -m power_predict predict --include-total

# 完整训练，模型保存到新的 models/runs/train_... 目录
python -m power_predict --threads 1 train

# 先训练，再用刚训练的模型预测
python -m power_predict --threads 1 run

# 只训练一个目标；全局选项写在子命令前
python -m power_predict --targets industry --threads 1 train

# 用某次训练的模型预测；不会自动切换到最新目录
python -m power_predict --model-dir models/runs/<实际运行目录名> predict

# 使用其他数据、天气和输出位置
python -m power_predict --data-dir <历史数据目录> --weather <天气CSV> --output-dir <输出目录> predict
```

`python main.py predict` 与 `python -m power_predict predict` 等价。可选执行 `python -m pip install -e .` 后使用 `power-predict predict`；安装为非editable包时，应显式用 `--project-dir` 指向数据/模型所在项目目录，wheel本身不包含业务模型和数据。

训练结果包含 `settings.json`、`meta.json` 和每个目标的模型组；预测结果包含CSV与 `settings.json`，记录所用模型目录。每次运行使用时间戳加唯一ID的新目录，不覆盖 `models/baseline` 或前次运行。失败的训练可能保留诊断文件，但不会自动被选为推理模型。

### 原始JSON转换

```powershell
python -m power_predict convert --electricity <电量JSON> --weather-json <天气JSON>
```

需要原有字段 `date/codeClsVal/apTotal/sumTotal`；天气字段映射见 `data/conversion.py`。转换结果写入独立的 `data/imports/import_.../`，不会覆盖现有CSV。打印目录后，通过 `--data-dir` 显式选择导入结果训练。输入电量单位须由上游确认，转换器保留原值。

## 5. HTTP服务

```powershell
python -m power_predict serve --host 127.0.0.1 --port 8000

# 单独提供居民接口
python -m power_predict serve --target resident --port 8002
```

接口：`GET /health`、`GET /`、`POST /predict/industry`、`POST /predict/resident`。`/docs` 提供自动生成的OpenAPI文档。也支持环境变量 `FORECAST_TARGET=industry/resident/all`，CLI的 `--target` 优先。

请求保持原项目契约：`weather`、可选 `anchor_date`、`strict_dates=False`。天气不足18天会按历史同星期均值/众数补齐。`strict_dates=True` 要求天气严格等于锚点后连续18天。成功响应保持 `ok/target/target_name/anchor_date/horizon/forecast` 等字段；不自动保存HTTP预测JSON。

参数错误返回400、模型不可用返回503、请求结构不合法返回422。行业和居民分别调用，调用方按日期加总全社会。

Docker Compose提供两个本地绑定端口8001/8002，数据和baseline模型只读挂载。新增Docker文件尚未在本次环境构建验证；已经验证的是本地Python与真实uvicorn HTTP运行。

## 6. 重构边界与已知限制

为使比较有意义，默认特征数值、训练标签、验证划分和默认预测策略保持原实现。没有宣称预测准确率提高，也没有实现Word v2.2的“基准水平×预测倍数”。

两项有单独回归测试的边界修复：居民若选择recursive策略，回填保留原始尺度避免下一轮重复log1p；hybrid在高温历史不足但暴雨较多时，确保 `normal_mean` 已定义。这两项不改变已验证默认样例的数值结果。其他工程改进包括拒绝空/重复天气日期、未知策略报错、缺输入明确失败、无全局可变配置、无隐式原始数据路径。

以下局限适用于旧模型与原训练流程；第8节的新时间对齐流程处理前述时点与验证问题，但没有补入业务dateType/remark，也没有历史天气预报版本可供验证：

- 训练是 `X(t) -> y(t+h)`，预测却使用第h个未来日期特征调用h模型，训练/推理时点契约需要统一。
- 验证按特征日期切分，h较大时训练标签可能跨过截止日，不等同严格按锚点截止的样本外回测。
- 预测默认保留最近40行历史，缺自然日时shift按行而非按自然日；没有实际值滞后自适应桥接。
- API默认锚点主要用于响应和可选日期校验，不会自动按锚点截断历史。过去日期回测应另行显式截断数据。
- 居民前7天不回填未来目标，短lag会缺失；后11天使用经验规则，训练指标不包含规则部分的完整策略效果。
- CSV中的业务 `dateType/remark` 仍未进入特征，天气日均温口径仍需在数据侧统一。

## 7. 测试和与原项目的对照

```powershell
python -B scripts/run_tests.py
python -B scripts/http_smoke.py

# 只比较同一批已有模型的特征、18天/7天天气预测和API响应
python -B scripts/compare.py

# 完整对比：另用2038天数据将原版与重构版各重训36个模型
python -B scripts/compare.py --train
```

对照脚本在独立子进程中导入原版，输出全部写在本目录；不会运行原项目的默认main或覆盖原模型。两边使用相同数据、相同参数、单线程执行，最多1000轮；单线程是测试执行设置，不代表改变学习率、叶子数等模型参数。

测试分三类证据：

1. 特征与已有模型等价：两个目标完整历史特征帧逐项相同，分别提供18天和7天天气，比较18天预测与成功API响应。
2. 完整重训等价：两边独立训练36个模型，比较全部horizon指标以及重训后预测，不将旧 `meta.json` 当作此次重训真值。
3. 可运行性：原有18项单元/集成测试，以及新增加的7项时间对齐/Optuna测试；最近完整测试共25项全部通过。真实uvicorn进程的健康检查、两个HTTP接口、7天天气补齐18天也已经验证。

具体运行数值、文件位置和旧保存样例的对比见 [RESULTS.md](RESULTS.md)。所有比较表示实现一致性，不是预测误差为0：未来样例没有对应真实用电量，不能据此计算新的样本外MAPE或验证README的4.08%/7.52%。

## 8. 自动调参与独立测试

新增模块：

```text
power_predict/
├── features/aligned.py       # 统一预测日期和起点可用的电量历史，拒绝自然日缺失
├── models/aligned.py         # 按标签日期切分，内部窗口早停，再用起点之前数据重训
├── forecasting/aligned.py    # 与回测一致的18步直接预测；按模型契约自动分派
└── tuning.py                 # Optuna TPE、开发折、独立测试、原策略对照及模型产物
scripts/
├── tune.py                   # 包装CLI，完整控制台日志写入独立目录
└── export_tuned.py           # 合并开发集已选定的各目标模型并执行预测
tests/test_tuning.py          # 防未来电量泄漏、日期划分、推理一致性和真实搜索测试
```

运行命令（全局选项在`tune`之前）：

```powershell
python -m pip install -r requirements.txt
python -B scripts/tune.py --threads 1 tune --trials 30

# 也可分目标执行；每次自动创建独立报告目录
python -B scripts/tune.py --targets industry --threads 1 tune --trials 30
python -B scripts/tune.py --targets resident --threads 1 tune --trials 30

# 本次只有工业候选通过留出测试；可单独导出工业实验模型
python -B scripts/export_tuned.py reports/tuning_20261111T024121960229Z_2b5a812b

# 加载导出的模型；模型特征契约决定推理逻辑
python -m power_predict --model-dir models/runs/<tuned运行目录> predict --include-total
python -m power_predict --model-dir models/runs/<tuned运行目录> serve
```

参数空间：学习率0.01～0.1对数采样；最大深度取-1/3/4/5/6/7/8；叶子数8～64并受深度约束；叶节点最小样本10～100；特征/行采样0.6～1.0；L1/L2从0、0.001、0.01、0.1、1、5、10取值；采样频率沿用5。候选最多3000轮，默认参数对照最多1000轮，两组都在起点前的30天内部窗口早停100轮，最终重训采用早停选出的轮数。

新契约`anchor_direct_v1`：第h个模型预测日期d的电量，使用d的日历/天气和不晚于起点d-h的电量历史。例如lag1取y(d-h)，lag7取y(d-h-6)，滚动均值截至d-h。训练标签就是y(d)，训练和预测使用同一特征函数；18个步长直接预测，不再套用居民远期规则或递归回填。它是一项独立的建模流程修改，所以结果分别比较原策略、对齐流程默认参数、对齐流程搜索参数。

默认保留历史末90天作独立测试；此前设置3折开发验证，折间隔120天，每折54天、每18天设一个预测起点。每折模型只在折首训练，后续起点可以读取当时已经观测到的真实历史，但不会重新拟合模型。测试阶段同样在测试开始前训练一次，每18天发起下一批预测。训练标签、早停标签全部不晚于训练截止日，所有开发预测标签早于测试期。

搜索以原始电量尺度MAPE为目标，工业/居民独立搜索，每目标各步长共享超参数。为控制计算量，试验先评价h=1/7/14/18（9个开发起点共36点）；完整试验与剪枝数记入报告。TPE前10次随机探索，之后根据历史试验采样，并可对表现差的试验提前剪枝。代理目标最优候选与默认参数再用所有18步、162个开发预测点比较。只依据这次完整开发评价确定`selected`，在打分独立测试前写入`selection.json`；独立测试分数不会反过来更换参数。该流程不保证得到全局最优，也没有用全部步长穷举所有候选。

`reports/tuning_*/`保留`study.sqlite3`、全部试验CSV、实际参数、输入/源代码SHA-256、依赖版本、开发/测试逐日预测和按步长/起点指标、训练时间边界审计、候选选择结果，以及基于开发选择、用完整历史重训的`selected_models/`。它们是试验模型，并不自动替换baseline。旧模型对照也只用测试期前历史重新训练，避免直接加载见过测试期的已有模型。原训练流程内部验证的已知缺陷仍属于该对照臂，最终测试期不参与它的训练。实际对比与是否适合推广见 [TUNING_RESULTS.md](TUNING_RESULTS.md)；此次工业改善、居民退步，因此默认baseline保持原样。

**天气限制：**现有历史数据只有实际观测天气，没有每个起点当时发布的天气预报。此次回测将实际天气视为预报输入，属于已知天气条件下的历史检验，不能解释为实际天气预报条件下的线上精度。模型固定、每18天更新电量历史的评估方式也不等于每天重训。

`train/run`仍是旧训练入口；`predict/serve`根据加载的模型契约分派。默认`models/baseline`不会自动切换到调参产物。实际参数、独立测试结果和是否改善见 [TUNING_RESULTS.md](TUNING_RESULTS.md)。

本次工业实验模型已单独导出到`models/runs/tuned_20261111T025427632490Z_6afcdc3d/`，对应18天预测见[forecast CSV](outputs/predict_20261111T025429559853Z_b4f30ce9/forecast_2026-08-01_2026-08-18.csv)。该模型只含工业目标；居民候选没有改善，所以没有作为调参模型导出。
