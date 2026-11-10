"""全局配置：列名、路径、训练与预测参数。完整数据到位后只改这里和列映射即可。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ----------------- 两个独立模型（分别训练、分别部署） -----------------
TARGETS: dict[str, str] = {
    "industry": "全行业用电",
    "resident": "城乡居民用电",
}
# 全社会 = 两者相加，由后端实现

HORIZON = 18

# ----------------- 两张训练表路径（key 与 TARGETS 一致） -----------------
DATA_FILES: dict[str, str] = {
    "industry": "data/industry_daily.csv",
    "resident": "data/resident_daily.csv",
}

# ----------------- 列名映射：每张表各自的「日电量」列名可不同 -----------------
COLS: dict[str, str] = {
    "date": "date",
    # 全行业表里的日电量列名
    "industry": "industry_kwh",
    # 城乡居民表里的日电量列名
    "resident": "resident_kwh",
    # 气象列（两张表结构相同即可）
    "tem": "tem",
    "tem_max": "tem_max",
    "tem_min": "tem_min",
    "rhum": "rhum",
    "precip": "precip",
    "wind_speed": "wind_speed",
    "weather_type": "weather_type",
}

WEATHER_KEYS = ("tem", "tem_max", "tem_min", "rhum", "precip", "wind_speed", "weather_type")


@dataclass
class Config:
    project_dir: Path = Path(__file__).resolve().parent
    data_files: dict[str, Path] = field(default_factory=dict)  # 由 resolve_paths 填充
    future_weather_csv: Path | None = None
    model_dir: Path = Path("models")
    out_dir: Path = Path("outputs")

    lags: tuple[int, ...] = (1, 2, 3, 5, 7, 14, 21)
    roll_windows: tuple[int, ...] = (7, 14)
    horizon: int = HORIZON
    val_days: int = 30

    # 预测策略：recursive=递归回填 lag / direct=每 horizon 独立（lag 为空）/ hybrid=前7天直接+后11天趋势外推
    predict_strategy: dict[str, str] = field(
        default_factory=lambda: {
            "industry": "recursive",
            "resident": "hybrid",
        }
    )

    lgb_params: dict = field(
        default_factory=lambda: dict(
            objective="regression",
            metric="l2",
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            n_estimators=1000,
            early_stopping_rounds=100,
        )
    )
    seed: int = 42

    # 对数变换：减少极端值影响，预测时用 expm1 还原
    log_transform: dict[str, bool] = field(
        default_factory=lambda: {
            "industry": False,
            "resident": True,
        }
    )


CFG = Config()


def resolve_paths(cfg: Config = CFG) -> Config:
    base = cfg.project_dir
    cfg.data_files = {}
    for target, rel in DATA_FILES.items():
        p = Path(rel)
        cfg.data_files[target] = p if p.is_absolute() else base / p

    if cfg.future_weather_csv is None:
        cfg.future_weather_csv = base / "data" / "demo_future_weather.csv"
    elif not cfg.future_weather_csv.is_absolute():
        cfg.future_weather_csv = base / cfg.future_weather_csv

    cfg.model_dir = base / cfg.model_dir
    cfg.out_dir = base / cfg.out_dir
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg
