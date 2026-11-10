"""入口：训练 + 预测一步到位。默认用演示数据；完整数据到位后改 config.COLS 和路径即可。"""

from __future__ import annotations

import argparse

from config import CFG, resolve_paths
from predict import predict
from train import train


def run(do_train: bool, do_predict: bool):
    cfg = resolve_paths(CFG)
    if do_train:
        m = train(cfg)
        from train import _summary

        print(_summary(m).to_string(index=False))
    if do_predict:
        df = predict(cfg)
        print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()
    run(not args.skip_train, not args.skip_predict)
