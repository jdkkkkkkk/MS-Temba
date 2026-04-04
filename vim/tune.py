#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
import subprocess
from pathlib import Path
from statistics import mean, pstdev

import optuna


# -----------------------------
# 1) 工具函数：解析 training.log（回退方案）
# -----------------------------
VAL_MAP_RE = re.compile(r"Val MAP:\s*([0-9]+(?:\.[0-9]+)?)")
VAL_LOSS_RE = re.compile(r"Val Loss:\s*([0-9]+(?:\.[0-9]+)?)")

def parse_training_log(log_path: Path):
    """
    从 training.log 里提取每个 epoch 的 Val MAP / Val Loss。
    返回:
      {
        "best_map": float,
        "best_loss_at_best_map": float
      }
    若解析失败则抛异常。
    """
    if not log_path.exists():
        raise FileNotFoundError(f"training.log not found: {log_path}")

    maps = []
    losses = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m1 = VAL_MAP_RE.search(line)
            if m1:
                maps.append(float(m1.group(1)))
            m2 = VAL_LOSS_RE.search(line)
            if m2:
                losses.append(float(m2.group(1)))

    if not maps:
        raise RuntimeError(f"No Val MAP found in {log_path}")

    # 这里用 best map 对应的索引去取 loss（若loss数量不足则兜底最小loss）
    best_idx = max(range(len(maps)), key=lambda i: maps[i])
    best_map = maps[best_idx]
    if losses:
        if best_idx < len(losses):
            best_loss = losses[best_idx]
        else:
            best_loss = min(losses)
    else:
        best_loss = 1.0  # 兜底

    return {
        "best_map": best_map,
        "best_loss_at_best_map": best_loss,
    }


# -----------------------------
# 2) 工具函数：解析 metrics.json（优先方案）
# -----------------------------
def parse_metrics_json(metrics_path: Path):
    """
    如果你有外部评估脚本输出 metrics.json（推荐），格式示例:
    {
      "map": 95.1,
      "macro_f1": 87.3,
      "recall": 88.0,
      "precision": 89.2
    }
    """
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return d


# -----------------------------
# 3) 评分函数：多指标综合 + 稳定性惩罚
# -----------------------------
def score_fold(metrics: dict):
    """
    单 fold 打分。
    优先使用 map/macro_f1/recall；缺失时自动降级。
    """
    mAP = float(metrics.get("map", metrics.get("best_map", 0.0)))
    macro_f1 = metrics.get("macro_f1", None)
    recall = metrics.get("recall", None)
    loss = metrics.get("best_loss_at_best_map", None)

    if macro_f1 is not None and recall is not None:
        # 多类别推荐评分
        return 0.5 * mAP + 0.3 * float(macro_f1) + 0.2 * float(recall)

    # 回退：只用 map/loss
    if loss is not None:
        return mAP - 10.0 * float(loss)  # 你可调权重
    return mAP


# -----------------------------
# 4) 运行单个 fold 训练
# -----------------------------
def run_one_fold(
    fold_id: int,
    trial_dir: Path,
    base_cmd: list,
    sampled_params: dict,
    epochs: int
):
    fold_dir = trial_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    cmd = list(base_cmd)

    # 传入本 trial 的超参数
    cmd += [
        "--lr", str(sampled_params["lr"]),
        "--weight-decay", str(sampled_params["weight_decay"]),
        "--drop", str(sampled_params["drop"]),
        "--drop-path", str(sampled_params["drop_path"]),
        "--warmup-epochs", str(sampled_params["warmup_epochs"]),
        "--min-lr", str(sampled_params["min_lr"]),
        "-num_clips", str(sampled_params["num_clips"]),
        "-alpha_l", str(sampled_params["alpha_l"]),
        "-beta_l", str(sampled_params["beta_l"]),
        "-epochs", str(epochs),
        "-output_dir", str(fold_dir),
        # 这里可按你的数据脚本习惯增加 fold 参数，例如:
        # "--fold_id", str(fold_id),
    ]

    # 启动训练
    # stdout/stderr 直接继承，方便你实时看日志
    subprocess.run(cmd, check=True)

    # 优先读取 metrics.json（如果你后处理会产出）
    metrics_json = fold_dir / "metrics.json"
    m = parse_metrics_json(metrics_json)
    if m is not None:
        return m

    # 回退读取 training.log
    log_path = fold_dir / "training.log"
    m = parse_training_log(log_path)
    return m


# -----------------------------
# 5) Optuna objective
# -----------------------------
def build_objective(args):
    base_output = Path(args.study_output)
    base_output.mkdir(parents=True, exist_ok=True)

    # 你的基础命令：可按实际环境改 python 路径
    # 示例基于你当前脚本调用风格
    # 参考: vim/scripts/run_MSTemba_TSU.sh
    # python MSTemba_main.py -dataset tsu -mode rgb ...
    base_cmd = [
        "python", "vim/MSTemba_main.py",
        "-dataset", args.dataset,
        "-mode", args.mode,
        "-backbone", args.backbone,
        "-model", args.model_name,
        "-train", "True",
        "-rgb_root", args.rgb_root,
        "-skip", str(args.skip),
        "-unisize", "True",
        "-batch_size", str(args.batch_size),
    ]

    fold_ids = [int(x) for x in args.folds.split(",")]

    def objective(trial: optuna.Trial):
        # 采样空间（小规模优先）
        sampled = {
            "lr": trial.suggest_float("lr", 1e-5, 8e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-2, log=True),
            "drop": trial.suggest_float("drop", 0.0, 0.3),
            "drop_path": trial.suggest_float("drop_path", 0.0, 0.2),
            "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 8),
            "min_lr": trial.suggest_float("min_lr", 1e-6, 5e-5, log=True),
            "num_clips": trial.suggest_categorical("num_clips", [512, 1024, 1536, 2048, 2500]),
            "alpha_l": trial.suggest_categorical("alpha_l", [0.5, 1.0, 1.5]),
            "beta_l": trial.suggest_categorical("beta_l", [0.0, 0.02, 0.05, 0.1]),
        }

        trial_dir = base_output / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        fold_scores = []
        fold_metrics = []

        for fid in fold_ids:
            try:
                m = run_one_fold(
                    fold_id=fid,
                    trial_dir=trial_dir,
                    base_cmd=base_cmd,
                    sampled_params=sampled,
                    epochs=args.epochs_quick,
                )
                s = score_fold(m)
                fold_scores.append(s)
                fold_metrics.append({"fold": fid, **m, "score": s})
            except Exception as e:
                # 训练失败直接给很差分
                trial.set_user_attr("error_fold", fid)
                trial.set_user_attr("error_msg", str(e))
                return -1e9

        mean_score = mean(fold_scores)
        std_score = pstdev(fold_scores) if len(fold_scores) > 1 else 0.0
        final_score = mean_score - args.std_penalty * std_score

        # 保存 trial 明细
        summary = {
            "trial_number": trial.number,
            "params": sampled,
            "fold_metrics": fold_metrics,
            "mean_score": mean_score,
            "std_score": std_score,
            "final_score": final_score,
        }
        with (trial_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        trial.set_user_attr("mean_score", mean_score)
        trial.set_user_attr("std_score", std_score)
        return final_score

    return objective


def main():
    parser = argparse.ArgumentParser("Small-scale hyperparameter tuning for MS-Temba")
    parser.add_argument("--dataset", type=str, default="tsu")
    parser.add_argument("--mode", type=str, default="rgb")
    parser.add_argument("--backbone", type=str, default="clip")
    parser.add_argument("--model_name", type=str, default="mstemba")
    parser.add_argument("--rgb_root", type=str, required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--folds", type=str, default="1,2", help="quick stage folds, e.g. 1,2")
    parser.add_argument("--epochs_quick", type=int, default=20)
    parser.add_argument("--std_penalty", type=float, default=0.3)

    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--study_name", type=str, default="mstemba_tune_stageA")
    parser.add_argument("--study_output", type=str, default="./tune_runs")
    parser.add_argument("--sqlite", type=str, default="./tune_runs/optuna.db")

    args = parser.parse_args()

    Path(args.study_output).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=f"sqlite:///{args.sqlite}",
        load_if_exists=True,
    )

    objective = build_objective(args)
    study.optimize(objective, n_trials=args.n_trials)

    print("\n===== BEST TRIAL =====")
    print("value:", study.best_value)
    print("params:", study.best_params)


if __name__ == "__main__":
    main()
