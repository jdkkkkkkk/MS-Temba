#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path
from statistics import mean, pstdev

RE_EPOCH = re.compile(r"Epoch\s+(\d+)\s+-\s+Val MAP:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*Val Loss:\s*([0-9]+(?:\.[0-9]+)?)")
RE_SAMPLED = re.compile(r"Epoch\s+(\d+)\s+-\s+Sampled Val MAP:\s*([0-9]+(?:\.[0-9]+)?)")
RE_BEST_SAVE = re.compile(r"Epoch\s+(\d+),\s+Best Sampled Val Map Update\s+([0-9]+(?:\.[0-9]+)?)")

# 可选：如果你日志里以后打印了 P/R/F1，这里会自动提取
RE_PREC = re.compile(r"precision\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_REC = re.compile(r"recall\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
RE_F1 = re.compile(r"f1\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_training_log(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"training.log not found: {log_path}")

    val_by_epoch = {}       # epoch -> {"val_map": x, "val_loss": y}
    sampled_by_epoch = {}   # epoch -> sampled_map
    best_epoch_from_save = None
    best_sampled_from_save = None

    # 可选统计（如果日志有）
    found_precision = None
    found_recall = None
    found_f1 = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_EPOCH.search(line)
            if m:
                ep = int(m.group(1))
                val_map = float(m.group(2))
                val_loss = float(m.group(3))
                val_by_epoch[ep] = {"val_map": val_map, "val_loss": val_loss}

            m = RE_SAMPLED.search(line)
            if m:
                ep = int(m.group(1))
                sampled_by_epoch[ep] = float(m.group(2))

            m = RE_BEST_SAVE.search(line)
            if m:
                best_epoch_from_save = int(m.group(1))
                best_sampled_from_save = float(m.group(2))

            mp = RE_PREC.search(line)
            if mp:
                found_precision = float(mp.group(1))
            mr = RE_REC.search(line)
            if mr:
                found_recall = float(mr.group(1))
            mf = RE_F1.search(line)
            if mf:
                found_f1 = float(mf.group(1))

    if not val_by_epoch:
        raise RuntimeError(f"No 'Val MAP/Val Loss' found in {log_path}")

    # 选择 best epoch：优先使用“保存best模型”对应epoch；否则按 val_map 最大
    if best_epoch_from_save is not None and best_epoch_from_save in val_by_epoch:
        best_epoch = best_epoch_from_save
    else:
        best_epoch = max(val_by_epoch.keys(), key=lambda e: val_by_epoch[e]["val_map"])

    best_val_map = val_by_epoch[best_epoch]["val_map"]
    best_val_loss = val_by_epoch[best_epoch]["val_loss"]
    best_sampled_map = sampled_by_epoch.get(best_epoch, best_sampled_from_save)

    metrics = {
        "best_epoch": best_epoch,
        "map": best_val_map,
        "best_loss": best_val_loss,
    }

    if best_sampled_map is not None:
        metrics["sampled_map"] = best_sampled_map

    # 可选字段（有就加）
    if found_precision is not None:
        metrics["precision"] = found_precision
    if found_recall is not None:
        metrics["recall"] = found_recall
    if found_f1 is not None:
        metrics["f1"] = found_f1

    return metrics

def save_fold_metrics(fold_dir: Path, fold_id: str = None):
    log_path = fold_dir / "training.log"
    metrics = parse_training_log(log_path)
    if fold_id is not None:
        metrics["fold"] = str(fold_id)

    out_path = fold_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] metrics saved: {out_path}")
    return metrics, out_path

def aggregate_root(root_dir: Path, out_path: Path):
    fold_metrics = []
    for p in sorted(root_dir.glob("fold_*")):
        mpath = p / "metrics.json"
        if not mpath.exists():
            continue
        with mpath.open("r", encoding="utf-8") as f:
            d = json.load(f)
        fold_metrics.append(d)

    if not fold_metrics:
        raise RuntimeError(f"No fold_x/metrics.json found under: {root_dir}")

    def stat(key):
        vals = [float(x[key]) for x in fold_metrics if key in x]
        if not vals:
            return None
        return {"mean": mean(vals), "std": pstdev(vals)}

    summary = {
        "num_folds": len(fold_metrics),
        "folds": fold_metrics,
        "map": stat("map"),
        "best_loss": stat("best_loss"),
        "precision": stat("precision"),
        "recall": stat("recall"),
        "f1": stat("f1"),
        "sampled_map": stat("sampled_map"),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] aggregated summary saved: {out_path}")
    return summary

def main():
    parser = argparse.ArgumentParser("Extract fold metrics from training.log and aggregate")
    parser.add_argument("--fold_dir", type=str, default=None, help="single fold output dir, e.g. .../fold_1")
    parser.add_argument("--fold_id", type=str, default=None, help="fold id tag")
    parser.add_argument("--root_dir", type=str, default=None, help="root dir containing fold_1...fold_5")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--out", type=str, default=None, help="aggregate output path")

    args = parser.parse_args()

    if args.aggregate_only:
        if not args.root_dir:
            raise ValueError("--aggregate_only requires --root_dir")
        root = Path(args.root_dir)
        out = Path(args.out) if args.out else (root / "metrics_5fold_summary.json")
        aggregate_root(root, out)
        return

    if args.fold_dir:
        fold_dir = Path(args.fold_dir)
        save_fold_metrics(fold_dir, args.fold_id)

    if args.root_dir:
        root = Path(args.root_dir)
        out = Path(args.out) if args.out else (root / "metrics_5fold_summary.json")
        aggregate_root(root, out)

if __name__ == "__main__":
    main()
