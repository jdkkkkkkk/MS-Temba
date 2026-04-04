#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def best_f1_for_class(y_true_c, y_score_c, num_thr=101):
    # 在 [0,1] 扫阈值，取该类最佳F1
    thrs = np.linspace(0.0, 1.0, num_thr)
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in thrs:
        y_pred = (y_score_c >= t).astype(np.int32)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_c, y_pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)}
    return best


def eval_one_npz(npz_path: Path):
    data = np.load(npz_path)
    y_true = data["y_true"]  # (N, C)
    y_score = data["y_score"]  # (N, C)

    assert y_true.shape == y_score.shape, f"shape mismatch: {y_true.shape} vs {y_score.shape}"
    N, C = y_true.shape

    per_class = []
    ap_list = []
    f1_list = []

    for c in range(C):
        yt = y_true[:, c].astype(np.int32)
        ys = y_score[:, c].astype(np.float32)

        # AP: 若该类全0或全1，average_precision_score可能退化
        if yt.sum() == 0:
            ap = None
        else:
            ap = float(average_precision_score(yt, ys))
            ap_list.append(ap)

        bf = best_f1_for_class(yt, ys, num_thr=201)
        f1_list.append(bf["f1"])

        per_class.append({
            "class_id": c,
            "num_pos": int(yt.sum()),
            "AP": ap,
            "best_threshold": bf["threshold"],
            "best_precision": bf["precision"],
            "best_recall": bf["recall"],
            "best_f1": bf["f1"],
        })

    macro_ap = float(np.mean(ap_list)) if len(ap_list) > 0 else None
    macro_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else None

    return {
        "file": str(npz_path),
        "num_samples": int(N),
        "num_classes": int(C),
        "macro_AP": macro_ap,
        "macro_F1_best_thr": macro_f1,
        "per_class": per_class
    }


def aggregate_fold_results(results):
    # 对多个fold结果做均值
    macro_ap_vals = [r["macro_AP"] for r in results if r["macro_AP"] is not None]
    macro_f1_vals = [r["macro_F1_best_thr"] for r in results if r["macro_F1_best_thr"] is not None]

    out = {
        "num_folds": len(results),
        "macro_AP_mean": float(np.mean(macro_ap_vals)) if macro_ap_vals else None,
        "macro_AP_std": float(np.std(macro_ap_vals)) if macro_ap_vals else None,
        "macro_F1_mean": float(np.mean(macro_f1_vals)) if macro_f1_vals else None,
        "macro_F1_std": float(np.std(macro_f1_vals)) if macro_f1_vals else None,
        "folds": results
    }
    return out


def main():
    parser = argparse.ArgumentParser("Per-class AP/F1 evaluator")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="List of fold npz files, each contains y_true/y_score"
    )
    parser.add_argument("--out", type=str, required=True, help="Output json path")
    args = parser.parse_args()

    results = []
    for p in args.inputs:
        r = eval_one_npz(Path(p))
        results.append(r)

    summary = aggregate_fold_results(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out_path}")
    print(f"macro_AP_mean={summary['macro_AP_mean']}, macro_F1_mean={summary['macro_F1_mean']}")


if __name__ == "__main__":
    main()
