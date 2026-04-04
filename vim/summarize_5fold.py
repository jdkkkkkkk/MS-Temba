import os
import re
import glob
import statistics

OUTPUT_ROOT = "/home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1"

map_pattern = re.compile(
    r"Epoch\s+(\d+)\s+-\s+Val MAP:\s+([0-9.]+),\s+Val Loss:\s+([0-9.]+)"
)
prf_pattern = re.compile(
    r"Epoch\s+(\d+)\s+-\s+Val Precision:\s+([0-9.]+),\s+Val Recall:\s+([0-9.]+),\s+Val F1:\s+([0-9.]+)"
)

def parse_log(log_path):
    epoch_metrics = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m1 = map_pattern.search(line)
            if m1:
                epoch = int(m1.group(1))
                val_map = float(m1.group(2))
                val_loss = float(m1.group(3))

                if epoch not in epoch_metrics:
                    epoch_metrics[epoch] = {}

                epoch_metrics[epoch]["val_map"] = val_map
                epoch_metrics[epoch]["val_loss"] = val_loss

            m2 = prf_pattern.search(line)
            if m2:
                epoch = int(m2.group(1))
                precision = float(m2.group(2))
                recall = float(m2.group(3))
                f1 = float(m2.group(4))

                if epoch not in epoch_metrics:
                    epoch_metrics[epoch] = {}

                epoch_metrics[epoch]["precision"] = precision
                epoch_metrics[epoch]["recall"] = recall
                epoch_metrics[epoch]["f1"] = f1

    # 只保留 map 和 prf 都完整的 epoch
    valid_epochs = []
    for epoch, metrics in epoch_metrics.items():
        if (
            "val_map" in metrics and
            "val_loss" in metrics and
            "precision" in metrics and
            "recall" in metrics and
            "f1" in metrics
        ):
            valid_epochs.append((epoch, metrics))

    if not valid_epochs:
        return {
            "best_epoch": None,
            "best_map": None,
            "best_loss": None,
            "best_precision": None,
            "best_recall": None,
            "best_f1": None,
        }

    # 按 val_map 最大选；如果 val_map 相同，则选 val_loss 更小的 epoch
    best_epoch, best_metrics = max(
        valid_epochs,
        key=lambda x: (x[1]["val_map"], -x[1]["val_loss"])
    )

    return {
        "best_epoch": best_epoch,
        "best_map": best_metrics["val_map"],
        "best_loss": best_metrics["val_loss"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_f1": best_metrics["f1"],
    }


def main():
    fold_dirs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "fold_*")))

    all_maps = []
    all_losses = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    print("=" * 80)
    print("5-FOLD SUMMARY")
    print("=" * 80)

    for fold_dir in fold_dirs:
        log_path = os.path.join(fold_dir, "training.log")
        if not os.path.exists(log_path):
            print(f"[WARN] Missing log: {log_path}")
            continue

        result = parse_log(log_path)

        print(f"{os.path.basename(fold_dir)}")
        print(f"  best_epoch     : {result['best_epoch']}")
        print(f"  best_map       : {result['best_map']}")
        print(f"  best_loss      : {result['best_loss']}")
        print(f"  best_precision : {result['best_precision']}")
        print(f"  best_recall    : {result['best_recall']}")
        print(f"  best_f1        : {result['best_f1']}")
        print("-" * 80)

        if result["best_map"] is not None:
            all_maps.append(result["best_map"])
        if result["best_loss"] is not None:
            all_losses.append(result["best_loss"])
        if result["best_precision"] is not None:
            all_precisions.append(result["best_precision"])
        if result["best_recall"] is not None:
            all_recalls.append(result["best_recall"])
        if result["best_f1"] is not None:
            all_f1s.append(result["best_f1"])

    print("\nFINAL MEAN RESULTS")
    print("=" * 80)

    if all_maps:
        print(f"mean best_map       : {statistics.mean(all_maps):.4f}")
        print(f"std  best_map       : {statistics.pstdev(all_maps):.4f}")
    if all_losses:
        print(f"mean best_loss      : {statistics.mean(all_losses):.4f}")
        print(f"std  best_loss      : {statistics.pstdev(all_losses):.4f}")
    if all_precisions:
        print(f"mean best_precision : {statistics.mean(all_precisions):.4f}")
        print(f"std  best_precision : {statistics.pstdev(all_precisions):.4f}")
    if all_recalls:
        print(f"mean best_recall    : {statistics.mean(all_recalls):.4f}")
        print(f"std  best_recall    : {statistics.pstdev(all_recalls):.4f}")
    if all_f1s:
        print(f"mean best_f1        : {statistics.mean(all_f1s):.4f}")
        print(f"std  best_f1        : {statistics.pstdev(all_f1s):.4f}")

if __name__ == "__main__":
    main()