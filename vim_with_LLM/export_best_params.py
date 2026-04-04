#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

import optuna


def to_shell_key(k: str) -> str:
    # 比如 weight_decay -> WEIGHT_DECAY
    return k.upper().replace("-", "_")


def main():
    parser = argparse.ArgumentParser("Export best params from Optuna sqlite db")
    parser.add_argument("--sqlite", type=str, required=True,
                        help="Path to optuna sqlite db, e.g. /path/to/optuna.db")
    parser.add_argument("--study_name", type=str, required=True,
                        help="Optuna study name")
    parser.add_argument("--out_json", type=str, default="best_params.json",
                        help="Output json path")
    parser.add_argument("--out_sh", type=str, default="best_params.sh",
                        help="Output shell vars path")
    parser.add_argument("--create_dir", action="store_true",
                        help="Create parent dirs for output files")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_sh = Path(args.out_sh)

    if args.create_dir:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_sh.parent.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{args.sqlite}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    best = study.best_trial
    payload = {
        "study_name": args.study_name,
        "best_value": best.value,
        "best_trial_number": best.number,
        "params": best.params,
        "user_attrs": best.user_attrs,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 同时输出 shell 文件，便于你的 run_5fold_tuned.sh 直接 source
    with out_sh.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f'# Auto-generated from study "{args.study_name}"\n')
        for k, v in best.params.items():
            kk = to_shell_key(k)
            # bash-safe
            if isinstance(v, str):
                f.write(f'{kk}="{v}"\n')
            else:
                f.write(f'{kk}="{v}"\n')

    print(f"[OK] best params json: {out_json}")
    print(f"[OK] best params shell: {out_sh}")
    print(f"[INFO] best trial: #{best.number}, value={best.value}")
    print("[INFO] params:")
    for k, v in best.params.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
