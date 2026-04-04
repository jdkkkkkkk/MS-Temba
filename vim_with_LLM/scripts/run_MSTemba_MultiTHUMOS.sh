#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python MSTemba_main.py \
-dataset multithumos \
-mode rgb \
-backbone clip/i3d/ \
-model mstemba \
-train True \
-rgb_root /path/to/multithumos_features/ \
-num_clips 800 \
-skip 0 \
--lr 5e-4 \
-comp_info False \
-epochs 100 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 1 \
-output_dir /path/to/output_folder/