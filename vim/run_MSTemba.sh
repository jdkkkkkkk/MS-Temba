#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python MSTemba_main.py \
-dataset self_def \
-mode rgb \
-backbone clip \
-model mstemba \
-train True \
-rgb_root /home/amax/ms_temba/data/reinforced_data_1/npy_all/ \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 5 \
-output_dir /home/amax/ms_temba/data/reinforced_data_1/output_res/
