#!/usr/bin/env bash
set -euo pipefail
export PATH=/pytorch_env/bin:$PATH

mkdir -p /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageA
mkdir -p /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageB

# STAGE A
python tune.py \
  --dataset self_def \
  --mode rgb \
  --backbone clip \
  --model_name mstemba \
  --rgb_root /home/amax/ms_temba/data/reinforced_data_1/npy_all/ \
  --folds 1,2 \
  --epochs_quick 20 \
  --n_trials 30 \
  --study_name mstemba_tune_0_1_noted_stageA \
  --study_output /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageA \
  --sqlite /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageA/optuna.db

python export_best_params.py \
  --sqlite /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageA/optuna.db \
  --study_name mstemba_tune_0_1_noted_stageA \
  --out_json /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageA/best_params.json \
  --out_sh /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageA/best_params.sh \
  --create_dir

# STAGE B
python tune.py \
  --dataset self_def \
  --mode rgb \
  --backbone clip \
  --model_name mstemba \
  --rgb_root /home/amax/ms_temba/data/reinforced_data_1/npy_all/ \
  --folds 1,2,3,4,5 \
  --epochs_quick 50 \
  --n_trials 25 \
  --std_penalty 0.25 \
  --study_name mstemba_tune_0_1_noted_stageB \
  --study_output /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageB \
  --sqlite /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageB/optuna.db

python export_best_params.py \
  --sqlite /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageB/optuna.db \
  --study_name mstemba_tune_0_1_noted_stageB \
  --out_json /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageB/best_params.json \
  --out_sh /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageB/best_params.sh \
  --create_dir

bash run_MSTemba_5_fold_with_tuning.sh
