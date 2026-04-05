#!/usr/bin/env bash
set -euo pipefail

source /home/amax/ms_temba/data/reinforced_data_1/tune_res_0_1_noted/stageB/best_params.sh

export PATH=/pytorch_env/bin:$PATH

FOLD_JSON_DIR="/home/amax/ms_temba/data/reinforced_data_1/5_fold_json/"
FEATURE_ROOT="/home/amax/ms_temba/data/reinforced_data_1/npy_all/"
OUTPUT_ROOT="/home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1_noted/Atest"

# ===== 从 tune 最佳trial里填入这些值 =====
LR="${LR:-4.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DROP="${DROP:-0.0}"
DROP_PATH="${DROP_PATH:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR="${MIN_LR:-1e-5}"
NUM_CLIPS="${NUM_CLIPS:-256}"
ALPHA_L="${ALPHA_L:-1.0}"
BETA_L="${BETA_L:-0.05}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-/home/amax/ms_temba/MS-Temba-main/vim_with_LLM/insert_llm/qwen2.5_7b/}"
MSTEMBA_GPU_IDS="${MSTEMBA_GPU_IDS:-0}"
LLM_DEVICE="${LLM_DEVICE:-cuda:1}"
LLM_DEVICE_MAP="${LLM_DEVICE_MAP:-auto}"
if [ -z "${LLM_MAX_MEMORY:-}" ]; then
  LLM_MAX_MEMORY='{"0":"2GiB","1":"10GiB","2":"10GiB","cpu":"64GiB"}'
fi
LLM_DTYPE="${LLM_DTYPE:-float16}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
# BATCH_SIZE="${BATCH_SIZE:-5}"
# EPOCHS="${EPOCHS:-50}"

mkdir -p "${OUTPUT_ROOT}"

for FOLD in 1 2 3 4 5
do
  JSON_PATH="${FOLD_JSON_DIR}/fold_${FOLD}.json"
  FOLD_OUTPUT="${OUTPUT_ROOT}/fold_${FOLD}"

  mkdir -p "${FOLD_OUTPUT}"

  echo "=============================="
  echo "Running fold ${FOLD}"
  echo "JSON: ${JSON_PATH}"
  echo "OUTPUT: ${FOLD_OUTPUT}"
  echo "=============================="

  #-epochs "${EPOCHS}" \
  #-batch_size "${BATCH_SIZE}" \

  python MSTemba_main.py \
    -dataset self_def \
    -mode rgb \
    -backbone clip \
    -model mstemba \
    -train True \
    -rgb_root "${FEATURE_ROOT}" \
    -split_json "${JSON_PATH}" \
    -num_clips "${NUM_CLIPS}" \
    -skip 0 \
    -comp_info False \
    -epochs 50 \
    -unisize True \
    -alpha_l "${ALPHA_L}" \
    -beta_l "${BETA_L}" \
    -batch_size 5 \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --drop "${DROP}" \
    --drop-path "${DROP_PATH}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --min-lr "${MIN_LR}" \
    -output_dir "${FOLD_OUTPUT}" \
    --use_llm_refiner \
    --llm_name_or_path "${LLM_MODEL_PATH}" \
    --mstemba_init_ckpt /home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1/fold_4/best_model.pth \
    --freeze_mstemba \
    --gpu_ids "${MSTEMBA_GPU_IDS}" \
    --llm_device "${LLM_DEVICE}" \
    --llm_device_map "${LLM_DEVICE_MAP}" \
    --llm_max_memory "${LLM_MAX_MEMORY}" \
    --llm_torch_dtype "${LLM_DTYPE}" \
    --log_interval "${LOG_INTERVAL}"

  python extract_metrics.py \
    --fold_dir "${FOLD_OUTPUT}" \
    --fold_id "${FOLD}"
done


python extract_metrics.py \
  --root_dir "${OUTPUT_ROOT}" \
  --aggregate_only \
  --out "${OUTPUT_ROOT}/metrics_5fold_summary.json"

python summarize_5fold.py


# run_eval_per_class.sh
python eval_per_class.py \
  --inputs \
  /home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1_noted/Atest/fold_1/eval_arrays.npz \
  /home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1_noted/Atest/fold_2/eval_arrays.npz \
  /home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1_noted/Atest/fold_3/eval_arrays.npz \
  /home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1_noted/Atest/fold_4/eval_arrays.npz \
  /home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1_noted/Atest/fold_5/eval_arrays.npz \
  --out /home/amax/ms_temba/data/reinforced_data_1/output_5fold_tuned_0_1_noted/Atest/per_class_5fold_summary.json

