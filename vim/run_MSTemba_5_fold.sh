#!/usr/bin/env bash

export PATH=/pytorch_env/bin:$PATH

FOLD_JSON_DIR="/home/amax/ms_temba/data/reinforced_data_1/5_fold_json/"
FEATURE_ROOT="/home/amax/ms_temba/data/reinforced_data_1/npy_all/"
OUTPUT_ROOT="/home/amax/ms_temba/data/reinforced_data_1/output_5fold"

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

  python MSTemba_main.py \
    -dataset self_def \
    -mode rgb \
    -backbone clip \
    -model mstemba \
    -train True \
    -rgb_root "${FEATURE_ROOT}" \
    -split_json "${JSON_PATH}" \
    -num_clips 256 \
    -skip 0 \
    -comp_info False \
    -epochs 50 \
    -unisize True \
    -alpha_l 1 \
    -beta_l 0.05 \
    -batch_size 5 \
    -output_dir "${FOLD_OUTPUT}"
done

python summarize_5fold.py
