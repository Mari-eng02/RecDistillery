#!/bin/bash

usage() {
  echo "Usage: bash experiments/baseline/teacher_training.sh [teacher_framework] [teacher_model] <dataset> [gpu] [grid|best]"
  echo
  echo "Examples:"
  echo "  bash experiments/baseline/teacher_training.sh recbole BPRMF citeulike 0 grid"
  echo "  bash experiments/baseline/teacher_training.sh lenskit LGCN amazon_cd 1 best"
  echo "  bash experiments/baseline/teacher_training.sh elliot NMF bookcrossing auto best"
}

if [ "${1}" = "--help" ] || [ "${1}" = "-h" ]; then
  usage
  exit 0
fi

TEACHER_FRAMEWORK="${1:-recbole}"
MODEL="${2:-BPRMF}"
DATASET="${3:-citeulike}"
GPU="${4:-0}"
BEST_FLAG="${5:-grid}"

if [ "${BEST_FLAG}" != "grid" ] && [ "${BEST_FLAG}" != "best" ]; then
  usage
  exit 1
fi

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

MODEL_LC="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
DATASET_LC="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
TEACHER_FRAMEWORK_LC="$(echo "${TEACHER_FRAMEWORK}" | tr '[:upper:]' '[:lower:]')"
MODEL_CONFIG_PATH="config/models/teacher/${TEACHER_FRAMEWORK_LC}/${MODEL_LC}.yaml"

if [ ! -f "${MODEL_CONFIG_PATH}" ]; then
  echo "Teacher model config not found: ${MODEL_CONFIG_PATH}"
  echo "Available configs for ${TEACHER_FRAMEWORK_LC}:"
  find "config/models/teacher/${TEACHER_FRAMEWORK_LC}" -maxdepth 1 -type f -name '*.yaml' -print 2>/dev/null | sort
  exit 1
fi

CMD=(
  python scripts/teacher_training/teacher_training.py
  --framework "${TEACHER_FRAMEWORK_LC}"
  --model "${MODEL}"
  --dataset "${DATASET_LC}"
)

OUT_FILE="${LOG_DIR}/${TEACHER_FRAMEWORK}_${MODEL}_${DATASET}_${BEST_FLAG}.out"
ERR_FILE="${LOG_DIR}/${TEACHER_FRAMEWORK}_${MODEL}_${DATASET}_${BEST_FLAG}.error"

echo "Framework: ${TEACHER_FRAMEWORK}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo "Mode: ${BEST_FLAG}"
echo "Model config: ${MODEL_CONFIG_PATH}"
echo "Logs: ${OUT_FILE} / ${ERR_FILE}"

if [ "${GPU}" = "auto" ]; then
  nohup "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}" &
else
  CUDA_VISIBLE_DEVICES="${GPU}" nohup "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}" &
fi
