#!/bin/bash

usage() {
  echo "Usage: bash experiments/baseline/teacher_training.sh <model> <dataset> [gpu] [grid|best]"
  echo
  echo "Examples:"
  echo "  bash experiments/baseline/teacher_training.sh BPRMF citeulike 0 grid"
  echo "  bash experiments/baseline/teacher_training.sh NMF citeulike 1 best"
}

if [ "${1}" = "--help" ] || [ "${1}" = "-h" ]; then
  usage
  exit 0
fi

MODEL="${1:-BPRMF}"
DATASET="${2:-citeulike}"
GPU="${3:-0}"
BEST_FLAG="${4:-grid}"

if [ "${BEST_FLAG}" != "grid" ] && [ "${BEST_FLAG}" != "best" ]; then
  usage
  exit 1
fi

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

MODEL_LC="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
DATASET_LC="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
if [ "${BEST_FLAG}" = "best" ]; then
  PRESET_STAGE="best"
  PRESET_FILE="${MODEL}_${DATASET_LC}_best.yaml"
else
  PRESET_STAGE="exploration"
  PRESET_FILE="${MODEL}_${DATASET_LC}_grid.yaml"
fi
PRESET_PATH="config/presets/elliot/teacher/${DATASET_LC}/${MODEL_LC}/${PRESET_STAGE}/${PRESET_FILE}"

if [ ! -f "${PRESET_PATH}" ]; then
  echo "Preset not found: ${PRESET_PATH}"
  exit 1
fi

CMD=(python scripts/teacher_training/teacher_training.py --config "${PRESET_PATH}")

OUT_FILE="${LOG_DIR}/${MODEL}_${DATASET}_${BEST_FLAG}.out"
ERR_FILE="${LOG_DIR}/${MODEL}_${DATASET}_${BEST_FLAG}.error"

echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo "Mode: ${BEST_FLAG}"
echo "Preset: ${PRESET_PATH}"
echo "Logs: ${OUT_FILE} / ${ERR_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" nohup "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}" &
