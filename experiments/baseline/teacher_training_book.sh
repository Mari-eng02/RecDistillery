#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: bash experiments/baseline/teacher_training_book.sh [gpu] [grid|best]"
  echo
  echo "Runs teacher_training.py sequentially on dataset 'bookcrossing'"
  echo "for the models: BPRMF, LGCN, NMF."
  echo "The launcher detaches automatically, so the terminal can be closed."
  echo
  echo "Examples:"
  echo "  bash experiments/baseline/teacher_training_book.sh"
  echo "  bash experiments/baseline/teacher_training_book.sh 0 grid"
  echo "  bash experiments/baseline/teacher_training_book.sh 1 best"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

INTERNAL_RUN="${1:-}"

if [ "${INTERNAL_RUN}" = "--internal-run" ]; then
  GPU="${2:-0}"
  BEST_FLAG="${3:-grid}"
else
  GPU="${1:-0}"
  BEST_FLAG="${2:-grid}"
fi

DATASET="bookcrossing"
MODELS=("BPRMF" "LGCN" "NMF")

if [ "${BEST_FLAG}" != "grid" ] && [ "${BEST_FLAG}" != "best" ]; then
  usage
  exit 1
fi

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"
RUNNER_LOG="${LOG_DIR}/teacher_training_book_${DATASET}_${BEST_FLAG}.log"

if [ "${INTERNAL_RUN}" != "--internal-run" ]; then
  nohup bash experiments/baseline/teacher_training_book.sh --internal-run "${GPU}" "${BEST_FLAG}" \
    > "${RUNNER_LOG}" 2>&1 < /dev/null &
  PID=$!

  echo "Detached runner started."
  echo "PID: ${PID}"
  echo "Runner log: ${RUNNER_LOG}"
  echo "Per-model logs: ${LOG_DIR}/*_${DATASET}_${BEST_FLAG}.out and .error"
  exit 0
fi

echo "Teacher training runner started at $(date)"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo "Mode: ${BEST_FLAG}"
echo

for MODEL in "${MODELS[@]}"; do
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

  echo "Running model: ${MODEL}"
  echo "Dataset: ${DATASET}"
  echo "GPU: ${GPU}"
  echo "Mode: ${BEST_FLAG}"
  echo "Preset: ${PRESET_PATH}"
  echo "Logs: ${OUT_FILE} / ${ERR_FILE}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}"

  echo "Completed model: ${MODEL}"
  echo
done

echo "Teacher training runner completed at $(date)"
