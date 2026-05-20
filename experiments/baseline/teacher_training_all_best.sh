#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: bash experiments/baseline/teacher_training_all_best.sh [gpu]"
  echo
  echo "Runs teacher_training.py sequentially in 'best' mode for:"
  echo "  datasets: citeulike, bookcrossing, amazon_cd"
  echo "  models:   BPRMF, LGCN, NMF"
  echo
  echo "The launcher detaches automatically, so the terminal can be closed."
  echo
  echo "Examples:"
  echo "  bash experiments/baseline/teacher_training_all_best.sh"
  echo "  bash experiments/baseline/teacher_training_all_best.sh 0"
  echo "  bash experiments/baseline/teacher_training_all_best.sh 1"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

INTERNAL_RUN="${1:-}"

if [ "${INTERNAL_RUN}" = "--internal-run" ]; then
  GPU="${2:-0}"
else
  GPU="${1:-0}"
fi

DATASETS=("citeulike" "bookcrossing" "amazon_cd")
MODELS=("BPRMF" "LGCN" "NMF")
BEST_FLAG="best"

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"
RUNNER_LOG="${LOG_DIR}/teacher_training_all_best.log"

if [ "${INTERNAL_RUN}" != "--internal-run" ]; then
  nohup bash experiments/baseline/teacher_training_all_best.sh --internal-run "${GPU}" \
    > "${RUNNER_LOG}" 2>&1 < /dev/null &
  PID=$!

  echo "Detached runner started."
  echo "PID: ${PID}"
  echo "Runner log: ${RUNNER_LOG}"
  echo "Per-model logs: ${LOG_DIR}/*_*_${BEST_FLAG}.out and .error"
  exit 0
fi

echo "Teacher training all-best runner started at $(date)"
echo "GPU: ${GPU}"
echo "Mode: ${BEST_FLAG}"
echo

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    MODEL_LC="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
    DATASET_LC="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
    PRESET_PATH="config/presets/elliot/teacher/${DATASET_LC}/${MODEL_LC}/best/${MODEL}_${DATASET_LC}_best.yaml"
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
    echo "Dataset: ${DATASET}"
    echo
  done
done

echo "Teacher training all-best runner completed at $(date)"
