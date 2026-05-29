#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: bash experiments/baseline/teacher_training_cd.sh [gpu]"
  echo
  echo "Runs teacher_training.py sequentially on dataset 'amazon_cd'"
  echo "for the models: BPRMF, LGCN, NMF."
  echo "The launcher detaches automatically, so the terminal can be closed."
  echo
  echo "Examples:"
  echo "  bash experiments/baseline/teacher_training_cd.sh"
  echo "  bash experiments/baseline/teacher_training_cd.sh 0"
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

DATASET="amazon_cd"
MODELS=("BPRMF" "LGCN" "NMF")

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"
RUNNER_LOG="${LOG_DIR}/teacher_training_cd_${DATASET}.log"

if [ "${INTERNAL_RUN}" != "--internal-run" ]; then
  nohup bash experiments/baseline/teacher_training_cd.sh --internal-run "${GPU}" \
    > "${RUNNER_LOG}" 2>&1 < /dev/null &
  PID=$!

  echo "Detached runner started."
  echo "PID: ${PID}"
  echo "Runner log: ${RUNNER_LOG}"
  echo "Per-model logs: ${LOG_DIR}/*_${DATASET}.out and .error"
  exit 0
fi

echo "Teacher training runner started at $(date)"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo

for MODEL in "${MODELS[@]}"; do
  MODEL_LC="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
  DATASET_LC="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
  MODEL_CONFIG_PATH="config/teacher/elliot/${MODEL_LC}.yaml"
  if [ ! -f "${MODEL_CONFIG_PATH}" ]; then
    echo "Teacher model config not found: ${MODEL_CONFIG_PATH}"
    exit 1
  fi
  CMD=(python scripts/teacher_training/teacher_training.py --framework elliot --model "${MODEL}" --dataset "${DATASET_LC}")

  OUT_FILE="${LOG_DIR}/${MODEL}_${DATASET}.out"
  ERR_FILE="${LOG_DIR}/${MODEL}_${DATASET}.error"

  echo "Running model: ${MODEL}"
  echo "Dataset: ${DATASET}"
  echo "GPU: ${GPU}"
  echo "Model config: ${MODEL_CONFIG_PATH}"
  echo "Logs: ${OUT_FILE} / ${ERR_FILE}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}"

  echo "Completed model: ${MODEL}"
  echo
done

echo "Teacher training runner completed at $(date)"
