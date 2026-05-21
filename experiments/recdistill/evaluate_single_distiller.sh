#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ "$#" -ne 3 ]; then

  echo "Usage: $0 <distiller> <dataset> <model>"

  exit 1

fi

DISTILLER="$1"

DATASET="$2"

MODEL="$3"
DISTILLER_LC="$(echo "${DISTILLER}" | tr '[:upper:]' '[:lower:]')"
DATASET_LC="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
MODEL_LC="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"

EMBEDDING_DIM=20

TOP_K=20
CONFIG_PATH="config/presets/recdistill/final_rerun/${DISTILLER_LC}/${MODEL_LC}/${DATASET_LC}/best.yaml"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "============================================================"

echo "[START] Student evaluation"

echo "Timestamp: $(date)"

echo "Distiller: ${DISTILLER}"

echo "Dataset:   ${DATASET}"

echo "Model:     ${MODEL}"

echo "Emb dim:   ${EMBEDDING_DIM}"

echo "Top-k:     ${TOP_K}"
echo "Config:    ${CONFIG_PATH}"

echo "============================================================"

echo "[1/1] Evaluating distilled student artifact..."

python3 scripts/recdistill/evaluate_students.py \
  --dataset "${DATASET_LC}" \
  --distiller "${DISTILLER_LC}" \
  --teacher-framework recbole \
  --teacher-model "${MODEL_LC}" \
  --student-framework recbole \
  --student-backbone "${MODEL_LC}" \
  --student-embedding-dim "${EMBEDDING_DIM}" \
  --top-k "${TOP_K}" \
  --assert-no-train-leak

echo "============================================================"

echo "[DONE] Student evaluation completed"

echo "Timestamp: $(date)"

echo "Distiller: ${DISTILLER}"

echo "Dataset:   ${DATASET}"

echo "Model:     ${MODEL}"

echo "============================================================"
