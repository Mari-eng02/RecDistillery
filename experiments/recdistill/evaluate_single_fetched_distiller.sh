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
FETCHED_ROOT="fetched"
CONFIG_ROOT="config/presets/recdistill/final_rerun"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRACKED_ROOT="${FETCHED_ROOT}/${DISTILLER_LC}/${MODEL_LC}/${DATASET_LC}/tracked"
if [ ! -d "${TRACKED_ROOT}" ]; then
  echo "Warning: tracked root not found: ${TRACKED_ROOT}" >&2
  exit 0
fi

RECAP_PATH="$(find "${TRACKED_ROOT}" -mindepth 2 -maxdepth 2 -name 'run_recap.json' | sort | tail -n 1 || true)"
if [ -z "${RECAP_PATH}" ]; then
  echo "Warning: no run_recap.json found under ${TRACKED_ROOT}" >&2
  exit 0
fi

RUN_DIR="$(dirname "${RECAP_PATH}")"
CONFIG_PATH="${CONFIG_ROOT}/${DISTILLER_LC}/${MODEL_LC}/${DATASET_LC}/best.yaml"
if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Warning: config not found: ${CONFIG_PATH}" >&2
  exit 0
fi

SOURCE_CHECKPOINT="$(RECAP_PATH="${RECAP_PATH}" python3 - <<'PY'
import json
import os
from pathlib import Path

recap = Path(os.environ["RECAP_PATH"])
payload = json.loads(recap.read_text(encoding="utf-8"))
record = payload[0] if isinstance(payload, list) and payload else {}
best = record.get("best_checkpoint_path")
if isinstance(best, str) and best:
    best_str = best
    if best_str.startswith("results/recdistill/"):
        best_str = best_str.replace("results/recdistill/", "fetched/", 1)
    elif best_str.startswith("results/"):
        best_str = best_str.replace("results/", "fetched/", 1)
    print(best_str)
PY
)"

if [ -z "${SOURCE_CHECKPOINT}" ] || [ ! -f "${SOURCE_CHECKPOINT}" ]; then
  SOURCE_CHECKPOINT="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.best.distilled_student' | sort | tail -n 1 || true)"
fi

if [ -z "${SOURCE_CHECKPOINT}" ] || [ ! -f "${SOURCE_CHECKPOINT}" ]; then
  echo "Warning: no best checkpoint found under ${RUN_DIR}/checkpoints" >&2
  exit 0
fi

PERF_DIR="${RUN_DIR}/perf"
OUTPUT_JSON="${PERF_DIR}/${DISTILLER_LC}_${MODEL_LC}_${DATASET_LC}_${EMBEDDING_DIM}_eval_top${TOP_K}.json"
OUTPUT_TSV="${OUTPUT_JSON%.json}.tsv"

mkdir -p "${PERF_DIR}"

echo "============================================================"
echo "[START] Fetched student evaluation"
echo "Timestamp: $(date)"
echo "Distiller: ${DISTILLER}"
echo "Dataset:   ${DATASET}"
echo "Model:     ${MODEL}"
echo "Config:    ${CONFIG_PATH}"
echo "Source:    ${SOURCE_CHECKPOINT}"
echo "Perf:      ${PERF_DIR}"
echo "============================================================"

echo "[1/1] Evaluating fetched distilled student artifact..."
python3 scripts/recdistill/evaluate_students.py \
  --path "${SOURCE_CHECKPOINT}" \
  --distiller "${DISTILLER_LC}" \
  --teacher-model "${MODEL_LC}" \
  --dataset "${DATASET_LC}" \
  --student-model "${MODEL_LC}" \
  --student-embedding-dim "${EMBEDDING_DIM}" \
  --top-k "${TOP_K}" \
  --assert-no-train-leak \
  --output-json "${OUTPUT_JSON}" \
  --output-tsv "${OUTPUT_TSV}"

echo "============================================================"
echo "[DONE] Fetched student evaluation completed"
echo "Timestamp: $(date)"
echo "Distiller: ${DISTILLER}"
echo "Dataset:   ${DATASET}"
echo "Model:     ${MODEL}"
echo "============================================================"
