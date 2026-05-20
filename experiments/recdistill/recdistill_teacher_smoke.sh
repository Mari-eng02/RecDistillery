#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  echo "Usage: bash experiments/recdistill/recdistill_teacher_smoke.sh <framework> <model> <dataset> <embedding_dim> [top_k]"
  echo
  echo "Examples:"
  echo "  bash experiments/recdistill/recdistill_teacher_smoke.sh recbole BPRMF bookcrossing 200"
  echo "  bash experiments/recdistill/recdistill_teacher_smoke.sh recbole LGCN amazon_cd 64 50"
  echo "  bash experiments/recdistill/recdistill_teacher_smoke.sh elliot NMF citeulike 200 20"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

FRAMEWORK="${1:-}"
MODEL="${2:-}"
DATASET="${3:-}"
EMBEDDING_DIM="${4:-}"
TOP_K="${5:-20}"

if [ -z "${FRAMEWORK}" ] || [ -z "${MODEL}" ] || [ -z "${DATASET}" ] || [ -z "${EMBEDDING_DIM}" ]; then
  usage
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 scripts/recdistill/teacher_smoke.py \
  --teacher-framework "${FRAMEWORK}" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --top-k "${TOP_K}"
