#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/logs"

mkdir -p "${LOG_DIR}"

RUN_ID=$(date +%Y%m%d_%H%M%S)

LOG_FILE="${LOG_DIR}/evaluate_distillers_${RUN_ID}.log"

{

  echo "============================================================"

  echo "[START] Batch evaluation"

  echo "Timestamp: $(date)"

  echo "Log file: ${LOG_FILE}"

  echo "============================================================"


  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_distiller.sh" HTD bookcrossing NMF


  echo "================================================------------"

  echo "[DONE] Batch evaluation"

  echo "Timestamp: $(date)"

  echo "============================================================"

} 2>&1 | tee "${LOG_FILE}"
