#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/logs"
mkdir -p "${LOG_DIR}"

RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/evaluate_fetched_distillers_${RUN_ID}.log"

{
  echo "============================================================"
  echo "[START] Batch fetch evaluation"
  echo "Timestamp: $(date)"
  echo "Log file: ${LOG_FILE}"
  echo "============================================================"

  # DE
  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de amazon_cd bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de bookcrossing bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de citeulike bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de amazon_cd lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de bookcrossing lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de citeulike lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de amazon_cd nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de bookcrossing nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de citeulike nmf

  # # DE_RRD
  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd amazon_cd bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd bookcrossing bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd citeulike bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd amazon_cd lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd bookcrossing lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd citeulike lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd amazon_cd nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd bookcrossing nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh de_rrd citeulike nmf

  # FTD
  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd amazon_cd bprmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd bookcrossing bprmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd citeulike bprmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd amazon_cd lgcn

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd bookcrossing lgcn

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd citeulike lgcn

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd amazon_cd nmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd bookcrossing nmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" ftd citeulike nmf

  # HTD
  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd amazon_cd bprmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd bookcrossing bprmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd citeulike bprmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd amazon_cd lgcn

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd bookcrossing lgcn

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd citeulike lgcn

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd amazon_cd nmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd bookcrossing nmf

  bash "${REPO_ROOT}/experiments/recdistill/evaluate_single_fetched_distiller.sh" htd citeulike nmf

  # RRD
  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd amazon_cd bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd bookcrossing bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd citeulike bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd amazon_cd lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd bookcrossing lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd citeulike lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd amazon_cd nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd bookcrossing nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh rrd citeulike nmf

  # # UNKD
  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd amazon_cd bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd bookcrossing bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd citeulike bprmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd amazon_cd lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd bookcrossing lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd citeulike lgcn

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd amazon_cd nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd bookcrossing nmf

  # bash experiments/recdistill/evaluate_single_fetched_distiller.sh unkd citeulike nmf

  echo "============================================================"
  echo "[DONE] Batch fetch evaluation"
  echo "Timestamp: $(date)"
  echo "============================================================"
} 2>&1 | tee "${LOG_FILE}"
