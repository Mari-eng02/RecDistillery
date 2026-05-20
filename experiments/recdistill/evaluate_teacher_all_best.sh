#!/bin/bash

set -u -o pipefail

usage() {
  echo "Usage: bash experiments/recdistill/evaluate_teacher_all_best.sh [embedding_dim] [top_k] [batch_size] [device] [assert_no_train_leak]"
  echo
  echo "Arguments:"
  echo "  embedding_dim         Default: 200"
  echo "  top_k                 Default: 20"
  echo "  batch_size            Default: 256"
  echo "  device                Default: auto (cuda if available, else cpu)"
  echo "  assert_no_train_leak  Default: 1 (1=true, 0=false)"
  echo
  echo "Examples:"
  echo "  bash experiments/recdistill/evaluate_teacher_all_best.sh"
  echo "  bash experiments/recdistill/evaluate_teacher_all_best.sh 200 20 256 cuda:0 1"
  echo "  bash experiments/recdistill/evaluate_teacher_all_best.sh 200 50 512 cpu 0"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

EMBEDDING_DIM="${1:-200}"
TOP_K="${2:-20}"
BATCH_SIZE="${3:-256}"
DEVICE="${4:-auto}"
ASSERT_NO_TRAIN_LEAK="${5:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/logs"
mkdir -p "${LOG_DIR}"

MODELS=("BPRMF" "LGCN" "NMF")
DATASETS=("citeulike" "amazon_cd" "bookcrossing")

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/evaluate_teacher_all_best_${TIMESTAMP}.log"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

success_count=0
fail_count=0

{
  echo "Teacher evaluation batch started at $(date)"
  echo "embedding_dim=${EMBEDDING_DIM}, top_k=${TOP_K}, batch_size=${BATCH_SIZE}, device=${DEVICE}, assert_no_train_leak=${ASSERT_NO_TRAIN_LEAK}"
  echo
} | tee -a "${RUN_LOG}"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    out_file="${LOG_DIR}/evaluate_teacher_${model}_${dataset}_${TIMESTAMP}.out"
    err_file="${LOG_DIR}/evaluate_teacher_${model}_${dataset}_${TIMESTAMP}.err"

    cmd=(
      python3 scripts/recdistill/evaluate_teacher.py
      --dataset "${dataset}"
      --model "${model}"
      --embedding-dim "${EMBEDDING_DIM}"
      --top-k "${TOP_K}"
      --batch-size "${BATCH_SIZE}"
    )

    if [ "${DEVICE}" != "auto" ]; then
      cmd+=(--device "${DEVICE}")
    fi
    if [ "${ASSERT_NO_TRAIN_LEAK}" = "1" ]; then
      cmd+=(--assert-no-train-leak)
    fi

    {
      echo "--------------------------------------------------------------------------------"
      echo "Running model=${model}, dataset=${dataset}"
      echo "Command: ${cmd[*]}"
      echo "Logs: ${out_file} / ${err_file}"
    } | tee -a "${RUN_LOG}"

    if "${cmd[@]}" >"${out_file}" 2>"${err_file}"; then
      success_count=$((success_count + 1))
      echo "Status: OK (${model}, ${dataset})" | tee -a "${RUN_LOG}"
    else
      fail_count=$((fail_count + 1))
      echo "Status: FAIL (${model}, ${dataset})" | tee -a "${RUN_LOG}"
      echo "Check: ${err_file}" | tee -a "${RUN_LOG}"
    fi
  done
done

{
  echo
  echo "Teacher evaluation batch completed at $(date)"
  echo "Successful runs: ${success_count}"
  echo "Failed runs: ${fail_count}"
  echo "Batch log: ${RUN_LOG}"
} | tee -a "${RUN_LOG}"

if [ "${fail_count}" -gt 0 ]; then
  exit 1
fi

exit 0
