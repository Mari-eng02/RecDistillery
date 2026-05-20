#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: bash experiments/recdistill/train_single_distiller.sh <distiller> <dataset> <teacher> [student|SAME] [gpu] [dry_run]"
  echo
  echo "Builds the run from centralized configs under config/."
  echo
  echo "The launcher detaches automatically with nohup."
  echo
  echo "Arguments:"
  echo "  distiller  Example: DE"
  echo "  dataset    Example: citeulike"
  echo "  teacher    Example: BPRMF | LGCN | NMF"
  echo "  student    Default: SAME. Example: BPRMF | LGCN | NMF"
  echo "  gpu        Default: 0 (set to 'auto' to avoid CUDA_VISIBLE_DEVICES)"
  echo "  dry_run    Default: 0 (1=true, 0=false)"
  echo
  echo "Examples:"
  echo "  bash experiments/recdistill/train_single_distiller.sh DE citeulike BPRMF"
  echo "  bash experiments/recdistill/train_single_distiller.sh DE citeulike LGCN 0"
  echo "  bash experiments/recdistill/train_single_distiller.sh DE citeulike NMF auto 1"
  echo "  bash experiments/recdistill/train_single_distiller.sh DE citeulike BPRMF LGCN 0 1"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

INTERNAL_RUN="${1:-}"

if [ "${INTERNAL_RUN}" = "--internal-run" ]; then
  DISTILLER="${2:-DE}"
  DATASET="${3:-citeulike}"
  TEACHER_MODEL="${4:-BPRMF}"
  STUDENT_MODEL="${5:-SAME}"
  GPU="${6:-0}"
  DRY_RUN="${7:-0}"
  RUN_ID="${8:-$(date +%Y%m%d_%H%M%S)}"
else
  DISTILLER="${1:-DE}"
  DATASET="${2:-citeulike}"
  TEACHER_MODEL="${3:-BPRMF}"
  STUDENT_MODEL="SAME"
  GPU="0"
  DRY_RUN="0"

  ARG4="${4:-}"
  if [ -n "${ARG4}" ]; then
    ARG4_LC="$(echo "${ARG4}" | tr '[:upper:]' '[:lower:]')"
    if [ "${ARG4_LC}" = "auto" ] || [[ "${ARG4}" =~ ^[0-9,]+$ ]]; then
      GPU="${ARG4}"
      DRY_RUN="${5:-0}"
    else
      STUDENT_MODEL="${ARG4}"
      GPU="${5:-0}"
      DRY_RUN="${6:-0}"
    fi
  fi
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

if [ "${DRY_RUN}" != "0" ] && [ "${DRY_RUN}" != "1" ]; then
  echo "Invalid dry_run value: ${DRY_RUN}. Use 0 or 1."
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/logs"
mkdir -p "${LOG_DIR}"

DISTILLER_LC="$(echo "${DISTILLER}" | tr '[:upper:]' '[:lower:]')"
DATASET_LC="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
TEACHER_MODEL_LC="$(echo "${TEACHER_MODEL}" | tr '[:upper:]' '[:lower:]')"
STUDENT_MODEL_LC="$(echo "${STUDENT_MODEL}" | tr '[:upper:]' '[:lower:]')"

if [ "${STUDENT_MODEL_LC}" = "same" ]; then
  STUDENT_MODEL="${TEACHER_MODEL}"
  STUDENT_MODEL_LC="${TEACHER_MODEL_LC}"
fi

TEACHER_CONFIG_PATH="${REPO_ROOT}/config/models/teacher/${TEACHER_MODEL_LC}.yaml"
STUDENT_CONFIG_PATH="${REPO_ROOT}/config/models/student/${STUDENT_MODEL_LC}.yaml"

if [ ! -f "${TEACHER_CONFIG_PATH}" ]; then
  echo "Teacher model config not found: ${TEACHER_CONFIG_PATH}"
  echo "Available teacher model configs:"
  find "${REPO_ROOT}/config/models/teacher" -maxdepth 1 -type f -name '*.yaml' -print | sed "s#${REPO_ROOT}/##"
  exit 1
fi

if [ ! -f "${STUDENT_CONFIG_PATH}" ]; then
  echo "Student model config not found: ${STUDENT_CONFIG_PATH}"
  echo "Available student model configs:"
  find "${REPO_ROOT}/config/models/student" -maxdepth 1 -type f -name '*.yaml' -print | sed "s#${REPO_ROOT}/##"
  exit 1
fi

RUNNER_LOG="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_MODEL}_student-${STUDENT_MODEL}_${RUN_ID}.log"
OUT_FILE="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_MODEL}_student-${STUDENT_MODEL}_${RUN_ID}.out"
ERR_FILE="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_MODEL}_student-${STUDENT_MODEL}_${RUN_ID}.err"

if [ "${INTERNAL_RUN}" != "--internal-run" ]; then
  nohup bash "${REPO_ROOT}/experiments/recdistill/train_distiller.sh" --internal-run \
    "${DISTILLER}" "${DATASET}" "${TEACHER_MODEL}" "${STUDENT_MODEL}" "${GPU}" "${DRY_RUN}" "${RUN_ID}" \
    > "${RUNNER_LOG}" 2>&1 < /dev/null &
  PID=$!

  echo "Detached runner started."
  echo "PID: ${PID}"
  echo "Config system: centralized ConfigLoader"
  echo "Teacher: ${TEACHER_MODEL}"
  echo "Student: ${STUDENT_MODEL}"
  echo "Runner log: ${RUNNER_LOG}"
  echo "Process logs: ${OUT_FILE} / ${ERR_FILE}"
  exit 0
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  python3 scripts/recdistill/train_student_from_config.py
  --dataset "${DATASET_LC}"
  --teacher "${TEACHER_MODEL_LC}"
  --distiller "${DISTILLER_LC}"
  --student "${STUDENT_MODEL_LC}"
)
if [ "${DRY_RUN}" = "1" ]; then
  CMD+=(--dry-run)
fi

echo "Distiller training runner started at $(date)"
echo "Distiller: ${DISTILLER}"
echo "Dataset: ${DATASET}"
echo "Teacher: ${TEACHER_MODEL}"
echo "Student: ${STUDENT_MODEL}"
echo "GPU: ${GPU}"
echo "Dry-run: ${DRY_RUN}"
echo "Config system: centralized ConfigLoader"
echo "Command: ${CMD[*]}"
echo "Logs: ${OUT_FILE} / ${ERR_FILE}"
echo

if [ "${GPU}" = "auto" ]; then
  "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}"
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}"
fi

echo "Distiller training runner completed at $(date)"
echo "Logs: ${OUT_FILE} / ${ERR_FILE}"
