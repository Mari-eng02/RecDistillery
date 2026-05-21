#!/bin/bash

if [ -z "${BASH_VERSION:-}" ]; then
  echo "This script requires bash. Run it with: bash experiments/recdistill/train_distiller.sh ..."
  exit 1
fi

set -euo pipefail

usage() {
  echo "Usage: bash experiments/recdistill/train_distiller.sh <distiller> <dataset> [teacher_framework] [teacher_model|ALL] [student_framework] [student_backbone|SAME|ALL] [gpu] [dry_run]"
  echo
  echo "If teacher_framework is omitted, recbole is used."
  echo "If teacher_model is omitted or ALL, all teacher models for that framework are run sequentially."
  echo "If student_framework is omitted, the teacher framework is used."
  echo "If student_backbone is omitted or SAME, each student uses the same backbone name as its teacher."
  echo "By default the runner is detached in background (nohup)."
  echo "Set DISTILLER_DETACH=0 to run in foreground."
  echo
  echo "Examples:"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike recbole ALL recbole SAME 0 0"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike recbole BPRMF lenskit LGCN 0 1"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike elliot NMF elliot SAME auto 1"
  echo "  DISTILLER_DETACH=0 bash experiments/recdistill/train_distiller.sh DE citeulike recbole BPRMF recbole LGCN auto 1"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

INTERNAL_RUN="${1:-}"

if [ "${INTERNAL_RUN}" = "--internal-run" ]; then
  DISTILLER="${2:-DE}"
  DATASET="${3:-citeulike}"
  TEACHER_FRAMEWORK="${4:-recbole}"
  TEACHER_MODEL="${5:-ALL}"
  STUDENT_FRAMEWORK="${6:-${TEACHER_FRAMEWORK}}"
  STUDENT_MODEL="${7:-SAME}"
  GPU="${8:-0}"
  DRY_RUN="${9:-0}"
  RUN_ID="${10:-$(date +%Y%m%d_%H%M%S)}"
else
  DISTILLER="${1:-DE}"
  DATASET="${2:-citeulike}"
  TEACHER_FRAMEWORK="${3:-recbole}"
  TEACHER_MODEL="${4:-ALL}"
  STUDENT_FRAMEWORK="${5:-${TEACHER_FRAMEWORK}}"
  STUDENT_MODEL="${6:-SAME}"
  GPU="${7:-0}"
  DRY_RUN="${8:-0}"
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

if [ "${DRY_RUN}" != "0" ] && [ "${DRY_RUN}" != "1" ]; then
  echo "Invalid dry_run value: ${DRY_RUN}. Use 0 or 1."
  usage
  exit 1
fi

DETACH="${DISTILLER_DETACH:-1}"
if [ "${DETACH}" != "0" ] && [ "${DETACH}" != "1" ]; then
  echo "Invalid DISTILLER_DETACH value: ${DETACH}. Use 0 or 1."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/experiments/logs"
TEACHER_CONFIG_DIR="${REPO_ROOT}/config/models/teacher"
STUDENT_CONFIG_DIR="${REPO_ROOT}/config/models/student"

mkdir -p "${LOG_DIR}"

DISTILLER_LC="$(echo "${DISTILLER}" | tr '[:upper:]' '[:lower:]')"
DATASET_LC="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')"
TEACHER_FRAMEWORK_LC="$(echo "${TEACHER_FRAMEWORK}" | tr '[:upper:]' '[:lower:]')"
TEACHER_MODEL_LC="$(echo "${TEACHER_MODEL}" | tr '[:upper:]' '[:lower:]')"
STUDENT_FRAMEWORK_LC="$(echo "${STUDENT_FRAMEWORK}" | tr '[:upper:]' '[:lower:]')"
STUDENT_MODEL_LC="$(echo "${STUDENT_MODEL}" | tr '[:upper:]' '[:lower:]')"

if [ ! -d "${TEACHER_CONFIG_DIR}/${TEACHER_FRAMEWORK_LC}" ]; then
  echo "Teacher framework config directory not found: ${TEACHER_CONFIG_DIR}/${TEACHER_FRAMEWORK_LC}"
  exit 1
fi

if [ ! -d "${STUDENT_CONFIG_DIR}/${STUDENT_FRAMEWORK_LC}" ]; then
  echo "Student framework config directory not found: ${STUDENT_CONFIG_DIR}/${STUDENT_FRAMEWORK_LC}"
  exit 1
fi

if [ "${INTERNAL_RUN}" != "--internal-run" ]; then
  RUNNER_LOG="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_FRAMEWORK}-${TEACHER_MODEL}_student-${STUDENT_FRAMEWORK}-${STUDENT_MODEL}_${RUN_ID}.runner.log"

  if [ "${DETACH}" = "1" ]; then
    nohup bash "${REPO_ROOT}/experiments/recdistill/train_distiller.sh" --internal-run \
      "${DISTILLER}" "${DATASET}" "${TEACHER_FRAMEWORK}" "${TEACHER_MODEL}" "${STUDENT_FRAMEWORK}" "${STUDENT_MODEL}" "${GPU}" "${DRY_RUN}" "${RUN_ID}" \
      > "${RUNNER_LOG}" 2>&1 < /dev/null &

    PID=$!

    echo "Detached runner started."
    echo "PID: ${PID}"
    echo "Distiller: ${DISTILLER}"
    echo "Dataset: ${DATASET}"
    echo "Teacher: ${TEACHER_FRAMEWORK}/${TEACHER_MODEL}"
    echo "Student: ${STUDENT_FRAMEWORK}/${STUDENT_MODEL}"
    echo "GPU: ${GPU}"
    echo "Dry-run: ${DRY_RUN}"
    echo "Runner log: ${RUNNER_LOG}"
    echo "Monitor with: tail -f ${RUNNER_LOG}"
  else
    echo "Foreground mode enabled (DISTILLER_DETACH=0)."
    echo "Distiller: ${DISTILLER}"
    echo "Dataset: ${DATASET}"
    echo "Teacher: ${TEACHER_FRAMEWORK}/${TEACHER_MODEL}"
    echo "Student: ${STUDENT_FRAMEWORK}/${STUDENT_MODEL}"
    echo "GPU: ${GPU}"
    echo "Dry-run: ${DRY_RUN}"
    echo "Runner log: ${RUNNER_LOG}"
    bash "${REPO_ROOT}/experiments/recdistill/train_distiller.sh" --internal-run \
      "${DISTILLER}" "${DATASET}" "${TEACHER_FRAMEWORK}" "${TEACHER_MODEL}" "${STUDENT_FRAMEWORK}" "${STUDENT_MODEL}" "${GPU}" "${DRY_RUN}" "${RUN_ID}" \
      | tee "${RUNNER_LOG}"
  fi
  exit 0
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

list_available_configs() {
  echo "Available teacher model configs:"
  find "${TEACHER_CONFIG_DIR}/${TEACHER_FRAMEWORK_LC}" -maxdepth 1 -type f -name '*.yaml' -print | sed "s#${REPO_ROOT}/##" | sort
  echo
  echo "Available student model configs:"
  find "${STUDENT_CONFIG_DIR}/${STUDENT_FRAMEWORK_LC}" -maxdepth 1 -type f -name '*.yaml' -print | sed "s#${REPO_ROOT}/##" | sort
}

discover_teacher_models() {
  find "${TEACHER_CONFIG_DIR}/${TEACHER_FRAMEWORK_LC}" -maxdepth 1 -type f -name '*.yaml' \
    | sed -E "s#^${TEACHER_CONFIG_DIR}/${TEACHER_FRAMEWORK_LC}/(.*)\.yaml#\1#" \
    | sort
}

discover_student_models() {
  find "${STUDENT_CONFIG_DIR}/${STUDENT_FRAMEWORK_LC}" -maxdepth 1 -type f -name '*.yaml' \
    | sed -E "s#^${STUDENT_CONFIG_DIR}/${STUDENT_FRAMEWORK_LC}/(.*)\.yaml#\1#" \
    | sort
}

run_one_pair() {
  local TEACHER_NAME="$1"
  local STUDENT_NAME="$2"
  local TEACHER_NAME_LC
  local STUDENT_NAME_LC
  local OUT_FILE
  local ERR_FILE

  TEACHER_NAME_LC="$(echo "${TEACHER_NAME}" | tr '[:upper:]' '[:lower:]')"
  STUDENT_NAME_LC="$(echo "${STUDENT_NAME}" | tr '[:upper:]' '[:lower:]')"

  if [ ! -f "${TEACHER_CONFIG_DIR}/${TEACHER_FRAMEWORK_LC}/${TEACHER_NAME_LC}.yaml" ]; then
    echo "Teacher model config not found: ${TEACHER_CONFIG_DIR}/${TEACHER_FRAMEWORK_LC}/${TEACHER_NAME_LC}.yaml"
    list_available_configs
    return 1
  fi

  if [ ! -f "${STUDENT_CONFIG_DIR}/${STUDENT_FRAMEWORK_LC}/${STUDENT_NAME_LC}.yaml" ]; then
    echo "Student model config not found: ${STUDENT_CONFIG_DIR}/${STUDENT_FRAMEWORK_LC}/${STUDENT_NAME_LC}.yaml"
    list_available_configs
    return 1
  fi

  OUT_FILE="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_FRAMEWORK}-${TEACHER_NAME}_student-${STUDENT_FRAMEWORK}-${STUDENT_NAME}_${RUN_ID}.out"
  ERR_FILE="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_FRAMEWORK}-${TEACHER_NAME}_student-${STUDENT_FRAMEWORK}-${STUDENT_NAME}_${RUN_ID}.err"

  CMD=(
    python3 -u scripts/recdistill/train_student_from_config.py
    --dataset "${DATASET_LC}"
    --teacher-framework "${TEACHER_FRAMEWORK_LC}"
    --teacher-model "${TEACHER_NAME_LC}"
    --distiller "${DISTILLER_LC}"
    --student-framework "${STUDENT_FRAMEWORK_LC}"
    --student-backbone "${STUDENT_NAME_LC}"
    --output-strategy best
  )

  if [ "${DRY_RUN}" = "1" ]; then
    CMD+=(--dry-run)
  fi

  echo "======================================="
  echo "Distiller training started at $(date)"
  echo "Distiller: ${DISTILLER}"
  echo "Dataset: ${DATASET}"
  echo "Teacher: ${TEACHER_FRAMEWORK}/${TEACHER_NAME}"
  echo "Student: ${STUDENT_FRAMEWORK}/${STUDENT_NAME}"
  echo "GPU: ${GPU}"
  echo "Dry-run: ${DRY_RUN}"
  echo "Config system: centralized ConfigLoader"
  echo "Command: ${CMD[*]}"
  echo "Logs: ${OUT_FILE} / ${ERR_FILE}"
  echo "======================================="
  echo

  if [ "${GPU}" = "auto" ]; then
    "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}"
  else
    CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${OUT_FILE}" 2> "${ERR_FILE}"
  fi

  echo
  echo "Completed teacher=${TEACHER_FRAMEWORK}/${TEACHER_NAME}, student=${STUDENT_FRAMEWORK}/${STUDENT_NAME} at $(date)"
  echo "Logs: ${OUT_FILE} / ${ERR_FILE}"
  echo
}

echo "Detached internal runner started at $(date)"
echo "Distiller: ${DISTILLER}"
echo "Dataset: ${DATASET}"
echo "Requested teacher: ${TEACHER_FRAMEWORK}/${TEACHER_MODEL}"
echo "Requested student: ${STUDENT_FRAMEWORK}/${STUDENT_MODEL}"
echo "GPU: ${GPU}"
echo "Dry-run: ${DRY_RUN}"
echo "Run ID: ${RUN_ID}"
echo

TEACHERS=()
if [ "${TEACHER_MODEL_LC}" = "all" ]; then
  while IFS= read -r _model; do
    [ -n "${_model}" ] && TEACHERS+=("${_model}")
  done < <(discover_teacher_models)

  if [ "${#TEACHERS[@]}" -eq 0 ]; then
    echo "No teacher model configs found"
    list_available_configs
    exit 1
  fi
else
  TEACHERS=("${TEACHER_MODEL}")
fi

if [ "${STUDENT_MODEL_LC}" = "all" ]; then
  STUDENTS=()
  while IFS= read -r _model; do
    [ -n "${_model}" ] && STUDENTS+=("${_model}")
  done < <(discover_student_models)

  if [ "${#STUDENTS[@]}" -eq 0 ]; then
    echo "No student model configs found"
    list_available_configs
    exit 1
  fi
fi

for T in "${TEACHERS[@]}"; do
  if [ "${STUDENT_MODEL_LC}" = "same" ]; then
    run_one_pair "${T}" "${T}"
  elif [ "${STUDENT_MODEL_LC}" = "all" ]; then
    for S in "${STUDENTS[@]}"; do
      run_one_pair "${T}" "${S}"
    done
  else
    run_one_pair "${T}" "${STUDENT_MODEL}"
  fi
done

echo "Detached internal runner completed at $(date)"
