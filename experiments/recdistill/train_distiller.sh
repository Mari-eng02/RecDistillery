#!/bin/bash

if [ -z "${BASH_VERSION:-}" ]; then
  echo "This script requires bash. Run it with: bash experiments/recdistill/train_distiller.sh ..."
  exit 1
fi

set -euo pipefail

usage() {
  echo "Usage: bash experiments/recdistill/train_distiller.sh <distiller> <dataset> [teacher|ALL] [student|SAME|ALL] [gpu] [dry_run]"
  echo
  echo "If teacher is omitted, all teacher models are run sequentially."
  echo "If student is omitted or SAME, each student uses the same backbone as its teacher."
  echo "By default the runner is detached in background (nohup)."
  echo "Set DISTILLER_DETACH=0 to run in foreground."
  echo
  echo "Examples:"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike ALL"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike BPRMF"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike LGCN 0"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike ALL auto 1"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike BPRMF LGCN 0 1"
  echo "  bash experiments/recdistill/train_distiller.sh DE citeulike ALL ALL auto 1"
  echo "  DISTILLER_DETACH=0 bash experiments/recdistill/train_distiller.sh DE citeulike BPRMF LGCN auto 1"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

INTERNAL_RUN="${1:-}"

if [ "${INTERNAL_RUN}" = "--internal-run" ]; then
  DISTILLER="${2:-DE}"
  DATASET="${3:-citeulike}"
  TEACHER_MODEL="${4:-ALL}"
  STUDENT_MODEL="${5:-SAME}"
  GPU="${6:-0}"
  DRY_RUN="${7:-0}"
  RUN_ID="${8:-$(date +%Y%m%d_%H%M%S)}"
else
  DISTILLER="${1:-DE}"
  DATASET="${2:-citeulike}"
  TEACHER_MODEL="${3:-ALL}"
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
TEACHER_MODEL_LC="$(echo "${TEACHER_MODEL}" | tr '[:upper:]' '[:lower:]')"
STUDENT_MODEL_LC="$(echo "${STUDENT_MODEL}" | tr '[:upper:]' '[:lower:]')"

if [ ! -d "${TEACHER_CONFIG_DIR}" ]; then
  echo "Teacher model config directory not found: ${TEACHER_CONFIG_DIR}"
  exit 1
fi

if [ ! -d "${STUDENT_CONFIG_DIR}" ]; then
  echo "Student model config directory not found: ${STUDENT_CONFIG_DIR}"
  exit 1
fi

if [ "${INTERNAL_RUN}" != "--internal-run" ]; then
  RUNNER_LOG="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_MODEL}_student-${STUDENT_MODEL}_${RUN_ID}.runner.log"

  if [ "${DETACH}" = "1" ]; then
    nohup bash "${REPO_ROOT}/experiments/recdistill/train_distiller.sh" --internal-run \
      "${DISTILLER}" "${DATASET}" "${TEACHER_MODEL}" "${STUDENT_MODEL}" "${GPU}" "${DRY_RUN}" "${RUN_ID}" \
      > "${RUNNER_LOG}" 2>&1 < /dev/null &

    PID=$!

    echo "Detached runner started."
    echo "PID: ${PID}"
    echo "Distiller: ${DISTILLER}"
    echo "Dataset: ${DATASET}"
    echo "Teacher: ${TEACHER_MODEL}"
    echo "Student: ${STUDENT_MODEL}"
    echo "GPU: ${GPU}"
    echo "Dry-run: ${DRY_RUN}"
    echo "Runner log: ${RUNNER_LOG}"
    echo "Monitor with: tail -f ${RUNNER_LOG}"
  else
    echo "Foreground mode enabled (DISTILLER_DETACH=0)."
    echo "Distiller: ${DISTILLER}"
    echo "Dataset: ${DATASET}"
    echo "Teacher: ${TEACHER_MODEL}"
    echo "Student: ${STUDENT_MODEL}"
    echo "GPU: ${GPU}"
    echo "Dry-run: ${DRY_RUN}"
    echo "Runner log: ${RUNNER_LOG}"
    bash "${REPO_ROOT}/experiments/recdistill/train_distiller.sh" --internal-run \
      "${DISTILLER}" "${DATASET}" "${TEACHER_MODEL}" "${STUDENT_MODEL}" "${GPU}" "${DRY_RUN}" "${RUN_ID}" \
      | tee "${RUNNER_LOG}"
  fi
  exit 0
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

list_available_configs() {
  echo "Available teacher model configs:"
  find "${TEACHER_CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' -print | sed "s#${REPO_ROOT}/##" | sort
  echo
  echo "Available student model configs:"
  find "${STUDENT_CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' -print | sed "s#${REPO_ROOT}/##" | sort
}

discover_teacher_models() {
  find "${TEACHER_CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' \
    | sed -E "s#^${TEACHER_CONFIG_DIR}/(.*)\.yaml#\1#" \
    | sort
}

discover_student_models() {
  find "${STUDENT_CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' \
    | sed -E "s#^${STUDENT_CONFIG_DIR}/(.*)\.yaml#\1#" \
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

  if [ ! -f "${TEACHER_CONFIG_DIR}/${TEACHER_NAME_LC}.yaml" ]; then
    echo "Teacher model config not found: ${TEACHER_CONFIG_DIR}/${TEACHER_NAME_LC}.yaml"
    list_available_configs
    return 1
  fi

  if [ ! -f "${STUDENT_CONFIG_DIR}/${STUDENT_NAME_LC}.yaml" ]; then
    echo "Student model config not found: ${STUDENT_CONFIG_DIR}/${STUDENT_NAME_LC}.yaml"
    list_available_configs
    return 1
  fi

  OUT_FILE="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_NAME}_student-${STUDENT_NAME}_${RUN_ID}.out"
  ERR_FILE="${LOG_DIR}/train_distiller_${DISTILLER}_${DATASET}_teacher-${TEACHER_NAME}_student-${STUDENT_NAME}_${RUN_ID}.err"

  CMD=(
    python3 -u scripts/recdistill/train_student_from_config.py
    --dataset "${DATASET_LC}"
    --teacher "${TEACHER_NAME_LC}"
    --distiller "${DISTILLER_LC}"
    --student "${STUDENT_NAME_LC}"
  )

  if [ "${DRY_RUN}" = "1" ]; then
    CMD+=(--dry-run)
  fi

  echo "======================================="
  echo "Distiller training started at $(date)"
  echo "Distiller: ${DISTILLER}"
  echo "Dataset: ${DATASET}"
  echo "Teacher: ${TEACHER_NAME}"
  echo "Student: ${STUDENT_NAME}"
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
  echo "Completed teacher=${TEACHER_NAME}, student=${STUDENT_NAME} at $(date)"
  echo "Logs: ${OUT_FILE} / ${ERR_FILE}"
  echo
}

echo "Detached internal runner started at $(date)"
echo "Distiller: ${DISTILLER}"
echo "Dataset: ${DATASET}"
echo "Requested teacher: ${TEACHER_MODEL}"
echo "Requested student: ${STUDENT_MODEL}"
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
