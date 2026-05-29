#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <distiller> <dataset> <student_model>"
  exit 1
fi

DISTILLER_LC="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
DATASET_LC="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
MODEL_LC="$(echo "$3" | tr '[:upper:]' '[:lower:]')"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ARTIFACT="$(python3 - "${DISTILLER_LC}" "${DATASET_LC}" "${MODEL_LC}" <<'PY'
from pathlib import Path
import sys

distiller, dataset, model = [arg.lower() for arg in sys.argv[1:4]]
candidates = [
    path
    for path in Path("results/recdistill").glob("*/artifacts/*_best.distilled_student")
    if distiller in path.name.lower()
    and dataset in path.name.lower()
    and model in path.name.lower()
]
if candidates:
    print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
)"

if [ -z "${ARTIFACT}" ]; then
  echo "No best distilled student artifact found for ${DISTILLER_LC}/${MODEL_LC}/${DATASET_LC}."
  exit 1
fi

python3 scripts/recdistill/evaluate_students.py \
  --path "${ARTIFACT}" \
  --top-k 20 \
  --assert-no-train-leak
