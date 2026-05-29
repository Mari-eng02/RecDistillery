#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mapfile -t CONFIGS < <(find results/recdistill -path '*/config/*_best.yaml' -type f | sort)
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "No best configs found under results/recdistill/*/config/*_best.yaml"
  exit 0
fi

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running tracked rerun from ${CONFIG}"
  python3 -u scripts/recdistill/train_student_from_config.py --config "${CONFIG}" --track
done
