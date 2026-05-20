#!/bin/bash

ENV="${1:-distillation}"

CONDA_BASE="$(conda info --base 2>/dev/null)"

if [ -z "$CONDA_BASE" ]; then
    echo "Unable to determine conda base. Ensure 'conda' is installed and on PATH."
    return 1 2>/dev/null || exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../"
export CUBLAS_WORKSPACE_CONFIG=:4096:8


echo "PYTHONPATH set to: ${PYTHONPATH}"
