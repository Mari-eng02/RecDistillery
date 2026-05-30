#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TARGET="${1:-cpu}"

if [ "${TARGET}" != "mps" ] && [ "${TARGET}" != "cuda" ] && [ "${TARGET}" != "cpu" ]; then
    echo "Usage: bash setup/setup_environment.sh [cpu|mps|cuda]"
    exit 1
fi

if [ "${TARGET}" = "mps" ]; then
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_mps.txt"
elif [ "${TARGET}" = "cuda" ]; then
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_cuda.txt"
else
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_cpu.txt"
fi

conda create -n distillation python=3.12 -y
conda activate distillation

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "setuptools<81"
python -m pip install ninja
python -m pip install -r "${REQUIREMENTS_FILE}"
