#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <machine> <destination_path>"
  echo "Example: $0 gpu-node-1 /Users/alberto/tracked_results/gpu-node-1"
  exit 1
fi

MACHINE="$1"
DESTINATION_PATH="$2"
REMOTE_ROOT="/home/alberto/RecSys-Distillation-Reproducibility/results/recdistill"

mkdir -p "${DESTINATION_PATH}"

ssh "${MACHINE}" "test -d '${REMOTE_ROOT}'"

rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='tracked/' \
  --include='tracked/***' \
  --exclude='*' \
  -e ssh \
  "${MACHINE}:${REMOTE_ROOT}/" \
  "${DESTINATION_PATH}/"
