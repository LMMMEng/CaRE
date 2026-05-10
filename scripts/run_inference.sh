#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Example: Run inference with a trained checkpoint
# Usage: bash scripts/run_inference.sh <config_path> <checkpoint_path> <gpu_id>

CONFIG_PATH=${1:-"$ROOT_DIR/exps/cifar100/care_cifar_inc10.json"}
CHECKPOINT_PATH=${2:-""}
GPU_ID=${3:-0}

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Usage: bash scripts/run_inference.sh [config_path] <checkpoint_path> [gpu_id]" >&2
    exit 1
fi

python3 "$ROOT_DIR/inference.py" \
       --config $CONFIG_PATH \
       --checkpoint $CHECKPOINT_PATH \
       --device $GPU_ID