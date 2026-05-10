#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 "$ROOT_DIR/main.py" --config "$ROOT_DIR/exps/imagenet_r/care_inr_inc20.json"