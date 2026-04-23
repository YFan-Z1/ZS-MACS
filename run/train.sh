#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train.yaml}
shift || true

python -m main --config "$CONFIG" "$@"
