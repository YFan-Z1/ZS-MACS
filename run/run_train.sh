#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-vaw_open_vocab_seg/configs/train_base.yaml}
shift || true

python -m vaw_open_vocab_seg.train_main --config "$CONFIG" "$@"
