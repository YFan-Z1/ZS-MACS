#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-vaw_open_vocab_seg/configs/train_base.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-2}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
shift || true

accelerate launch   --multi_gpu   --num_processes=${NUM_PROCESSES}   --mixed_precision=${MIXED_PRECISION}   -m vaw_open_vocab_seg.train_main   --config "$CONFIG" "$@"
