#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-vaw_open_vocab_seg/configs/train_base.yaml}
DS_CONFIG=${DS_CONFIG:-vaw_open_vocab_seg/configs/deepspeed_zero2.json}
NUM_PROCESSES=${NUM_PROCESSES:-2}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
shift || true

accelerate launch   --use_deepspeed   --deepspeed_config_file "${DS_CONFIG}"   --num_processes=${NUM_PROCESSES}   --mixed_precision=${MIXED_PRECISION}   -m vaw_open_vocab_seg.train_main   --config "$CONFIG" "$@"
