#!/usr/bin/env bash
set -euo pipefail
unset PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bitvla
export PYTHONPATH="/root/robotic/bitvla-tar/openvla-oft/LIBERO:/root/robotic/bitvla-tar/openvla-oft/bitvla:${PYTHONPATH:-}"
cd /root/robotic/bitvla-tar
CKPT="/root/autodl-tmp/robotic-data/runs/bitvla-bf16+libero_10_no_noops+b16+lr-0.0004--image_aug--v4-tar01--image_aug+tar-0.1--10000_chkpt/lora_adapter"
exec bash scripts/eval_tar.sh "${CKPT}" v4-tar01
