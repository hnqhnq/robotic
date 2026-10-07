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
CKPT="/root/robotic/bitvla-tar/checkpoints/ft-bitvla-libero-long"
exec bash scripts/eval_tar.sh "${CKPT}" baseline-long
