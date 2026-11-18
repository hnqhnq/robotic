#!/usr/bin/env bash
# Environment wrapper for detached evals (survives IDE disconnects when started with setsid nohup).
#
# Usage:
#   setsid nohup bash scripts/run_eval_bg.sh <tar|base> <label> > logs/<label>.out 2>&1 < /dev/null &
#
# Env (forwarded to scripts/eval_tar.sh): TASK_IDS, NUM_TRIALS, OPEN_LOOP, EXEC_MODE, TE_DECAY, CUDA_VISIBLE_DEVICES
set -uo pipefail
unset PYOPENGL_PLATFORM
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bitvla
export PYTHONPATH="/root/robotic/bitvla-tar/openvla-oft/LIBERO:/root/robotic/bitvla-tar/openvla-oft/bitvla:${PYTHONPATH:-}"
cd /root/robotic/bitvla-tar

case "${1:?model: tar|base}" in
  tar)  CKPT="/root/autodl-tmp/robotic-data/runs/bitvla-bf16+libero_10_no_noops+b16+lr-0.0004--image_aug--v4-tar01--image_aug+tar-0.1--10000_chkpt/lora_adapter" ;;
  base) CKPT="/root/robotic/bitvla-tar/checkpoints/ft-bitvla-libero-long" ;;
  *) echo "unknown model $1"; exit 1 ;;
esac
LABEL="${2:?label required}"

bash scripts/eval_tar.sh "${CKPT}" "${LABEL}"
echo "EVAL_EXIT=$?"
