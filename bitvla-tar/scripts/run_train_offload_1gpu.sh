#!/usr/bin/env bash
# E8: single-GPU full-parameter training feasibility with FSDP CPU offload (params/grads/Adam state in CPU RAM).
#   setsid nohup bash scripts/run_train_offload_1gpu.sh > logs/e8_offload_driver.out 2>&1 < /dev/null &
# Env: MAX_STEPS (default 4 -> 3 optimizer steps), GRAD_ACCUM (default 8)
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bitvla
cd /root/robotic/bitvla-tar
export PYTHONPATH="${PWD}/openvla-oft/bitvla:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

AUTO_RESUME=0 NPROC=1 MAX_STEPS="${MAX_STEPS:-4}" SAVE_FREQ=99999 KEEP_LAST_N_CHECKPOINTS=0 \
  GRAD_ACCUM="${GRAD_ACCUM:-8}" FSDP_CPU_OFFLOAD=True SEED=1 RUN_TAG=smoke-1gpu-offload \
  bash scripts/train_tar.sh 0.10 smoke-1gpu-offload

echo "E8_EXIT=$?"
