#!/usr/bin/env bash
# Single-GPU smoke: ~100 training steps with SEED=1 (FSDP + gradient checkpointing).
# NOTE: On 1×4090D (~24GB) this typically OOMs at optimizer.step() — FSDP still needs full Adam
# state on one rank. Use scripts/run_train_seed_bg.sh with NPROC=4 for real seed runs.
# Launched via: setsid nohup bash scripts/run_train_smoke_1gpu.sh > logs/smoke_1gpu_driver.out 2>&1 &
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bitvla
cd /root/robotic/bitvla-tar
export PYTHONPATH="${PWD}/openvla-oft/bitvla:${PYTHONPATH:-}"

AUTO_RESUME=0 NPROC=1 MAX_STEPS=105 SAVE_FREQ=99999 KEEP_LAST_N_CHECKPOINTS=0 \
  SEED=1 RUN_TAG=smoke-1gpu-s1 \
  bash scripts/train_tar.sh 0.10 smoke-1gpu-s1

echo "SMOKE_1GPU_EXIT=$?"
