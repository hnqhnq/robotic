#!/usr/bin/env bash
# Stage A1: TAR λ=0.10, same recipe as v4-tar01 (SEED=1, 10K steps, 4-GPU FSDP). Loss via W&B offline (default).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/stage_a1_driver.out"
{
  echo "[A1] $(date -Iseconds) start"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate bitvla
  export PYTHONPATH="${ROOT}/openvla-oft/bitvla:${PYTHONPATH:-}"
  # USE_WANDB defaults to True in train_tar.sh; WANDB_MODE=offline
  AUTO_RESUME=0 NPROC=4 SEED=1 FSDP_CPU_OFFLOAD=False \
    bash scripts/train_tar.sh 0.10
  echo "[A1] $(date -Iseconds) exit=$?"
} >>"$LOG" 2>&1