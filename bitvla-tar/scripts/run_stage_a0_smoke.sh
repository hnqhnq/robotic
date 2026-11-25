#!/usr/bin/env bash
# Stage A0: 4-GPU ~100-step smoke (SEED=1, lambda=0.10).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/stage_a0_driver.out"
{
  echo "[A0] $(date -Iseconds) start"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate bitvla
  export PYTHONPATH="${ROOT}/openvla-oft/bitvla:${PYTHONPATH:-}"
  AUTO_RESUME=0 MAX_STEPS=100 SAVE_FREQ=99999 KEEP_LAST_N_CHECKPOINTS=0 \
    NPROC=4 SEED=1 RUN_TAG=a0-smoke-s1 FSDP_CPU_OFFLOAD=False \
    bash scripts/train_tar.sh 0.10
  echo "[A0] $(date -Iseconds) exit=$?"
} >>"$LOG" 2>&1