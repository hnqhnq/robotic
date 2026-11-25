#!/usr/bin/env bash
# Stage A3: λ=0 fair baseline, same recipe as v4-tar01 (SEED=1, 10K steps, 4-GPU FSDP).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/stage_a3_driver.out"
{
  echo "[A3] $(date -Iseconds) start"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate bitvla
  export PYTHONPATH="${ROOT}/openvla-oft/bitvla:${PYTHONPATH:-}"
  AUTO_RESUME=0 NPROC=4 SEED=1 FSDP_CPU_OFFLOAD=False \
    bash scripts/train_tar.sh 0.0
  echo "[A3] $(date -Iseconds) exit=$?"
} >>"$LOG" 2>&1