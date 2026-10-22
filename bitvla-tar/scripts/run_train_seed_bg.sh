#!/usr/bin/env bash
# Detached full TAR train with fixed seed (for tier-2 seed replication). Requires 4 GPUs by default.
#
# Usage:
#   SEED=1 bash scripts/run_train_seed_bg.sh          # λ=0.10 -> tag v4-tar01-s1
#   SEED=1 TAR_LAMBDA=0 bash scripts/run_train_seed_bg.sh   # baseline λ=0 -> tar-l0.0-s1
#
# Monitor: bash scripts/check_background_jobs.sh
#          tail -f logs/train_<run_tag>_lambda*.log
set -uo pipefail
SEED="${SEED:?set SEED=1 or 2}"
TAR_LAMBDA="${TAR_LAMBDA:-0.10}"
NPROC="${NPROC:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/seed_train_s${SEED}_lambda${TAR_LAMBDA}_driver.out"

{
  echo "[seed_bg] start $(date -Iseconds) SEED=${SEED} TAR_LAMBDA=${TAR_LAMBDA} NPROC=${NPROC}"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate bitvla
  cd "${ROOT}"
  AUTO_RESUME=0 NPROC="${NPROC}" SEED="${SEED}" \
    bash scripts/train_tar.sh "${TAR_LAMBDA}"
  echo "SEED_TRAIN_EXIT=$?"
} >> "${LOG}" 2>&1
