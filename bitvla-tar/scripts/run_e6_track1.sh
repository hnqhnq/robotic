#!/usr/bin/env bash
# E6: failure tagging on E3 (+ optional E4) npz. CPU-only, no GPU.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs/track1"
mkdir -p "$LOG_DIR" report/data
OUT="$LOG_DIR/e6_classify.out"
{
  echo "[e6] $(date -Iseconds) start"
  python3 scripts/classify_failures_e6.py \
    --labels e3-base-h8 e3-tar-h8 e4-base-h4 e4-tar-h4 e4-base-h2 e4-tar-h2 \
  --out report/data/e6_failure_modes.json \
  --md report/data/e6_failure_modes.md
  echo "[e6] $(date -Iseconds) done"
} >>"$OUT" 2>&1