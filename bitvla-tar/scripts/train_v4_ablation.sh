#!/usr/bin/env bash
# v4 ablation training: λ=0.01 then λ=0.10 (sequential, ~80h on 4×4090D)
#
# Usage: bash scripts/train_v4_ablation.sh
#
# Matches doc/20260504-实验报告.md (v4 消融 A + B)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[train_v4_ablation] === Ablation A: tar_lambda=0.01 (~40h) ==="
bash "${ROOT}/scripts/train_tar.sh" 0.01 v4-tar001

echo "[train_v4_ablation] === Ablation B: tar_lambda=0.10 (~40h) ==="
bash "${ROOT}/scripts/train_tar.sh" 0.10 v4-tar01

echo "[train_v4_ablation] Done. Run eval: bash scripts/eval_v4_both.sh <ckpt_a> <ckpt_b>"
