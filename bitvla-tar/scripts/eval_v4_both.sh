#!/usr/bin/env bash
# v4 chained eval: Ablation A (λ=0.01) then Ablation B (λ=0.10)
#
# Usage:
#   bash scripts/eval_v4_both.sh <ckpt_lambda_0.01> <ckpt_lambda_0.10>
#
# Example (paths from doc/20260504-实验报告.md):
#   bash scripts/eval_v4_both.sh \
#     runs/.../tar-0.01--v4-tar001--10000_chkpt/lora_adapter \
#     runs/.../tar-0.1--v4-tar01--10000_chkpt/lora_adapter
#
# Matches: screen -dmS v4eval bash ft_script/eval_v4_both.sh
set -euo pipefail

CKPT_A="${1:?Usage: bash scripts/eval_v4_both.sh <ckpt_0.01> <ckpt_0.10>}"
CKPT_B="${2:?Usage: bash scripts/eval_v4_both.sh <ckpt_0.01> <ckpt_0.10>}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[eval_v4_both] === Eval A (λ=0.01), ~11h ==="
bash "${ROOT}/scripts/eval_tar.sh" "${CKPT_A}" "v4-tar001"

echo "[eval_v4_both] === Eval B (λ=0.10), ~11h ==="
bash "${ROOT}/scripts/eval_tar.sh" "${CKPT_B}" "v4-tar01"

echo "[eval_v4_both] Done. Total ~23h for both groups (500 episodes each)."
