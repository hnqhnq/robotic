#!/usr/bin/env bash
# LIBERO-10 eval for a TAR (or baseline) checkpoint
#
# Usage: bash scripts/eval_tar.sh /path/to/checkpoint [label]
#
# Example:
#   bash scripts/eval_tar.sh runs/.../lora_adapter v4-tar001
set -euo pipefail

CKPT="${1:?Usage: bash scripts/eval_tar.sh /path/to/checkpoint [label]}"
LABEL="${2:-eval}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVAL_SCRIPT="${ROOT}/openvla-oft/experiments/robot/libero/run_libero_eval_bitnet.py"
LOG_DIR="${ROOT}/runs/eval_logs"
mkdir -p "${LOG_DIR}"

if [[ ! -e "${CKPT}" ]]; then
  echo "Error: checkpoint not found: ${CKPT}"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_${LABEL}_${STAMP}.log"

echo "[eval_tar] Label: ${LABEL}"
echo "[eval_tar] Checkpoint: ${CKPT}"
echo "[eval_tar] Log: ${LOG_FILE}"

cd "${ROOT}/openvla-oft"

python experiments/robot/libero/run_libero_eval_bitnet.py \
  --model_family bitnet \
  --pretrained_checkpoint "${CKPT}" \
  --task_suite_name libero_10 \
  --num_trials_per_task 50 \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_l1_regression True \
  --center_crop True \
  --use_wandb False \
  --local_log_dir "${LOG_DIR}" \
  --info_in_path "${LABEL}" \
  --run_id_note "${LABEL}" \
  2>&1 | tee "${LOG_FILE}"
