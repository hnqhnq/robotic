#!/usr/bin/env bash
# LIBERO-10 eval for a TAR (or baseline) checkpoint
#
# Usage: bash scripts/eval_tar.sh /path/to/checkpoint [label]
#
# Example:
#   bash scripts/eval_tar.sh runs/.../lora_adapter v4-tar001
#
# Optional env: TASK_IDS="0,1,2"  evaluate only these LIBERO-10 task ids (0-9). Used for sharding; see
#   scripts/run_eval_sharded.sh. Shard logs get a "-t0_1_2" suffix; unset => all 10 tasks (original behaviour).
# Optional env: NUM_TRIALS (default 50), OPEN_LOOP (execution horizon, default 8),
#   EXEC_MODE chunk|te|blend (default chunk), TE_DECAY (default 0.01).
#   Raw per-episode npz files go to runs/eval_logs/<label>/raw/.
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

TASK_IDS="${TASK_IDS:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
OPEN_LOOP="${OPEN_LOOP:-8}"
EXEC_MODE="${EXEC_MODE:-chunk}"
TE_DECAY="${TE_DECAY:-0.01}"
NOTE="${LABEL}"
TASK_ARG=()
if [[ -n "${TASK_IDS}" ]]; then
  NOTE="${LABEL}-t${TASK_IDS//,/_}"
  TASK_ARG=(--task_ids "${TASK_IDS}")
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_${NOTE}_${STAMP}.log"

echo "[eval_tar] Label: ${LABEL}"
echo "[eval_tar] Checkpoint: ${CKPT}"
echo "[eval_tar] Log: ${LOG_FILE}"

cd "${ROOT}/openvla-oft"

python experiments/robot/libero/run_libero_eval_bitnet.py \
  --model_family bitnet \
  --pretrained_checkpoint "${CKPT}" \
  --task_suite_name libero_10 \
  --num_trials_per_task "${NUM_TRIALS}" \
  --num_open_loop_steps "${OPEN_LOOP}" \
  --execution_mode "${EXEC_MODE}" \
  --te_decay "${TE_DECAY}" \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_l1_regression True \
  --center_crop True \
  --use_wandb False \
  --local_log_dir "${LOG_DIR}" \
  --info_in_path "${LABEL}" \
  --run_id_note "${NOTE}" \
  ${TASK_ARG[@]+"${TASK_ARG[@]}"} \
  2>&1 | tee "${LOG_FILE}"
