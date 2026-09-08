#!/usr/bin/env bash
# Full-parameter LIBERO-10 finetune with TAR (v3 / v4 unified)
#
# Usage:
#   bash scripts/train_tar.sh [tar_lambda] [run_tag]
#
# Examples:
#   bash scripts/train_tar.sh 0.05              # v3 main experiment
#   bash scripts/train_tar.sh 0.01              # v4 ablation A (auto tag: v4-tar001)
#   bash scripts/train_tar.sh 0.10              # v4 ablation B (auto tag: v4-tar01)
#   bash scripts/train_tar.sh 0.05 my-exp       # custom run tag
#
# Environment overrides: VLA_PATH, DATA_ROOT, RUN_ROOT, MAX_STEPS, NPROC
set -euo pipefail

TAR_LAMBDA="${1:-0.05}"
RUN_TAG="${2:-${RUN_TAG:-}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FT_DIR="${ROOT}/openvla-oft/ft_script"

VLA_PATH="${VLA_PATH:-${ROOT}/checkpoints/bitvla-bf16}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/modified_libero_rlds}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
MAX_STEPS="${MAX_STEPS:-10001}"
NPROC="${NPROC:-4}"

# Match checkpoint naming in experiment reports (doc/20260418, doc/20260504)
tar_lambda_note() {
  case "$1" in
    0.10|0.1) echo "0.1" ;;
    *) echo "$1" ;;
  esac
}

default_run_tag() {
  case "$1" in
    0.05) echo "v3-tar" ;;
    0.01) echo "v4-tar001" ;;
    0.10|0.1) echo "v4-tar01" ;;
    *) echo "tar-l${1}" ;;
  esac
}

if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="$(default_run_tag "${TAR_LAMBDA}")"
fi

TAR_NOTE="$(tar_lambda_note "${TAR_LAMBDA}")"
RUN_ID_NOTE="${RUN_TAG}--image_aug+tar-${TAR_NOTE}"
LOG_FILE="${ROOT}/logs/train_${RUN_TAG}_lambda${TAR_NOTE}.log"

if [[ ! -d "${VLA_PATH}" ]]; then
  echo "Error: VLA checkpoint not found: ${VLA_PATH}"
  echo "Run: bash scripts/download_models.sh"
  exit 1
fi

if [[ ! -d "${DATA_ROOT}/libero_10_no_noops" ]]; then
  echo "Error: Dataset not found: ${DATA_ROOT}/libero_10_no_noops"
  echo "Run: bash scripts/download_data.sh"
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${ROOT}/logs"

echo "[train_tar] tar_lambda=${TAR_LAMBDA} run_tag=${RUN_TAG} max_steps=${MAX_STEPS} nproc=${NPROC}"
echo "[train_tar] run_id_note=${RUN_ID_NOTE}"
echo "[train_tar] Log: ${LOG_FILE}"

cd "${FT_DIR}"

# NOTE: TAR loss, FSDP, and gradient checkpointing are implemented in finetune_bitnet.py
torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC}" ../vla-scripts/finetune_bitnet.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT}/" \
  --dataset_name libero_10_no_noops \
  --run_root_dir "${RUN_ROOT}" \
  --use_l1_regression True \
  --warmup_steps 375 \
  --use_lora False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 4e-4 \
  --lr_vit 8e-5 \
  --max_steps "${MAX_STEPS}" \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --run_id_note "${RUN_ID_NOTE}" \
  --tar_lambda "${TAR_LAMBDA}" \
  --gradient_checkpointing True \
  --use_fsdp True \
  --use_wandb False \
  2>&1 | tee "${LOG_FILE}"
