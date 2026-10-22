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
# Environment overrides: VLA_PATH, DATA_ROOT, RUN_ROOT, MAX_STEPS, NPROC, SAVE_FREQ,
#   KEEP_LAST_N_CHECKPOINTS, AUTO_RESUME, SEED
# SEED=<int>: seed torch/numpy/random + RLDS shuffle (per-rank derived); run tag gets suffix "-s<SEED>" so it never
#   collides with / auto-resumes an existing unseeded run. Unset => original unseeded behaviour.
#   e.g.  SEED=1 bash scripts/train_tar.sh 0.10     # -> run tag v4-tar01-s1
#         SEED=1 bash scripts/train_tar.sh 0.0      # -> run tag tar-l0.0-s1 (lambda=0 baseline, same recipe)
# AUTO_RESUME=1 (default): if a prior run_id checkpoint exists, continue from the latest step.
# Disk-friendly defaults (~30G free on 50G data disk): save every 5000 steps, keep latest 1 ckpt
#   (~12G peak under runs/). Step 10000 still exports lora_adapter for eval.
set -euo pipefail

TAR_LAMBDA="${1:-0.05}"
RUN_TAG="${2:-${RUN_TAG:-}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/openvla-oft/bitvla:${PYTHONPATH:-}"
FT_DIR="${ROOT}/openvla-oft/ft_script"

VLA_PATH="${VLA_PATH:-${ROOT}/checkpoints/bitvla-bf16}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/modified_libero_rlds}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
MAX_STEPS="${MAX_STEPS:-10001}"
NPROC="${NPROC:-4}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
if [[ "${AUTO_RESUME}" == "0" || "${AUTO_RESUME}" == "false" || "${AUTO_RESUME}" == "False" ]]; then
  AUTO_RESUME_ARG="False"
else
  AUTO_RESUME_ARG="True"
fi

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
  if [[ -n "${SEED:-}" ]]; then
    RUN_TAG="${RUN_TAG}-s${SEED}"
  fi
fi
SEED_ARG=()
if [[ -n "${SEED:-}" ]]; then
  SEED_ARG=(--seed "${SEED}")
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
echo "[train_tar] save_freq=${SAVE_FREQ} keep_last_n_checkpoints=${KEEP_LAST_N_CHECKPOINTS} auto_resume=${AUTO_RESUME}"
echo "[train_tar] seed=${SEED:-<unseeded>}"
echo "[train_tar] run_id_note=${RUN_ID_NOTE}"
echo "[train_tar] Log: ${LOG_FILE}"

cd "${FT_DIR}"

# FSDP checkpoint all-gather can run >10 min on 3B; avoid NCCL watchdog kill mid-save.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

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
  --save_freq "${SAVE_FREQ}" \
  --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS}" \
  --save_latest_checkpoint_only False \
  --auto_resume "${AUTO_RESUME_ARG}" \
  --image_aug True \
  --run_id_note "${RUN_ID_NOTE}" \
  --tar_lambda "${TAR_LAMBDA}" \
  --gradient_checkpointing True \
  --use_fsdp True \
  --use_wandb False \
  ${SEED_ARG[@]+"${SEED_ARG[@]}"} \
  2>&1 | tee "${LOG_FILE}"
