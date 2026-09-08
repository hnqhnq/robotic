#!/usr/bin/env bash
# Download pretrained BitVLA and optional LIBERO-long baseline
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${ROOT}/checkpoints"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${CKPT_DIR}"

clone_model() {
  local repo="$1"
  local dest="$2"
  if [[ -d "${dest}" ]]; then
    echo "[download_models] Skip (exists): ${dest}"
    return 0
  fi
  echo "[download_models] Cloning ${repo} -> ${dest}"
  git clone "${HF_ENDPOINT}/${repo}" "${dest}"
}

# Required: pretrain checkpoint for TAR finetuning
clone_model "lxsy/bitvla-bf16" "${CKPT_DIR}/bitvla-bf16"

# Optional: official LIBERO-long baseline for eval comparison
clone_model "hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16" \
  "${CKPT_DIR}/ft-bitvla-libero-long" || true

echo "[download_models] Done."
