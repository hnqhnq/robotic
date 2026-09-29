#!/usr/bin/env bash
# Download pretrained BitVLA and optional LIBERO-long baseline
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${ROOT}/checkpoints"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

mkdir -p "${CKPT_DIR}"

model_ready() {
  local dest="$1"
  [[ -f "${dest}/config.json" ]] || return 1
  [[ -f "${dest}/model.safetensors" ]] && return 0
  compgen -G "${dest}/*.safetensors" > /dev/null && return 0
  compgen -G "${dest}/*.bin" > /dev/null
}

download_model() {
  local repo_id="$1"
  local dest="$2"

  if model_ready "${dest}"; then
    echo "[download_models] Skip (ready): ${dest}"
    return 0
  fi

  if [[ -d "${dest}/.git" ]] && ! model_ready "${dest}"; then
    echo "[download_models] Removing incomplete checkout: ${dest}"
    rm -rf "${dest}"
  fi

  echo "[download_models] Downloading ${repo_id} -> ${dest}"

  if python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${repo_id}",
    repo_type="model",
    local_dir="${dest}",
    endpoint=os.environ.get("HF_ENDPOINT"),
    token=os.environ.get("HF_TOKEN"),
)
print("[download_models] snapshot_download finished: ${repo_id}")
PY
  else
    local clone_url="${HF_ENDPOINT}/${repo_id}"
    if [[ -n "${HF_TOKEN:-}" ]]; then
      local host="${HF_ENDPOINT#https://}"
      host="${host#http://}"
      clone_url="https://oauth2:${HF_TOKEN}@${host}/${repo_id}"
    fi
    git clone "${clone_url}" "${dest}"
  fi

  if ! model_ready "${dest}"; then
    echo "[download_models] Error: weights missing in ${dest} (LFS smudge failed?)"
    return 1
  fi
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[download_models] WARNING: HF_TOKEN is not set (anonymous downloads may be rate-limited)."
  echo "  export HF_TOKEN=hf_xxxxxxxx  # https://huggingface.co/settings/tokens"
fi

download_model "lxsy/bitvla-bf16" "${CKPT_DIR}/bitvla-bf16"

download_model "hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16" \
  "${CKPT_DIR}/ft-bitvla-libero-long" || \
  echo "[download_models] Optional baseline skipped (non-fatal)."

echo "[download_models] Done."
