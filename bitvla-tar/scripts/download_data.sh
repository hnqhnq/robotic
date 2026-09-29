#!/usr/bin/env bash
# Download LIBERO RLDS dataset into bitvla-tar/data/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/data/modified_libero_rlds"
REPO_ID="openvla/modified_libero_rlds"

mkdir -p "${ROOT}/data"

dataset_ready() {
  compgen -G "${DATA_DIR}/libero_10_no_noops/*/*.tfrecord*" > /dev/null
}

if dataset_ready; then
  echo "[download_data] Dataset already present at ${DATA_DIR}"
  exit 0
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[download_data] WARNING: HF_TOKEN is not set."
  echo "  HuggingFace / mirror often rate-limits anonymous LFS downloads."
  echo "  Create a token at https://huggingface.co/settings/tokens then:"
  echo "    export HF_TOKEN=hf_xxxxxxxx"
  echo "  Or: huggingface-cli login"
fi

# Remove broken git-lfs checkout (metadata only, no .tfrecord)
if [[ -d "${DATA_DIR}/.git" ]] && ! dataset_ready; then
  echo "[download_data] Removing incomplete checkout: ${DATA_DIR}"
  rm -rf "${DATA_DIR}"
fi

echo "[download_data] Downloading ${REPO_ID} -> ${DATA_DIR}"
echo "[download_data] Using endpoint: ${HF_ENDPOINT}"

if python3 -c "import huggingface_hub" 2>/dev/null; then
  python3 - <<PY
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${REPO_ID}",
    repo_type="dataset",
    local_dir="${DATA_DIR}",
    endpoint=os.environ.get("HF_ENDPOINT"),
    token=os.environ.get("HF_TOKEN"),
)
print("[download_data] snapshot_download finished.")
PY
else
  CLONE_URL="${HF_ENDPOINT}/datasets/${REPO_ID}"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    HOST="${HF_ENDPOINT#https://}"
    HOST="${HOST#http://}"
    CLONE_URL="https://oauth2:${HF_TOKEN}@${HOST}/datasets/${REPO_ID}"
  fi
  GIT_LFS_SKIP_SMUDGE=0 git clone "${CLONE_URL}" "${DATA_DIR}"
fi

if ! dataset_ready; then
  echo "[download_data] Error: tfrecord files not found after download."
  echo "  Set HF_TOKEN and re-run, or retry later if rate-limited."
  exit 1
fi

echo "[download_data] Done."
