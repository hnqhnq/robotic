#!/usr/bin/env bash
# Download LIBERO RLDS dataset into bitvla-tar/data/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/data/modified_libero_rlds"

mkdir -p "${ROOT}/data"

if [[ -d "${DATA_DIR}/libero_10_no_noops" ]] && \
   compgen -G "${DATA_DIR}/libero_10_no_noops/*.tfrecord*" > /dev/null; then
  echo "[download_data] Dataset already present at ${DATA_DIR}"
  exit 0
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="openvla/modified_libero_rlds"

echo "[download_data] Cloning ${REPO} -> ${DATA_DIR}"
if command -v git &>/dev/null; then
  GIT_LFS_SKIP_SMUDGE=0 git clone "${HF_ENDPOINT}/datasets/${REPO}" "${DATA_DIR}"
else
  echo "Error: git is required. Install git-lfs for large files."
  exit 1
fi

echo "[download_data] Done."
