#!/usr/bin/env bash
# Symlink data/checkpoints from robotic monorepo (local dev only)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONO="$(cd "${ROOT}/.." && pwd)"

link_if_missing() {
  local src="$1"
  local dest="$2"
  if [[ -e "${dest}" ]]; then
    echo "[link_local_assets] Skip (exists): ${dest}"
    return 0
  fi
  if [[ ! -e "${src}" ]]; then
    echo "[link_local_assets] Source not found: ${src}"
    return 0
  fi
  ln -s "${src}" "${dest}"
  echo "[link_local_assets] ${dest} -> ${src}"
}

link_if_missing "${MONO}/src/data" "${ROOT}/data"
link_if_missing "${MONO}/src/checkpoints" "${ROOT}/checkpoints"

echo "[link_local_assets] Done."
