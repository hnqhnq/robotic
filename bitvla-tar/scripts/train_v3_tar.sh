#!/usr/bin/env bash
# Backward-compatible wrapper → scripts/train_tar.sh 0.05
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${ROOT}/scripts/train_tar.sh" 0.05 v3-tar "$@"
