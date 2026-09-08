#!/usr/bin/env bash
# Backward-compatible wrapper → scripts/eval_tar.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${ROOT}/scripts/eval_tar.sh" "$@"
