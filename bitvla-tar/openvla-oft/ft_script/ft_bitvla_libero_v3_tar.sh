#!/usr/bin/env bash
# v3 (λ=0.05) — see doc/20260418-实验报告.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "${ROOT}/scripts/train_tar.sh" 0.05 v3-tar "$@"
