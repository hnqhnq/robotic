#!/usr/bin/env bash
# v4 ablation A (λ=0.01) — see doc/20260504-实验报告.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "${ROOT}/scripts/train_tar.sh" 0.01 v4-tar001 "$@"
