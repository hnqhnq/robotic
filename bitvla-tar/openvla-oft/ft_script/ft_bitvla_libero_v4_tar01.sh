#!/usr/bin/env bash
# v4 ablation B (λ=0.10) — see doc/20260504-实验报告.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "${ROOT}/scripts/train_tar.sh" 0.10 v4-tar01 "$@"
