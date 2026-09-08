#!/usr/bin/env bash
# Chained v4 eval — see doc/20260504-实验报告.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "${ROOT}/scripts/eval_v4_both.sh" "$@"
