#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "${ROOT}/scripts/eval_tar.sh" "${1:?Usage: eval_bitvla_v4_tar001.sh /path/to/checkpoint}" "v4-tar001"
