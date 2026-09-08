#!/usr/bin/env bash
# Minimal GPU smoke test for TAR code paths (after NVIDIA driver works).
#
# Usage: bash scripts/verify_tar_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi not working. Run: bash scripts/setup_nvidia_driver.sh"
  exit 1
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== PyTorch CUDA ==="
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(4, 8, 7, device="cuda")
v_start = x[:, 1, :] - x[:, 0, :]
v_end = x[:, -1, :] - x[:, -2, :]
tar = (v_start.abs() + v_end.abs()).mean()
print("TAR tensor on GPU OK, tar_loss=", float(tar))
PY

echo "=== Smoothness metrics (CPU) ==="
python3 - <<PY
import sys
sys.path.insert(0, "${ROOT}/openvla-oft/experiments/robot/libero")
import numpy as np
from smoothness_metrics import SmoothnessTracker
tr = SmoothnessTracker()
tr.start_episode()
tr.add_chunk(np.random.randn(8, 7))
tr.end_episode(True)
print("Smoothness OK:", tr.aggregate())
PY

echo "=== FinetuneConfig import ==="
python3 - <<PY
import sys
sys.path.insert(0, "${ROOT}/openvla-oft/vla-scripts")
from finetune_bitnet import FinetuneConfig
c = FinetuneConfig(tar_lambda=0.05, use_fsdp=True, gradient_checkpointing=True)
print("FinetuneConfig OK:", c.tar_lambda, c.use_fsdp)
PY

echo ""
echo "GPU logic checks passed."
echo "Note: full train/eval still needs downloaded bitvla-bf16 weights + LIBERO sim."
echo "  bash scripts/link_local_assets.sh"
echo "  bash scripts/download_models.sh"
