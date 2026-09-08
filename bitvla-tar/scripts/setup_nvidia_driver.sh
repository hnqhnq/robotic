#!/usr/bin/env bash
# Fix NVIDIA driver for current Ubuntu HWE kernel (590-open).
#
# Problem this fixes:
#   nvidia-smi fails with "Module nvidia not found" when the running kernel
#   (e.g. 6.8.0-124) has no matching linux-modules-nvidia-* package yet.
#
# Usage:
#   bash scripts/setup_nvidia_driver.sh          # install modules for current kernel
#   bash scripts/setup_nvidia_driver.sh --reboot-110   # boot older kernel with modules
set -euo pipefail

MODE="${1:-install}"

echo "[setup_nvidia] kernel: $(uname -r)"
echo "[setup_nvidia] gpu:    $(lspci -nn | rg -i 'nvidia' | head -1 || true)"

if [[ "${MODE}" == "--reboot-110" ]]; then
  if ! [[ -d /lib/modules/6.8.0-110-generic/kernel/nvidia-590-open ]]; then
    echo "Error: nvidia-590-open modules for 6.8.0-110-generic not found."
    exit 1
  fi
  ENTRY='gnulinux-advanced-51b4450d-05d5-429e-8261-b132f815d46f>gnulinux-6.8.0-110-generic-advanced-51b4450d-05d5-429e-8261-b132f815d46f'
  echo "[setup_nvidia] Setting GRUB default to 6.8.0-110-generic ..."
  sudo grub-set-default "${ENTRY}"
  echo "[setup_nvidia] Rebooting in 5 seconds (Ctrl+C to cancel) ..."
  sleep 5
  sudo reboot
fi

echo "[setup_nvidia] Updating apt index ..."
sudo apt-get update -qq

echo "[setup_nvidia] Installing NVIDIA kernel modules for HWE kernel ..."
sudo apt-get install -y \
  nvidia-driver-590-open \
  linux-modules-nvidia-590-open-generic-hwe-22.04

echo "[setup_nvidia] Loading nvidia module ..."
sudo modprobe nvidia || true
sudo modprobe nvidia_uvm || true

echo "[setup_nvidia] nvidia-smi:"
nvidia-smi

python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY

echo "[setup_nvidia] Done. If nvidia-smi still fails, reboot once:"
echo "  sudo reboot"
echo "Or force boot kernel 6.8.0-110 (modules already installed):"
echo "  bash scripts/setup_nvidia_driver.sh --reboot-110"
