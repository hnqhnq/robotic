#!/usr/bin/env bash
# BitVLA-TAR env on AutoDL (1×4090). Aligns with BitVLA README + doc/REPRODUCE.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# PyPI mirror (AutoDL / 国内). Override: PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED="${PIP_TRUSTED:-mirrors.aliyun.com}"
export PIP_INDEX_URL="${PIP_INDEX}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED}"

pip_mirror() {
  pip "$@" -i "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"; exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx bitvla; then
  conda create -n bitvla python=3.10 -y
fi
conda activate bitvla

# PyTorch wheels: official CUDA index only (do not use PyPI mirror here).
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip_mirror install -e "${ROOT}/transformers/"

# Pin versions to avoid long pip backtracking (opencv / wandb / protobuf).
pip_mirror install \
  "accelerate>=0.25.0" draccus==0.8.0 einops json-numpy jsonlines matplotlib \
  peft==0.11.1 "protobuf>=3.20,<5" rich sentencepiece==0.1.99 timm==0.9.10 \
  tokenizers==0.19.1 "wandb>=0.16,<0.18" \
  tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 \
  "tensorflow-metadata==1.14.0" "protobuf>=4.21,<5" \
  "diffusers==0.32.2" imageio uvicorn fastapi opencv-python-headless==4.8.1.78

pip_mirror install "dlimp @ git+https://github.com/moojink/dlimp_openvla"

pip_mirror install -e "${ROOT}/openvla-oft" --no-deps
pip_mirror install -e "${ROOT}/openvla-oft/bitvla/"

LIBERO_DIR="${ROOT}/openvla-oft/LIBERO"
if [[ ! -f "${LIBERO_DIR}/setup.py" ]]; then
  rm -rf "${LIBERO_DIR}"
  # GitHub over HTTP/2 often fails on AutoDL; zip mirror is most reliable.
  if ! GIT_HTTP_VERSION=HTTP/1.1 git clone --depth 1 \
      https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"; then
    LIBERO_ZIP="/tmp/libero-master.zip"
    curl -L --retry 3 -o "${LIBERO_ZIP}" \
      "https://ghfast.top/https://github.com/Lifelong-Robot-Learning/LIBERO/archive/refs/heads/master.zip"
    unzip -q "${LIBERO_ZIP}" -d /tmp
    mv /tmp/LIBERO-master "${LIBERO_DIR}"
  fi
fi
pip_mirror install -r "${ROOT}/openvla-oft/experiments/robot/libero/libero_requirements.txt"
pip_mirror install -e "${LIBERO_DIR}"
pip_mirror install "tokenizers>=0.21,<0.22"
pip_mirror install "protobuf==4.25.9" "tensorflow-metadata==1.14.0" --no-deps

bash "${ROOT}/scripts/verify_tar_gpu.sh"
echo "Environment ready. Activate: conda activate bitvla"
