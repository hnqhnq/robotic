#!/usr/bin/env bash
# Fix FSDP checkpoint layout for LIBERO eval (see doc/20260418-实验报告.md Section IV)
#
# Usage: bash scripts/fix_fsdp_checkpoint.sh /path/to/run--10000_chkpt
set -euo pipefail

CHKPT_DIR="${1:?Usage: bash scripts/fix_fsdp_checkpoint.sh /path/to/checkpoint_run_dir}"
ADAPTER_DIR="${CHKPT_DIR}/lora_adapter"

if [[ ! -d "${CHKPT_DIR}" ]]; then
  echo "Error: checkpoint dir not found: ${CHKPT_DIR}"
  exit 1
fi

mkdir -p "${ADAPTER_DIR}"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${dst}"
    echo "[fix] copied $(basename "${src}")"
  fi
}

# 1. dataset_statistics.json
copy_if_exists "${CHKPT_DIR}/dataset_statistics.json" "${ADAPTER_DIR}/dataset_statistics.json"

# 2. tokenizer / processor files
for f in tokenizer.json tokenizer_config.json special_tokens_map.json \
         preprocessor_config.json processor_config.json generation_config.json config.json; do
  copy_if_exists "${CHKPT_DIR}/${f}" "${ADAPTER_DIR}/${f}"
done

# 3. Symlink action head / proprio projector weights from parent run dir
for pt in "${CHKPT_DIR}"/*.pt; do
  [[ -e "${pt}" ]] || continue
  base="$(basename "${pt}")"
  if [[ ! -e "${ADAPTER_DIR}/${base}" ]]; then
    ln -sf "../${base}" "${ADAPTER_DIR}/${base}"
    echo "[fix] linked ${base}"
  fi
done

echo "[fix_fsdp_checkpoint] Done: ${ADAPTER_DIR}"
