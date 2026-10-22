#!/usr/bin/env bash
# Quick status for detached eval/train jobs (safe to run anytime).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(no GPU)"
echo ""
echo "=== Running processes (train/eval) ==="
pgrep -af "finetune_bitnet|run_libero_eval_bitnet|train_tar.sh|run_eval_sharded|run_train_" 2>/dev/null | grep -v pgrep || echo "(none)"
echo ""
echo "=== Shard check (Task 5 consistency) ==="
if grep -q "SHARDCHECK_EXIT=0" "${ROOT}/runs/eval_logs/shardcheck_driver.out" 2>/dev/null; then
  grep "success:" "${ROOT}/runs/eval_logs/shardcheck_driver.out" 2>/dev/null | tail -1
else
  tail -3 "${ROOT}/runs/eval_logs/shardcheck_driver.out" 2>/dev/null || echo "(no shardcheck driver log)"
fi
echo ""
echo "=== 1-GPU smoke train ==="
D="${ROOT}/logs/smoke_1gpu_driver.out"
if [[ -f "${D}" ]]; then
  if grep -q "SMOKE_1GPU_EXIT=0" "${D}"; then echo "finished OK"; elif grep -q "SMOKE_1GPU_EXIT=" "${D}"; then
    grep "SMOKE_1GPU_EXIT=" "${D}"; grep -m1 "OutOfMemoryError\|CUDA out of memory" "${ROOT}/logs/train_smoke-1gpu-s1_lambda0.1.log" 2>/dev/null && echo "(likely OOM on 1×24GB — use 4 GPUs for full finetune)"
  else echo "still running or starting — tail: ${ROOT}/logs/train_smoke-1gpu-s1_lambda0.1.log"; fi
else
  echo "(not started)"
fi
echo ""
echo "=== Seed full train (4×GPU) ==="
for s in 1 2; do
  L="${ROOT}/logs/train_v4-tar01-s${s}_lambda0.1.log"
  if [[ -f "${L}" ]]; then
    echo -n "seed=${s}: "
    grep -oE '[0-9]+/10001' "${L}" 2>/dev/null | tail -1 || tail -1 "${L}" | cut -c1-120
  fi
done
