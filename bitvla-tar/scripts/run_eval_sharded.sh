#!/usr/bin/env bash
# Run one LIBERO-10 eval (500 episodes) as N parallel shards (task-id round-robin), then merge + report.
# Total GPU work is unchanged; wall-clock drops ~N-fold (BitVLA inference needs only ~1.4GB/process, and
# OSMesa rendering is CPU-bound, so several shards per GPU is fine as long as there are enough CPU cores).
#
# Usage:
#   bash scripts/run_eval_sharded.sh /path/to/checkpoint label
#
# Env:
#   NUM_SHARDS   number of parallel processes            (default 4; max useful = 10 tasks)
#   GPUS         space-separated GPU ids, round-robin     (default "0 1 2 3")
#   MUJOCO_GL    osmesa | egl                             (default osmesa = the setting validated in 20261003 report)
#   TASK_SUBSET  restrict to e.g. "6" for a quick shard-vs-monolithic sanity check (comma-separated task ids)
#
# Example (sanity check on one task before trusting sharding):
#   TASK_SUBSET=4 NUM_SHARDS=1 GPUS="0" bash scripts/run_eval_sharded.sh <ckpt> shardcheck-v4tar01
set -euo pipefail

CKPT="${1:?Usage: bash scripts/run_eval_sharded.sh /path/to/checkpoint label}"
LABEL="${2:?label required}"
NUM_SHARDS="${NUM_SHARDS:-4}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
TASK_SUBSET="${TASK_SUBSET:-0,1,2,3,4,5,6,7,8,9}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset PYOPENGL_PLATFORM
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}/openvla-oft/LIBERO:${ROOT}/openvla-oft/bitvla:${PYTHONPATH:-}"

# Build round-robin shard -> task lists
IFS=',' read -r -a TASKS <<< "${TASK_SUBSET}"
declare -a SHARD_TASKS
for ((i = 0; i < NUM_SHARDS; i++)); do SHARD_TASKS[i]=""; done
for ((j = 0; j < ${#TASKS[@]}; j++)); do
  i=$((j % NUM_SHARDS))
  SHARD_TASKS[i]="${SHARD_TASKS[i]:+${SHARD_TASKS[i]},}${TASKS[j]}"
done

mkdir -p "${ROOT}/runs/eval_logs"
PIDS=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  [[ -z "${SHARD_TASKS[i]}" ]] && continue
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  echo "[sharded] shard ${i}: GPU ${gpu}, tasks ${SHARD_TASKS[i]}"
  CUDA_VISIBLE_DEVICES="${gpu}" TASK_IDS="${SHARD_TASKS[i]}" \
    bash "${ROOT}/scripts/eval_tar.sh" "${CKPT}" "${LABEL}" > "${ROOT}/runs/eval_logs/shard_${LABEL}_${i}.out" 2>&1 &
  PIDS+=($!)
  sleep 5   # avoid identical timestamps / simultaneous model-load spikes
done

fail=0
for pid in "${PIDS[@]}"; do wait "${pid}" || fail=$((fail + 1)); done
if [[ ${fail} -gt 0 ]]; then
  echo "[sharded] ${fail} shard(s) failed; see runs/eval_logs/shard_${LABEL}_*.out"
  exit 1
fi

echo "[sharded] all shards done. Merged summary:"
python3 "${ROOT}/scripts/eval_stats.py" --summary "${ROOT}/runs/eval_logs/${LABEL}/EVAL-*.txt"
