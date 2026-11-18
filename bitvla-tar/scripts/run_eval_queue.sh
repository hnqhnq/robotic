#!/usr/bin/env bash
# Run a queue of eval jobs with bounded concurrency on one GPU (detached).
#
#   setsid nohup bash scripts/run_eval_queue.sh scripts/eval_queue_track1.txt > logs/track1_queue.out 2>&1 < /dev/null &
#
# Env: PARALLEL (default 3; ~6-7GB GPU memory per eval process on a 24GB card)
#      WAIT_FOR_FILE + WAIT_FOR_PATTERN: block until the pattern appears in the file (e.g. E2 finished)
# Each job's stdout goes to logs/track1/<label>-t<tasks>.out; finished jobs are appended to logs/track1/done.txt
# and skipped on restart.
set -uo pipefail
QUEUE="${1:?queue file}"
PARALLEL="${PARALLEL:-3}"
ROOT=/root/robotic/bitvla-tar
OUT="${ROOT}/logs/track1"
mkdir -p "${OUT}"
touch "${OUT}/done.txt"

if [[ -n "${WAIT_FOR_FILE:-}" ]]; then
  echo "[queue] waiting for '${WAIT_FOR_PATTERN}' in ${WAIT_FOR_FILE}"
  until grep -q "${WAIT_FOR_PATTERN}" "${WAIT_FOR_FILE}" 2>/dev/null; do sleep 60; done
fi

run_job() {
  local model=$1 label=$2 open_loop=$3 mode=$4 tasks=$5
  local key="${label} ${tasks}"
  if grep -qxF "${key}" "${OUT}/done.txt"; then echo "[queue] skip ${key}"; return 0; fi
  # Model loading rewrites the checkpoint's config.json; keep job starts >= 90 s apart.
  flock "${OUT}/.start.lock" sleep 90
  echo "[queue] $(date '+%F %T') start ${key}"
  TASK_IDS="${tasks}" OPEN_LOOP="${open_loop}" EXEC_MODE="${mode}" \
    bash "${ROOT}/scripts/run_eval_bg.sh" "${model}" "${label}" > "${OUT}/${label}-t${tasks//,/_}.out" 2>&1
  if grep -q "EVAL_EXIT=0" "${OUT}/${label}-t${tasks//,/_}.out"; then
    echo "${key}" >> "${OUT}/done.txt"
    echo "[queue] $(date '+%F %T') done  ${key}"
  else
    echo "[queue] $(date '+%F %T') FAIL  ${key}"
  fi
}
export -f run_job
export ROOT OUT

grep -vE '^\s*(#|$)' "${QUEUE}" | xargs -P "${PARALLEL}" -L 1 bash -c 'run_job "$@"' _
echo "[queue] $(date '+%F %T') all jobs finished"
