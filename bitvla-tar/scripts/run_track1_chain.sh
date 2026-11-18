#!/usr/bin/env bash
# Track-1 GPU chain on one card: wait for E2 (teacher-forcing diag) -> E8 (1-GPU offload training test, needs
# the whole card) -> eval queue (E3/E4/E5).
#   setsid nohup bash scripts/run_track1_chain.sh > logs/track1_chain.out 2>&1 < /dev/null &
set -uo pipefail
cd /root/robotic/bitvla-tar
until grep -q TF_DIAG_base_EXIT logs/e2_tf_diag.out; do sleep 30; done
echo "[chain] $(date '+%F %T') E2 done, starting E8"
bash scripts/run_train_offload_1gpu.sh > logs/e8_offload_driver.out 2>&1
echo "[chain] $(date '+%F %T') E8 done, starting eval queue"
bash scripts/run_eval_queue.sh scripts/eval_queue_track1.txt > logs/track1_queue.out 2>&1
echo "[chain] $(date '+%F %T') all done"
