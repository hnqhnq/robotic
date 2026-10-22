#!/usr/bin/env bash
# Sanity check: sharded eval path vs the original monolithic eval (Task 5 / index 4 of LIBERO-10, v4-tar01).
# Reference (20261003 report): v4-tar01 Task 5 = 92% (46/50).  Expect ~the same here.
# Launched detached via setsid/nohup so it survives IDE disconnects.
set -uo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bitvla
cd /root/robotic/bitvla-tar
CK="/root/autodl-tmp/robotic-data/runs/bitvla-bf16+libero_10_no_noops+b16+lr-0.0004--image_aug--v4-tar01--image_aug+tar-0.1--10000_chkpt/lora_adapter"
TASK_SUBSET=4 NUM_SHARDS=1 GPUS="0" bash scripts/run_eval_sharded.sh "${CK}" shardcheck-v4tar01
echo "SHARDCHECK_EXIT=$?"
