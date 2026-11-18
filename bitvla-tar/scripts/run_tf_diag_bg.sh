#!/usr/bin/env bash
# E2 teacher-forcing diagnostic for both models, detached.
#   setsid nohup bash scripts/run_tf_diag_bg.sh > logs/e2_tf_diag.out 2>&1 < /dev/null &
set -uo pipefail
unset PYOPENGL_PLATFORM OMP_NUM_THREADS
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bitvla
export PYTHONPATH="/root/robotic/bitvla-tar/openvla-oft/LIBERO:/root/robotic/bitvla-tar/openvla-oft/bitvla:${PYTHONPATH:-}"
cd /root/robotic/bitvla-tar
for m in tar base; do
  python scripts/teacher_forcing_diag.py --model "${m}" --stride "${STRIDE:-20}"
  echo "TF_DIAG_${m}_EXIT=$?"
done
