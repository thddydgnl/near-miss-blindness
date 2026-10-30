#!/bin/bash
# DP pipeline on the b-dp pod: data for the 6 M2 tasks (parallel), then training (TRAIN_PAR tasks at a time),
# then the quick evaluation of every checkpoint on held start states. Resumable: finished pieces are skipped.
source /workspace/code/env.sh
TASKS=${TASKS:-"1 2 4 5 6 8"}
TRAIN_PAR=${TRAIN_PAR:-6}
L=$W/logs/dp; mkdir -p $L /workspace/dp/data
cd $CP
for t in $TASKS; do
  [ -f /workspace/dp/data/t$t.npz ] || echo "$VP /workspace/code/dp_data.py --task $t > $L/data_t$t.log 2>&1"
done > $L/data_jobs.txt
xargs -P 6 -I{} bash -c "{}" < $L/data_jobs.txt
echo "DATA_DONE $(date -u +%H:%M:%S)"
for t in $TASKS; do
  [ -f /workspace/dp/ckpt/t${t}_s0/ckpt_50000.pt ] || echo "$VP /workspace/code/dp_train.py --task $t > $L/train_t$t.log 2>&1"
done > $L/train_jobs.txt
xargs -P $TRAIN_PAR -I{} bash -c "{}" < $L/train_jobs.txt
echo "TRAIN_DONE $(date -u +%H:%M:%S)"
for t in $TASKS; do echo "$VP /workspace/code/dp_eval.py --task $t --episodes 20 > $L/eval_t$t.log 2>&1"; done > $L/eval_jobs.txt
xargs -P 6 -I{} bash -c "{}" < $L/eval_jobs.txt
echo "EVAL_DONE $(date -u +%H:%M:%S)"
