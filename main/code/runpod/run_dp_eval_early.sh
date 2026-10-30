#!/bin/bash
# Quick evaluation (20 held episodes) of the early-stage DP checkpoints (500/1k/2k/3k) once all early runs are done.
source /workspace/code/env.sh
while pgrep -f "ckpt_early > " > /dev/null || pgrep -f "out /workspace/dp/ckpt_early" > /dev/null; do sleep 20; done
cd $CP
for t in 1 2 4 5 6 8; do
  $VP /workspace/code/dp_eval.py --task $t --episodes 20 --ckpt_dir /workspace/dp/ckpt_early --out /workspace/results/dp_eval_early > /workspace/logs/dp/eval_early_t$t.log 2>&1 &
  if [ $t = 4 ]; then wait; fi
done
wait
echo "EVAL_EARLY_DONE $(date -u +%H:%M:%S)"
