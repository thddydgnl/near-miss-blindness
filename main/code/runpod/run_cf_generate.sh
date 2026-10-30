#!/bin/bash
# CureWM counterfactual generation on LIBERO-Goal, unmodified engine (scripts/generate_libero.py), one process per task.
# 6 families x 5 severities x 1 seed per demo (as in their held-out probes); demos = first N official demos per task.
# Usage: bash run_cf_generate.sh <demos_per_task> <parallel> [tasks...]
source /workspace/code/env.sh
N=${1:-2}; P=${2:-6}; shift 2
TASKS=${@:-0 1 2 3 4 5 6 7 8 9}
OUT=$W/data/cf_goal
mkdir -p $OUT $W/logs/cf
cd $RB/src/CureWM
for t in $TASKS; do
  if [ -f $OUT/t$t/sanity_report.json ]; then echo "skip t$t" >&2; continue; fi
  [ -d $OUT/t$t ] && mv $OUT/t$t $OUT/t$t.partial-$(date +%H%M%S)   # partial run: generate_pairs appends, so set it aside
  echo "PYTHONPATH=src PYTHONHASHSEED=0 OMP_NUM_THREADS=1 $VP scripts/generate_libero.py --suite libero_goal --demos-per-task $N --tasks $t --out $OUT/t$t > $W/logs/cf/t$t.log 2>&1"
done > $W/logs/cf/jobs.txt
T0=$(date +%s)
xargs -P $P -I{} bash -c "{}" < $W/logs/cf/jobs.txt
echo "CF_GENERATE_DONE $(( $(date +%s) - T0 ))s"
