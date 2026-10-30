#!/bin/bash
# Wait for the first DP checkpoints, then (1) the E2 runner smoke test (task 8, 2 held start states, modes O and C)
# and (2) a 5-episode sanity evaluation of every 5k checkpoint (held start states) to catch pipeline errors early.
source /workspace/code/env.sh
until [ -f /workspace/dp/ckpt/t8_s0/ckpt_5000.pt ]; do sleep 10; done
sleep 20
cd $CP
$VP /workspace/code/b_runner.py run --manifest /workspace/runs/smoke_e2.jsonl --stage smoke_e2 --cpu 1 > /workspace/logs/runner_smoke_e2b.log 2>&1
grep runner /workspace/logs/runner_smoke_e2b.log
for t in 1 2 4 5 6 8; do
  $VP /workspace/code/dp_eval.py --task $t --episodes 5 --ckpt_dir /workspace/dp/ckpt --out /workspace/results/dp_eval_5k_sanity > /workspace/logs/dp/eval5k_t$t.log 2>&1 &
done
wait
echo "SMOKE_E2_DONE $(date -u +%H:%M:%S)"
