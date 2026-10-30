#!/bin/bash
# B E1 (M2) phase 2 on one pod: the selected counterfactuals (cf regenerated with renders) + end-aligned imagination
# k = 1..4 + lift, judges (a) CNN, (c) value head, (d) VLM 7B. Refuses to start unless the regenerated judge (a)
# weights (sha256 given) and the VLM weights are in place.
# Usage on the pod: setsid nohup bash /workspace/code/e1_phase2.sh <i/N> <sha256 of judge_a_cnn.pt> > /workspace/logs/e1_phase2.log 2>&1 < /dev/null &
SHARD=$1
SHA=$2
source /workspace/code/env.sh
echo "$SHA  /workspace/judges/judge_a_cnn.pt" | sha256sum -c - || { echo "E1_PHASE2_REFUSED judge (a) weights"; exit 1; }
test -f /workspace/logs/vlm.done || { echo "E1_PHASE2_REFUSED no VLM"; exit 1; }
test -f /workspace/manifests/e1_selected.jsonl || { echo "E1_PHASE2_REFUSED no manifest"; exit 1; }
cd $CP
for attempt in 1 2 3; do  # the runner skips finished jobs: a rerun only redoes what failed or was interrupted
  B_VLM_CACHE=/workspace/cache/hf_vlm/hub $VP /workspace/code/b_runner.py run --manifest /workspace/manifests/e1_selected.jsonl \
    --stage e1 --cpu 6 --shard $SHARD --vlm >> /workspace/logs/e1_runner.log 2>&1
  rc=$?
  echo "runner attempt $attempt rc=$rc $(date -u +%H:%M:%S)"
  [ $rc -eq 0 ] && break
done
$VP /workspace/code/b_runner.py status --manifest /workspace/manifests/e1_selected.jsonl --stage e1
echo "E1_PHASE2_DONE $(date -u +%H:%M:%S)"
