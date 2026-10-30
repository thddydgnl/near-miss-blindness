#!/bin/bash
# Mac: pull one E1 pod's phase-2 outputs (results/, small/ features, filtered logs) into the scratchpad.
# Usage: e1_pull_pod.sh <n>
R="/Users/song-yonghwi/Documents/physical ai - Neural Networks/runpod_main"
S="/private/tmp/claude-501/-Users-song-yonghwi-Documents-physical-ai---Neural-Networks/d3639ab6-a0e1-42cf-ad6b-e084af659cc6/scratchpad/e1"
N=$1
D="$S/pulled/e1/pod$N"
mkdir -p "$D/results" "$D/small" "$D/logs"
bash "$R/sync.sh" b-e1.$N pull /workspace/runs/e1/results/ "$D/results/" || exit 1
bash "$R/sync.sh" b-e1.$N pull /workspace/runs/e1/small/ "$D/small/" || exit 1
bash "$R/rsh.sh" b-e1.$N 'grep -E "^\[runner\]|Traceback|Error" /workspace/logs/e1_runner.log > /workspace/logs/e1_runner_filtered.log; cp /workspace/logs/e1_phase2.log /workspace/logs/e1_phase2_copy.log' || exit 1
bash "$R/sync.sh" b-e1.$N pull /workspace/logs/e1_runner_filtered.log "$D/logs/" || exit 1
bash "$R/sync.sh" b-e1.$N pull /workspace/logs/e1_phase2.log "$D/logs/" || exit 1
bash "$R/sync.sh" b-e1.$N pull /workspace/logs/e1_pod.log "$D/logs/" || exit 1
echo "pod$N: $(ls "$D/results/imagine" | wc -l) imagine, $(ls "$D/results/cf" | wc -l) cf, $(ls "$D/small" | wc -l) small"
