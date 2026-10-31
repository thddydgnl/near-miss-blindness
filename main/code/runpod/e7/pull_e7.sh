#!/bin/bash
# Mac: pull the E7 results (JSON, JSONL, PNG; no feature tensors) and the pod logs into main/results/E7_runpod/.
# Usage: bash main/code/runpod/e7/pull_e7.sh
B="/Users/song-yonghwi/Documents/physical ai - Neural Networks/B_near_miss_blindness"
R="$B/workspace/runpod_e7"
D="$B/main/results/E7_runpod"
mkdir -p "$D/pm" "$D/pod_logs"
bash "$R/sync.sh" e7 pull /workspace/pmx/B/results/E7/pm/ "$D/pm/" --exclude '*.pt' || exit 1
bash "$R/sync.sh" e7 pull /workspace/logs/ "$D/pod_logs/" --exclude 'apt.log' --exclude 'uv_install.log' || exit 1
ls -la "$D/pm" | head -40
du -sh "$D"
