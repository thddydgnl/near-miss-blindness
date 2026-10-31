#!/bin/bash
# Hourly health check for the B E7 pod: spend, liveness, GPU use and the newest log lines per pod.
# Usage: check_all.sh
N="/Users/song-yonghwi/Documents/physical ai - Neural Networks/B_near_miss_blindness/workspace/runpod_e7"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) RunPod (bx7)"
python3 "$N/budget.py" status
for env in "$N"/pods/*.env; do
  [ -f "$env" ] || continue
  tag=$(basename "$env" .env)
  echo "--- pod $tag"
  bash "$N/rsh.sh" "$tag" 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; for f in $(ls -t /workspace/logs/*.log 2>/dev/null | head -3); do echo "$f"; tail -n 2 "$f" | cut -c1-200; done' 2>&1 | head -12
done
pgrep -f "budget.py watch" > /dev/null && echo "watchdog: alive" || echo "watchdog: NOT RUNNING"
