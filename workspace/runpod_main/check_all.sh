#!/bin/bash
# Hourly health check for the main experiments: server jobs (tmux, GPU, disk, recent logs) and RunPod pods (spend, liveness).
# Usage: check_all.sh
R="/Users/song-yonghwi/Documents/physical ai - Neural Networks"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) server"
ssh -o ConnectTimeout=20 kiis-mvf-gpu 'bash -s' <<'EOS'
tmux ls 2>/dev/null || echo "no tmux sessions"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
df -h ~ | tail -1
for f in $(ls -t ~/pmx/*/logs/*.log ~/pmx/logs/*.log 2>/dev/null | head -10); do
  age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
  err=$(grep -c -i -E "traceback|exception|error:|nan loss|out of memory|killed" "$f")
  echo "--- $f (updated ${age} min ago, error-like lines: $err)"
  tail -n 3 "$f" | cut -c1-200
done
EOS
echo "=== RunPod"
python3 "$R/runpod_main/budget.py" status
for env in "$R"/runpod_main/pods/*.env; do
  [ -f "$env" ] || continue
  tag=$(basename "$env" .env)
  echo "--- pod $tag"
  bash "$R/runpod_main/rsh.sh" "$tag" 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; for f in $(ls -t /workspace/logs/*.log 2>/dev/null | head -3); do echo "$f"; tail -n 2 "$f" | cut -c1-200; done' 2>&1 | head -12
done
pgrep -f "budget.py watch" > /dev/null && echo "watchdog: alive" || echo "watchdog: NOT RUNNING"
