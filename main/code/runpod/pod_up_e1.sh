#!/bin/bash
# E1 (M2) helper on the Mac: create one more A40 pod with tag b-e1 through runpod_main/pod_up.sh (so the ledger tag, and
# with it the watchdog cap for b-e1, is the same for every E1 pod), then give its env/ssh files a per-pod name
# (pod_up.sh keeps one pods/<tag>.env per tag). Afterwards use rsh.sh / sync.sh / pod_down.sh with tag b-e1.<n>.
# Usage: pod_up_e1.sh <n>
R="/Users/song-yonghwi/Documents/physical ai - Neural Networks/runpod_main"
N=$1
[ -n "$N" ] || { echo "usage: pod_up_e1.sh <n>"; exit 1; }
[ -f "$R/pods/b-e1.$N.env" ] && { echo "pods/b-e1.$N.env exists"; exit 1; }
TERM_H=8 bash "$R/pod_up.sh" b-e1 "NVIDIA A40" 60 100 || exit 1
for ext in env json ssh.json; do
  [ -f "$R/pods/b-e1.$ext" ] && mv "$R/pods/b-e1.$ext" "$R/pods/b-e1.$N.$ext"
done
cat "$R/pods/b-e1.$N.env"
