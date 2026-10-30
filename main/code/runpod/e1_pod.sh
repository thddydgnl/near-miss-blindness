#!/bin/bash
# B E1 (M2) pod chain. Usage on the pod (after rsync and code push):
#   setsid nohup bash /workspace/code/e1_pod.sh <role> [shard] > /workspace/logs/e1_pod.log 2>&1 < /dev/null &
# roles
#   judge_a        pod 1: setup, then regenerate judge (a) with the E0b code/data/seed (gen_e0b.py t1-t8 d00-d19 ->
#                  judge_a.py --stage cnn --train_demos 0-16) and install its weights for the runner
#   cand <i/N>     E1 candidate stage (physics only, stage e1cand) on shard i/N of /workspace/manifests/e1_candidates.jsonl
#   setup          setup only (phase-2 pods)
# Every role also downloads the VLM (d) weights (Qwen2.5-VL-7B) into /workspace/cache/hf_vlm/hub in the background.
ROLE=$1
SHARD=${2:-0/1}
source /workspace/code/env.sh
mkdir -p $W/logs $W/manifests $W/judges
if [ ! -f $W/logs/setup.done ]; then
  WITH_COSMOS=1 bash /workspace/code/setup_pod_v2.sh > $W/logs/setup.log 2>&1
  grep -q SETUP_ALL_DONE $W/logs/setup.log && touch $W/logs/setup.done
fi
echo "setup $(test -f $W/logs/setup.done && echo ok || echo FAILED) $(date -u +%H:%M:%S)"
if [ ! -f $W/logs/vlm.done ]; then
  ( $VP - > $W/logs/vlm_download.log 2>&1 <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="/workspace/cache/hf_vlm/hub"))
PY
    [ $? -eq 0 ] && touch $W/logs/vlm.done ) &
fi
cd $CP
case $ROLE in
  judge_a)
    echo "gen start $(date -u +%H:%M:%S)"
    $VP /workspace/code/gen_e0b.py --workers 7 --out /workspace/e0b/cf_v2 --tasks 1 2 4 5 6 8 > $W/logs/e1_gen_judge_data.log 2>&1
    echo "gen rc=$? $(date -u +%H:%M:%S) $(ls /workspace/e0b/cf_v2/*.json | wc -l) cf"
    $VP /workspace/code/judge_a.py --stage cnn --cf /workspace/e0b/cf_v2 --out $W/results/judge_a_e1 --train_demos 0-16 \
      --tag _e1 > $W/logs/judge_a_e1_cnn.log 2>&1
    echo "cnn rc=$? $(date -u +%H:%M:%S)"
    cp /workspace/e0b/judge_a_cnn_e1.pt $W/judges/judge_a_cnn.pt
    sha256sum /workspace/e0b/judge_a_cnn_e1.pt $W/judges/judge_a_cnn.pt
    echo "JUDGE_A_E1_DONE $(date -u +%H:%M:%S)"
    ;;
  cand)
    $VP /workspace/code/b_runner.py run --manifest $W/manifests/e1_candidates.jsonl --stage e1cand --cpu 7 \
      --shard $SHARD --cf_render none > $W/logs/e1cand_runner.log 2>&1
    echo "cand rc=$? $(date -u +%H:%M:%S)"
    echo "E1CAND_DONE $(date -u +%H:%M:%S)"
    ;;
  setup)
    echo "SETUP_ROLE_DONE $(date -u +%H:%M:%S)"
    ;;
esac
wait
echo "vlm $(test -f $W/logs/vlm.done && echo ok || echo FAILED) $(date -u +%H:%M:%S)"
echo "E1_POD_CHAIN_END $(date -u +%H:%M:%S)"
