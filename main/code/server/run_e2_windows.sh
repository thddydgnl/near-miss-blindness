#!/bin/bash
# B E2 M1: DP mode-O windows (rerun after the replay-identity fix), queued behind lane 1 so that B keeps <= 2 GPU jobs.
source $HOME/pmx/B/code/benv.sh
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap'
until grep -q "E2E3_LANE1_DONE" $HOME/pmx/B/logs/B-E2-lane1.log; do sleep 20; done
export B_RES=$HOME/pmx/B/results/E2/m1 B_FEAT=$HOME/pmx/B/feat/E2/m1
[ -f $B_RES/dp_windows.jsonl ] || $BPY_DINOWM e2_dp_windows.py 2>&1 | grep --line-buffered -v -i "$F"
echo "E2_WINDOWS_LANE_DONE $(date -u +%FT%TZ)"
