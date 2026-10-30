#!/bin/bash
# B E2/E3 lane 1 (GPU 1): after both DP rollout lanes -> DP mode-O windows (a_std judgments + decoded frames) ->
# regeneration and decoding of the E1 val H5 cells for judge (d). B keeps <= 2 GPU jobs on GPU 1.
source $HOME/pmx/B/code/benv.sh
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap'
until grep -q "DP_LANE_A_DONE" $HOME/pmx/B/logs/B-E2-dpA.log && grep -q "DP_LANE_B_DONE" $HOME/pmx/B/logs/B-E2-dpB.log; do sleep 20; done
export B_RES=$HOME/pmx/B/results/E2/m1 B_FEAT=$HOME/pmx/B/feat/E2/m1
[ -f $B_RES/dp_windows.jsonl ] || $BPY_DINOWM e2_dp_windows.py 2>&1 | grep -v -i "$F"
export B_RES=$HOME/pmx/B/results/E3/m1 B_FEAT=$HOME/pmx/B/feat/E3/m1
for cell in ${E3_CELLS:-"5 r" "5 r_obj"}; do
  set -- $cell
  [ -f $B_FEAT/e1dec_val_H$1_$2.npz ] || $BPY_DINOWM e3_regen_e1.py val $1 $2 2>&1 | grep -v -i "$F" | grep "batch 0 \|batch 5 \|E3_REGEN_DONE\|Traceback\|Error"
done
echo "E2E3_LANE1_DONE $(date -u +%FT%TZ)"
