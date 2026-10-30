#!/bin/bash
# B E2/E3 lane 2 (GPU 1): after DP rollout lane A -> judge (d) rule sets (dset, std; decoded real frames) -> M3 policy
# evaluation of the 9 E1 models -> M1 control set (R2-12 main) with the E1 judges. B keeps <= 2 GPU jobs on GPU 1.
source $HOME/pmx/B/code/benv.sh
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap'
until grep -q "DP_LANE_A_DONE" $HOME/pmx/B/logs/B-E2-dpA.log; do sleep 20; done
export B_RES=$HOME/pmx/B/results/E3/m1 B_FEAT=$HOME/pmx/B/feat/E3/m1
[ -f $B_FEAT/rules_dset.npz ] || $BPY_DINOWM e3_rules_dset.py 2>&1 | grep -v -i "$F"
[ -f $B_FEAT/rules_std.npz ] || $BPY_DINOWM e3_rules_std.py 2>&1 | grep -v -i "$F"
export B_M3_FEAT=$HOME/pmx/B/feat/E1/m3 R29_CK=$HOME/pmx/B/ckpt/E1/m3 R29_OUT=$HOME/pmx/B/results/E2/m3
mkdir -p $R29_OUT
(cd m3 && [ -f $R29_OUT/e2_m3_meta.json ] || $BPY_DINOWM m3_e2_eval.py 2>&1 | grep -v -i "FutureWarning\|weights_only\|torch.load\|xFormers\|warnings.warn")
export B_RES=$HOME/pmx/B/results/E2/m1 B_FEAT=$HOME/pmx/B/feat/E2/m1
[ -f $B_RES/ctrl_main.jsonl ] || $BPY_DINOWM e2_ctrl.py 2>&1 | grep -v -i "$F"
echo "E2E3_LANE2_DONE $(date -u +%FT%TZ)"
