#!/bin/bash
# B E1 lane 1 (GPU 1): after the M3 models -> E1 grid bank -> M3 curves for the 9 models.
source $HOME/pmx/B/code/benv.sh
export B_M3_FEAT=$HOME/pmx/B/feat/E1/m3 R29_CK=$HOME/pmx/B/ckpt/E1/m3 R29_OUT=$HOME/pmx/B/results/E1/m3
until grep -q "M3_TRAIN_DONE" $HOME/pmx/B/logs/B-E1-m3train.log; do sleep 20; done
cd $BCODE/m3
F='FutureWarning\|weights_only\|torch.load\|xFormers\|warnings.warn'
[ -f $B_M3_FEAT/eval_e1.pt ] || $BPY_DINOWM world_e1.py 2>&1 | grep -v "$F"
$BPY_DINOWM m3_e1_eval.py 2>&1 | grep -v "$F"
echo "LANE1_DONE $(date -u +%FT%TZ)"
