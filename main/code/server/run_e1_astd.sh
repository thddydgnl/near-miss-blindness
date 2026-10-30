#!/bin/bash
# B E1: a_std ensemble, started once the M3 lane has released its GPU slot (B keeps <= 2 GPU jobs on GPU 1).
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E1/m1 B_FEAT=$HOME/pmx/B/feat/E1/m1
until grep -q "LANE1_DONE" $HOME/pmx/B/logs/B-E1-lane1.log; do sleep 20; done
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap'
[ -f $B_RES/astd_e1.json ] || $BPY_DINOWM e1_astd.py 2>&1 | grep -v -i "$F"
echo "ASTD_LANE_DONE $(date -u +%FT%TZ)"
