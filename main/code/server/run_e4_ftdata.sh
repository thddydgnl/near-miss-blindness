#!/bin/bash
# B E4/E5 M1: fine-tuning data (pool, S97 / S50 selections, E5 counterfactuals). GPU 1 (DINOv2 encoding), CPU workers.
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E4/m1 B_FEAT=$HOME/pmx/B/feat/E4/m1
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap'
[ -f $B_FEAT/ft_meta.npz ] || $BPY_DINOWM e4_m1_ftdata.py 2>&1 | grep --line-buffered -v -i "$F"
echo "E4_FTDATA_LANE_DONE $(date -u +%FT%TZ)"
