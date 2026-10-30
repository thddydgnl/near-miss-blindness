#!/bin/bash
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E0/dp B_FEAT=$HOME/pmx/B/feat/E0/dp
mkdir -p $B_RES
cd $BCODE
$BPY_DP dp_conventions.py 2>&1 | grep --line-buffered -v -i 'warn\|distutils\|Gymnasium\|Please upgrade\|See the migration\|get_cmap'
