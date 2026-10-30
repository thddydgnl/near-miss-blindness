#!/bin/bash
# B E1 M1 analysis + figures (CPU analysis env, few BLAS threads to share the CPU).
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E1/m1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $BCODE
$HOME/pmx/envs/b-ana/bin/python e1_m1_analyze.py 1000
$HOME/pmx/envs/b-ana/bin/python e1_figs.py
echo "M1ANA_DONE $(date -u +%FT%TZ)"
