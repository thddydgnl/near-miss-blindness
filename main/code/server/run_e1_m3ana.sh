#!/bin/bash
# B E1 M3 analysis + figures (CPU analysis env, few BLAS threads to share the CPU).
source $HOME/pmx/B/code/benv.sh
export R29_OUT=$HOME/pmx/B/results/E1/m3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $BCODE
$HOME/pmx/envs/b-ana/bin/python m3/m3_e1_analyze.py 1000
