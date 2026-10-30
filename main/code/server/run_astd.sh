#!/bin/bash
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E0/dp B_FEAT=$HOME/pmx/B/feat/E0/dp
R=$B_RES/rollouts
cd $BCODE
$BPY_DINOWM dp_imagine.py --rollouts $R/demo_s43_ep0025.pt,$R/demo_s43_ep0030.pt,$R/demo_s43_ep0050.pt --H 3,5,7,10 --out $B_FEAT/demo_imag.pt
$BPY_DINOWM astd_judge_e0.py $B_FEAT/demo_imag.pt
