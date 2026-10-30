#!/bin/bash
# B E0 feasibility of a standard-goal oracle judge (a_std) for DP episodes: rollouts on dedicated E0 seed ranges
# (judge-training 300000-300099, E0 demo 900000-900099), never the E2 evaluation seeds (100000-).
# usage: rollouts_e0_astd.sh PART   (PART = a|b|c, splits the jobs over three tmux sessions)
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E0/dp
O=$B_RES/rollouts
mkdir -p $O
cd $BCODE
run() {  # seed epoch seedrange tag
  local f=$HOME/pmx/B/ckpt/dp/s$1/ema_ep$2.pt o=$O/$4_s$1_ep$2.pt
  [ -f $o ] || $BPY_DP dp_rollout.py --ckpt $f --seeds $3 --out $o --batch 100 2>&1 | grep SUMMARY | cut -c1-240 | sed "s/^/$4 s$1 ep$2 /"
}
case $1 in
  a) run 42 0020 300000:300100 jt; run 42 0030 300000:300100 jt; run 43 0030 900000:900100 demo ;;
  b) run 42 0025 300000:300100 jt; run 42 0050 300000:300100 jt; run 43 0050 900000:900100 demo ;;
  c) run 42 0040 300000:300100 jt; run 42 0080 300000:300100 jt; run 43 0025 900000:900100 demo ;;
esac
echo ROLLOUTS_$1_DONE
