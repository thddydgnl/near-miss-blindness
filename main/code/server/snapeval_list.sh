#!/bin/bash
# B E0: sim-only evaluation of selected DP snapshots (calibration seeds 200000-200049). usage: snapeval_list.sh SEED EPOCH...
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E0/dp
mkdir -p $B_RES/snapeval
cd $BCODE
s=$1; shift
for e in "$@"; do
  f=$HOME/pmx/B/ckpt/dp/s$s/ema_ep$e.pt
  o=$B_RES/snapeval/s${s}_ep$e.pt
  [ -f $o ] && continue
  [ -f $f ] || { echo "missing $f"; continue; }
  $BPY_DP dp_rollout.py --ckpt $f --seeds 200000:200050 --out $o --batch 50 2>&1 | grep SUMMARY | cut -c1-260 | sed "s/^/s$s ep$e /"
done
echo SNAPEVAL_LIST_DONE
