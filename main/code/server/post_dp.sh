#!/bin/bash
# B E0: after the DP training finishes, run the GPU-1 throughput benchmark (includes the epoch-1000 stage evaluation of
# the 3 seeds), then the stage summary and figure.
source $HOME/pmx/B/code/benv.sh
until [ -f $HOME/pmx/B/ckpt/dp/s42/DONE ] && [ -f $HOME/pmx/B/ckpt/dp/s43/DONE ] && [ -f $HOME/pmx/B/ckpt/dp/s44/DONE ]; do sleep 20; done
echo "$(date -u +%FT%TZ) DP done; starting benchmark"
bash $BCODE/bench_uncontended.sh
cd $BCODE
export B_RES=$HOME/pmx/B/results/E0/dp
$BPY_DP dp_snapeval_summary.py 2>&1 | grep -v -i warn
$BPY_DINOWM fig_dp_stages.py 2>&1 | grep -v -i warn
echo "$(date -u +%FT%TZ) POST_DP_DONE"
