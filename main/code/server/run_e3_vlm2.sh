#!/bin/bash
# B E3 judge (d) lane, v2 (00:25Z: batch 64 and GPU budget 11 GiB after the DP windows job ended; v1 used batch 32 / 9 GiB
# and was restarted before its first rule set finished, all scores come from this configuration). Starts after lane 2 so that B
# keeps <= 2 GPU jobs on GPU 1. Order (DESIGN_main v2.2 section 7): the three rule sets -> rule fixed (e3_d_rules.py) ->
# evaluation sets E1 val H5 r, DP H5, E1 val H5 r_obj (further cells only if throughput allows, decided afterwards).
source $HOME/pmx/B/code/benv.sh
cd $BCODE
VP=$HOME/pmx/envs/b-vlm/bin/python
AP=$HOME/pmx/envs/b-ana/bin/python
F3=$HOME/pmx/B/feat/E3/m1; F2=$HOME/pmx/B/feat/E2/m1; O=$HOME/pmx/B/results/E3/m1/vlm
GM=${GPU_MEM:-11GiB}; BS=${VLM_BS:-64}
mkdir -p $O
F='Warning\|warn(\|torch_dtype\|Fetching\|it/s\]\|Loading checkpoint\|^$'
until grep -q "E2E3_LANE2_DONE" $HOME/pmx/B/logs/B-E2-lane2.log; do sleep 30; done
$VP e3_vlm.py --npz $F3/rules_dset.npz --variant dset_r --pairs goal:final --out $O/rules_dset_r --gpu_mem $GM --bs $BS 2>&1 | grep --line-buffered -v "$F"
$VP e3_vlm.py --npz $F3/rules_dset.npz --variant dset_robj --pairs goal:final --out $O/rules_dset_robj --gpu_mem $GM --bs $BS 2>&1 | grep --line-buffered -v "$F"
$VP e3_vlm.py --npz $F3/rules_std.npz --variant std --pairs goal:final --out $O/rules_std --gpu_mem $GM --bs $BS 2>&1 | grep --line-buffered -v "$F"
$AP e3_d_rules.py
echo "D_RULES_FIXED $(date -u +%FT%TZ)"
until [ -f $F3/e1dec_val_H5_r.npz ]; do sleep 30; done
$VP e3_vlm.py --npz $F3/e1dec_val_H5_r.npz --variant dset_r --pairs goal:real,goal:imag --out $O/e1_val_H5_r --gpu_mem $GM --bs $BS 2>&1 | grep --line-buffered -v "$F"
until [ -f $F2/dp_dec_H5.npz ]; do sleep 30; done
$VP e3_vlm.py --npz $F2/dp_dec_H5.npz --variant std --pairs goal:real,goal:imag --out $O/dp_H5 --gpu_mem $GM --bs $BS 2>&1 | grep --line-buffered -v "$F"
until [ -f $F3/e1dec_val_H5_r_obj.npz ]; do sleep 30; done
$VP e3_vlm.py --npz $F3/e1dec_val_H5_r_obj.npz --variant dset_robj --pairs goal:real,goal:imag --out $O/e1_val_H5_r_obj --gpu_mem $GM --bs $BS 2>&1 | grep --line-buffered -v "$F"
echo "E3_VLM_LANE_DONE $(date -u +%FT%TZ)"
