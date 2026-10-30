#!/bin/bash
# B E4/E5 lane C (GPU 1): once all 21 M1 fine-tunes exist -> re-imagine the three E1 cells with the public model and every
# fine-tuned variant (judges a_full / a_obj, d_vis) -> E4 / E5 analyses (CPU). Runs when lane A has finished, so B keeps
# <= 2 GPU jobs on GPU 1.
source $HOME/pmx/B/code/benv.sh
cd $BCODE
CKD=$HOME/pmx/B/ckpt/E4/m1
V=orig
for c in S97 S50 rw near full cp bw; do for s in 0 1 2; do V=$V,${c}_s$s; done; done
until [ $(ls $CKD/*_s?.pt 2>/dev/null | wc -l) -ge 21 ] && grep -q "E4_FT_LANE_DONE" $HOME/pmx/B/logs/B-E4-ftA.log; do sleep 60; done
export B_RES=$HOME/pmx/B/results/E4/m1 B_FEAT=$HOME/pmx/B/feat/E4/m1
for cell in "val 5 r" "val 5 r_obj" "trainsplit 5 r"; do
  set -- $cell
  $BPY_DINOWM e4_m1_eval.py imagine $1 $2 $3 $V 2>&1 | grep --line-buffered "imagined\|identity\|Traceback\|Error\|E4_EVAL_DONE"
done
echo "E45_LANEC_IMAG_DONE $(date -u +%FT%TZ)"
AP=$HOME/pmx/envs/b-ana/bin/python
$AP e4_analyze.py 2>&1 | tail -n 20
$AP e5_analyze.py 2>&1 | tail -n 12
echo "E45_LANEC_DONE $(date -u +%FT%TZ)"
