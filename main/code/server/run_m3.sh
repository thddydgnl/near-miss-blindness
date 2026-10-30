#!/bin/bash
# B E0 M3: R2-9 controlled model on the server (GPU 1), resumable. Frame banks for seeds 0,1,2 (one call, as R2-9:
# PCA and token normalisation come from this call), frozen classifier, then p = 0.03 std 1x (seeds 0,1,2) and
# p = 0.3 / p = 0.1 std 1x (seed 0), each trained then evaluated, then the R2-9 summary.
source $HOME/pmx/B/code/benv.sh
export B_M3_FEAT=$HOME/pmx/B/feat/E0/m3 R29_CK=$HOME/pmx/B/ckpt/m3 R29_OUT=$HOME/pmx/B/results/E0/m3
mkdir -p $B_M3_FEAT $R29_CK $R29_OUT
cd $BCODE/m3
F='FutureWarning\|weights_only\|torch.load\|xFormers\|warnings.warn\|^  [A-Za-z_]* = torch.load\|^  [a-z_]*\.load_state_dict'
t0=$(date +%s)
if [ ! -f $B_M3_FEAT/bank_s2_N30000.pt ]; then $BPY_DINOWM world.py 0,1,2 30000 2>&1 | grep -v "$F"; fi
echo "banks done $(( $(date +%s) - t0 ))s"
if [ ! -f $R29_CK/classifier.pt ]; then $BPY_DINOWM wm.py clf 2>&1 | grep -v "$F"; fi
run() {   # name data seed
  if [ ! -f $R29_CK/$1.pt ]; then $BPY_DINOWM wm.py train --data $2 --mode std --seed $3 --T 6000 --name $1 2>&1 | grep -v "$F" | tail -3; fi
  if [ ! -f $R29_OUT/eval_$1.json ]; then $BPY_DINOWM evaluate.py $1 2>&1 | grep -v "$F"; fi
}
run p0.03_std_1x_s0 std 0
run p0.3_std_1x_s0 p03 0
run p0.1_std_1x_s0 p01 0
run p0.03_std_1x_s1 std 1
run p0.03_std_1x_s2 std 2
$BPY_DINOWM summarize.py $R29_OUT > $R29_OUT/summarize_console.txt 2>&1
echo "M3_DONE $(( $(date +%s) - t0 ))s"
