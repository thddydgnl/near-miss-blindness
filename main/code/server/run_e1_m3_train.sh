#!/bin/bash
# B E1 M3 part 1 (GPU 1): R2-9 frame banks (seeds 0,1,2 in one call, as R2-9), frozen classifier, and the 9 models
# p in {0.03, 0.1, 0.3} x seed {0,1,2}, 1x (T1 = 6000). Resumable. Afterwards: bit-identity check against the E0
# models that overlap (same code, seeds, GPU model).
source $HOME/pmx/B/code/benv.sh
export B_M3_FEAT=$HOME/pmx/B/feat/E1/m3 R29_CK=$HOME/pmx/B/ckpt/E1/m3 R29_OUT=$HOME/pmx/B/results/E1/m3
mkdir -p $B_M3_FEAT $R29_CK $R29_OUT
cd $BCODE/m3
F='FutureWarning\|weights_only\|torch.load\|xFormers\|warnings.warn\|^  [A-Za-z_]* = torch.load\|^  [a-z_]*\.load_state_dict'
t0=$(date +%s)
if [ ! -f $B_M3_FEAT/bank_s2_N30000.pt ]; then $BPY_DINOWM world.py 0,1,2 30000 2>&1 | grep -v "$F"; fi
echo "banks $(( $(date +%s) - t0 ))s"
if [ ! -f $R29_CK/classifier.pt ]; then $BPY_DINOWM wm.py clf 2>&1 | grep -v "$F"; fi
for p in 0.03 0.1 0.3; do
  case $p in 0.03) data=std;; 0.1) data=p01;; 0.3) data=p03;; esac
  for s in 0 1 2; do
    n=p${p}_std_1x_s$s
    if [ ! -f $R29_CK/$n.pt ]; then $BPY_DINOWM wm.py train --data $data --mode std --seed $s --T 6000 --name $n 2>&1 | grep -v "$F" | tail -1; fi
  done
done
$BPY_DINOWM - <<PY
import torch, os
for n in ['p0.03_std_1x_s0', 'p0.03_std_1x_s1', 'p0.03_std_1x_s2', 'p0.1_std_1x_s0', 'p0.3_std_1x_s0']:
    a = torch.load(os.path.expanduser('~/pmx/B/ckpt/m3/%s.pt' % n), map_location='cpu', weights_only=False)['state']
    b = torch.load(os.path.expanduser('~/pmx/B/ckpt/E1/m3/%s.pt' % n), map_location='cpu', weights_only=False)['state']
    d = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    print('E0 vs E1 weights', n, 'max abs diff', d)
a = torch.load(os.path.expanduser('~/pmx/B/ckpt/m3/classifier.pt'), map_location='cpu'); b = torch.load(os.path.expanduser('~/pmx/B/ckpt/E1/m3/classifier.pt'), map_location='cpu')
print('classifier max abs diff', max(float((a[k] - b[k]).abs().max()) for k in a))
PY
echo "M3_TRAIN_DONE $(( $(date +%s) - t0 ))s"
