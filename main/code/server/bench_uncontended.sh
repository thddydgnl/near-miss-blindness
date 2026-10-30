#!/bin/bash
# B E0 throughput on GPU 1 after the DP training finished (GPU 1 otherwise idle; GPU 0 / CPU shared with others).
# 1 M1 R2-12 evaluation sets (main H5 400 ep, H10 300 ep; new tags, features deleted afterwards)
# 2 DP rollouts (final stage, 3 seeds x 50 calibration episodes, batch 50) and one 100-episode batch
# 3 mode-O imagination of 100 DP episodes at H = 5
# 4 M3 one 1x model (p = 0.03 seed 0, temp ckpt dir) train + evaluate (also a determinism check vs E0)
# 5 DINO-WM predictor fine-tuning step
source $HOME/pmx/B/code/benv.sh
cd $BCODE
T=$HOME/pmx/B/results/E0/bench
mkdir -p $T
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache'
tm() { local n=$1; shift; local t0=$(date +%s.%N); "$@" 2>&1 | grep -v -i "$F" | tail -3; local t1=$(date +%s.%N); echo "BENCH $n $(echo "$t1 - $t0" | bc) s"; }
export B_RES=$HOME/pmx/B/results/E0/m1 B_FEAT=$HOME/pmx/B/feat/E0/m1
tm m1_main_400ep_x9 $BPY_DINOWM main12.py main 400
tm m1_H10_300ep_x9 $BPY_DINOWM main12.py H10 300
rm -f $B_FEAT/eval_main_n400_*.pt $B_FEAT/eval_H10_n300_*.pt
export B_RES=$HOME/pmx/B/results/E0/dp B_FEAT=$HOME/pmx/B/feat/E0/dp
for s in 42 43 44; do
  tm dp_rollout_50ep_s$s $BPY_DP dp_rollout.py --ckpt $HOME/pmx/B/ckpt/dp/s$s/ema_ep1000.pt --seeds 200000:200050 --out $B_RES/snapeval/s${s}_ep1000.pt --batch 50
done
tm dp_rollout_100ep $BPY_DP dp_rollout.py --ckpt $HOME/pmx/B/ckpt/dp/s42/ema_ep1000.pt --seeds 900100:900200 --out $B_RES/rollouts/bench_s42_ep1000.pt --batch 100
tm dp_imagine_100ep_H5 $BPY_DINOWM dp_imagine.py --rollouts $B_RES/rollouts/bench_s42_ep1000.pt --H 5 --out $B_FEAT/bench_imag.pt
rm -f $B_FEAT/bench_imag.pt
export B_M3_FEAT=$HOME/pmx/B/feat/E0/m3 R29_CK=$HOME/pmx/B/tmp/m3bench R29_OUT=$HOME/pmx/B/tmp/m3bench_out
mkdir -p $R29_CK $R29_OUT
cp $HOME/pmx/B/ckpt/m3/classifier.pt $R29_CK/
cd $BCODE/m3
tm m3_bank_seed0 $BPY_DINOWM world.py 0 30000
tm m3_train_1x $BPY_DINOWM wm.py train --data std --mode std --seed 0 --T 6000 --name p0.03_std_1x_s0
tm m3_eval $BPY_DINOWM evaluate.py p0.03_std_1x_s0
python3 -c "import json; a=json.load(open('$HOME/pmx/B/results/E0/m3/eval_p0.03_std_1x_s0.json')); b=json.load(open('$R29_OUT/eval_p0.03_std_1x_s0.json')); print('M3 determinism: mean_over', a['mean_over'], b['mean_over'], 'max policy diff', max(abs(a['policies'][k][m]-b['policies'][k][m]) for k in a['policies'] for m in ('imagined','over')))"
rm -rf $HOME/pmx/B/tmp/m3bench $HOME/pmx/B/tmp/m3bench_out $B_M3_FEAT/bank_s0_N30000.pt
cd $BCODE
export B_RES=$HOME/pmx/B/results/E0/bench
tm dinowm_ft_30steps $BPY_DINOWM bench_dinowm_ft.py 30
echo BENCH_ALL_DONE
