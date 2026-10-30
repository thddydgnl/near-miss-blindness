#!/bin/bash
# B E4/E5 lane B, part 2 (GPU 1; restarted 05:21Z with a new order so that the hypothesis runs come first):
# remaining M3 prescription models -> M3 E5 evaluation -> H5b MC dropout (dev, eval) -> joins the M1 fine-tuning list
# (shared, locked items) -> M3 resolution sweep (banks R1/R2, 12 models, evaluation). B keeps <= 2 GPU jobs on GPU 1.
source $HOME/pmx/B/code/benv.sh
cd $BCODE
export B_M3_FEAT=$HOME/pmx/B/feat/E1/m3 R29_CK=$HOME/pmx/B/ckpt/E1/m3 B_E5_FEAT=$HOME/pmx/B/feat/E5/m3 B_E5_CK=$HOME/pmx/B/ckpt/E5/m3 B_E5_OUT=$HOME/pmx/B/results/E5/m3
W='FutureWarning\|weights_only\|torch.load\|xFormers\|warnings.warn'
cd m3
LAM=$($HOME/pmx/envs/b-ana/bin/python -c "import json; print(json.load(open('$HOME/pmx/B/results/E4/m3/e4c_m3.json'))['lambda_obj'])")
for p in 0.03 0.1; do
  for presc in rw near full cp bw; do
    for s in 0 1 2; do
      $BPY_DINOWM wm_e5.py --p $p --presc $presc --seed $s --lam $LAM 2>&1 | grep --line-buffered -v -i "$W"
    done
  done
done
$BPY_DINOWM m3_e5_eval.py 2>&1 | grep --line-buffered -v -i "$W"
$BPY_DINOWM m3_e5_eval.py p0.03_std_1x_s0,p0.03_std_1x_s1,p0.03_std_1x_s2,p0.1_std_1x_s0,p0.1_std_1x_s1,p0.1_std_1x_s2 2>&1 | grep --line-buffered -v -i "$W"
echo "E45_LANEB2_M3E5_DONE $(date -u +%FT%TZ)"
cd ..
export B_RES=$HOME/pmx/B/results/E5/m1 B_FEAT=$HOME/pmx/B/feat/E5/m1
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap\|num_action\|num_proprio\|proprio_dim\|action_dim\|Model emb'
for st in dev eval; do $BPY_DINOWM e5_h5b_mcdrop.py $st 2>&1 | grep --line-buffered -v -i "$F" | grep --line-buffered "DONE\|Traceback\|Error\|sample 16"; done
echo "E45_LANEB2_H5B_DONE $(date -u +%FT%TZ)"
bash $BCODE/run_e4_ft.sh $(cat $HOME/pmx/B/ckpt/E4/m1/ft_order.txt)
echo "E45_LANEB2_FT_DONE $(date -u +%FT%TZ)"
cd m3
for v in R1 R2; do
  [ -f $HOME/pmx/B/feat/E4/m3_sweep/$v/bank_info.json ] || $BPY_DINOWM sweep.py bank $v 2>&1 | grep --line-buffered -v -i "$W"
done
for v in R3 R4 R1 R2; do
  for s in 0 1 2; do $BPY_DINOWM sweep.py train $v $s 2>&1 | grep --line-buffered -v -i "$W"; done
  $BPY_DINOWM sweep.py eval $v 2>&1 | grep --line-buffered -v -i "$W"
done
echo "E45_LANEB2_DONE $(date -u +%FT%TZ)"
