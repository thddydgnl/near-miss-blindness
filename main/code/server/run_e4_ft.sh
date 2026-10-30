#!/bin/bash
# B E4a / E5 M1: fine-tuning runs from a shared ordered list; each item is claimed with an atomic lock directory so that
# two lanes can work through the same list (B keeps <= 2 GPU jobs on GPU 1). bw runs wait for lambda from E4c.
# usage: run_e4_ft.sh S97_s0,S50_s0,...
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E4/m1 B_FEAT=$HOME/pmx/B/feat/E4/m1
CKD=$HOME/pmx/B/ckpt/E4/m1
mkdir -p $CKD
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap\|num_action\|num_proprio\|proprio_dim\|action_dim\|Model emb'
for item in $(echo $1 | tr ',' ' '); do
  cond=${item%_s*}; seed=${item##*_s}
  [ -f $CKD/${cond}_s${seed}.pt ] && continue
  mkdir $CKD/lock_${item} 2>/dev/null || continue
  LAM=1.0
  if [ "$cond" = "bw" ]; then
    until [ -f $B_RES/e4c_m1.json ]; do sleep 30; done
    LAM=$($HOME/pmx/envs/b-ana/bin/python -c "import json; print(json.load(open('$B_RES/e4c_m1.json'))['lambda_block'])")
  fi
  $BPY_DINOWM e4_m1_finetune.py --cond $cond --seed $seed --lam $LAM 2>&1 | grep --line-buffered -v -i "$F"
done
echo "E4_FT_LANE_DONE $(date -u +%FT%TZ)"
