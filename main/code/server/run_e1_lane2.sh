#!/bin/bash
# B E1 lane 2 (GPU 1): after the M1 judges -> a_std ensemble -> the 16 M1 curve cells (val first; primary cell first).
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E1/m1 B_FEAT=$HOME/pmx/B/feat/E1/m1
until grep -q "JUDGES_E1_DONE\|Traceback" $HOME/pmx/B/logs/B-E1-judges.log; do sleep 20; done
grep -q "JUDGES_E1_DONE" $HOME/pmx/B/logs/B-E1-judges.log || { echo "judges failed"; exit 1; }
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap'
# a_std runs in its own session (run_e1_astd.sh); every judge must be fixed before any evaluation set is imagined
until [ -f $B_RES/astd_e1.json ]; do sleep 20; done
for st in ${E1_SETS:-val}; do
  for cell in "5 r" "3 r" "7 r" "10 r" "5 r_obj" "3 r_obj" "7 r_obj" "10 r_obj"; do
    set -- $cell
    [ -f $B_RES/eval_${st}_H$1_$2.jsonl.done ] || $BPY_DINOWM e1_m1_eval.py $st $1 $2 2>&1 | grep -v -i "$F" | grep "EVAL_CELL_DONE\|Traceback\|Error\|batch 0 \|batch 5 "
  done
done
echo "LANE2_DONE ${E1_SETS:-val} $(date -u +%FT%TZ)"
