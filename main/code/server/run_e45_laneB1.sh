#!/bin/bash
# B E4/E5 lane B, part 1 (GPU 1): M1 E1-cell caches for re-imagination, E4c (M1, lambda), M3 E5 frames and tables.
source $HOME/pmx/B/code/benv.sh
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap\|num_action\|num_proprio\|proprio_dim\|action_dim\|Model emb'
export B_RES=$HOME/pmx/B/results/E4/m1 B_FEAT=$HOME/pmx/B/feat/E4/m1
for cell in "val 5 r" "val 5 r_obj" "trainsplit 5 r"; do
  set -- $cell
  [ -f $B_FEAT/cell_$1_H$2_$3.npz ] || $BPY_DINOWM e4_m1_eval.py cache $1 $2 $3 2>&1 | grep --line-buffered "CACHED\|Traceback\|Error"
done
[ -f $B_RES/e4c_m1.json ] || $BPY_DINOWM e4c_m1.py 2>&1 | grep --line-buffered -v -i "$F"
export B_M3_FEAT=$HOME/pmx/B/feat/E1/m3 R29_CK=$HOME/pmx/B/ckpt/E1/m3 B_E5_FEAT=$HOME/pmx/B/feat/E5/m3
(cd m3 && $BPY_DINOWM m3_e5_data.py 0,1,2 2>&1 | grep --line-buffered -v -i "FutureWarning\|weights_only\|torch.load\|xFormers\|warnings.warn")
echo "E45_LANEB1_DONE $(date -u +%FT%TZ)"
