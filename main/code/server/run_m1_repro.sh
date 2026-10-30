#!/bin/bash
# B E0 M1: R2-12 pipeline on the server (GPU 1), resumable (each step skipped if its output exists).
# smoke -> sim calibration H5/H10 -> judge-training set -> judges -> evaluation sets main/H10/trainsplit -> analysis.
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E0/m1 B_FEAT=$HOME/pmx/B/feat/E0/m1
mkdir -p $B_RES $B_FEAT $B_RES/logs
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration'
step() {  # name, output file, command...
  local n=$1 o=$2; shift 2
  if [ -e "$o" ]; then echo "[skip] $n ($o exists)"; return 0; fi
  echo "[run] $n $(date -u +%H:%M:%S)"
  "$@" 2>&1 | grep --line-buffered -v -i "$F" > $B_RES/logs/$n.log
  local rc=${PIPESTATUS[0]}
  echo "[done] $n rc=$rc $(date -u +%H:%M:%S)"; return $rc
}
step smoke $B_RES/logs/smoke.ok bash -c "$BPY_DINOWM smoke12.py && touch $B_RES/logs/smoke.ok" || exit 1
step calib5 $B_RES/policies_H5.json $BPY_DINOWM calib12.py 5 || exit 1
step calib10 $B_RES/policies_H10.json $BPY_DINOWM calib12.py 10 || exit 1
step clfset $B_FEAT/clf.pt $BPY_DINOWM clfset12.py || exit 1
step judges $B_FEAT/judges.pt $BPY_DINOWM judges12.py || exit 1
for S in main H10 trainsplit; do
  step main_$S $B_RES/logs/main_$S.ok bash -c "$BPY_DINOWM main12.py $S && touch $B_RES/logs/main_$S.ok" || exit 1
done
step analyze $B_RES/criteria.json $BPY_DINOWM analyze12.py main H10 trainsplit || exit 1
step posthoc $B_RES/posthoc.json $BPY_DINOWM posthoc12.py || exit 1
echo M1_REPRO_DONE
