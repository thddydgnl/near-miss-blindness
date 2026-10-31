#!/bin/bash
# B E7 chain on the pod (DESIGN_main section 12), resumable: each stage leaves <stage>.done in /workspace/logs.
# Usage: setsid nohup bash /workspace/pmx/B/code/run_e7_chain.sh > /workspace/logs/e7_chain.log 2>&1 < /dev/null &
source /workspace/pmx/B/code/e7_env.sh
cd $BCODE
L=$W/logs; mkdir -p $L $B_RES $B_FEAT
stage() {   # stage <name> <command...>
  local n=$1; shift
  if [ -f $L/$n.done ]; then echo "skip $n"; return 0; fi
  echo "START $n $(date -u +%FT%TZ)"
  "$@" > $L/$n.log 2>&1; local rc=$?
  echo "END $n rc=$rc $(date -u +%FT%TZ)"
  [ $rc -eq 0 ] && touch $L/$n.done
  return $rc
}
stage smoke $VP e7_smoke.py || { echo "E7_STOP smoke"; exit 1; }
stage calib $VP e1_calib.py || { echo "E7_STOP calib"; exit 1; }
stage convcheck $VP e7_convcheck.py || { echo "E7_STOP convcheck (rule 12.8)"; exit 2; }
stage judgeset $VP e1_judgeset.py || { echo "E7_STOP judgeset"; exit 1; }
stage judges $VP e1_judges.py || { echo "E7_STOP judges"; exit 1; }
# admission rule (12.7): ensemble held-out accuracy of a_full on real frames >= 0.95, else stop (never lowered)
$VP - <<'PY' || { echo "E7_STOP judge admission (rule 12.7)"; exit 3; }
import json, os, sys
j = json.load(open(os.environ['B_RES'] + '/judges_e7.json'))
acc = j['a_full']['ensemble']['acc']
print('a_full ensemble held-out acc', acc, 'PASS' if acc >= 0.95 else 'FAIL')
sys.exit(0 if acc >= 0.95 else 1)
PY
for cell in "val 5" "val 3" "val 7" "val 10" "trainsplit 5" "trainsplit 3" "trainsplit 7" "trainsplit 10"; do
  set -- $cell
  stage eval_$1_H$2 $VP e1_m1_eval.py $1 $2 r || { echo "E7_STOP eval $1 H$2"; exit 1; }
done
stage analyze $VP e1_m1_analyze.py 1000 || { echo "E7_STOP analyze"; exit 1; }
stage figs $VP e7_figs.py || echo "figs failed (non-fatal)"
echo "E7_CHAIN_DONE $(date -u +%FT%TZ)"
