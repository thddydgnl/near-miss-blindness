#!/bin/bash
# training-length check (PLAN 4장): p=0.3 std seed 0 at T1=6000; balanced imagined accuracy >= 0.90 keeps T1
set -e
cd /workspace/code9
python wm.py train --data p03 --mode std --seed 0 --T 6000 --name p0.3_std_1x_s0
python evaluate.py p0.3_std_1x_s0
python -c "
import json; r=json.load(open('/workspace/results9/eval_p0.3_std_1x_s0.json'))
acc=0.5*(r['imag_acc_succ']+r['imag_acc_fail']); print('balanced imagined acc', acc, 'T1 =', 6000 if acc>=0.9 else 12000)
json.dump(dict(acc=acc, T1=6000 if acc>=0.9 else 12000), open('/workspace/results9/T1_choice.json','w'))"
echo CAL_OK
