"""Pre-registered training-length rule (PLAN.md 4장): uses ONLY the p=0.3 calibration models.
T = smallest T in {5k,10k,20k,40k} whose seed-mean balanced-test success accuracy and failure accuracy are
both >= 0.97 at the end of that model's own schedule; if none, T = 40k."""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results/cal/final_metrics.json'
out = sys.argv[2] if len(sys.argv) > 2 else '/workspace/results/cal/T_choice.json'
F = json.load(open(path))['finals']
Ts = sorted(set(r['T'] for r in F))
table = []
choice = None
for T in Ts:
    rs = [r for r in F if r['T'] == T and abs(r['p'] - 0.3) < 1e-9 and r['mode'] == 'std']
    a_s = float(np.mean([r['bal_acc_s'] for r in rs]))
    a_f = float(np.mean([r['bal_acc_f'] for r in rs]))
    mse = float(np.mean([r['bal_mse_s'] for r in rs]))
    table.append(dict(T=T, acc_s=a_s, acc_f=a_f, mse_s=mse, n=len(rs)))
    if choice is None and a_s >= 0.97 and a_f >= 0.97:
        choice = T
if choice is None:
    choice = max(Ts)
res = dict(T=choice, table=table)
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
print(choice)
