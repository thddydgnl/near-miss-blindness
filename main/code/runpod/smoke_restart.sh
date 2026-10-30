#!/bin/bash
# Restart / new-pod smoke test of b_runner.py on the E1 smoke manifest (E0b held demos only).
#  (1) interrupted run: start, kill after the first imagine result appears, rerun -> must finish without redoing jobs
#  (2) new pod: copy only the results of run (1) into a fresh root (no artifacts), drop two imagine results, pass the
#      remaining ids as --done -> the runner must regenerate the two cf artifacts and redo only those two imagine jobs;
#      regenerated cf records and imagine values are compared with run (1).
source /workspace/code/env.sh
M=/workspace/runs/smoke_e1.jsonl
cd $CP
# (1)
rm -rf /workspace/runs_restart  # scratch roots created by this test only
$VP /workspace/code/b_runner.py run --manifest $M --stage s1 --root /workspace/runs_restart --cpu 1 > /workspace/logs/smoke_restart_1a.log 2>&1 &
until ls /workspace/runs_restart/s1/results/imagine/*.json > /dev/null 2>&1; do sleep 2; done
bash /workspace/code/killp.sh "runs_restart --cpu"; sleep 5
echo "after kill: $(ls /workspace/runs_restart/s1/results/cf | wc -l) cf, $(ls /workspace/runs_restart/s1/results/imagine | wc -l) imagine"
$VP /workspace/code/b_runner.py run --manifest $M --stage s1 --root /workspace/runs_restart --cpu 1 > /workspace/logs/smoke_restart_1b.log 2>&1
grep runner /workspace/logs/smoke_restart_1b.log
# (2)
mkdir -p /workspace/runs_newpod/s1
cp -r /workspace/runs_restart/s1/results /workspace/runs_newpod/s1/
ls /workspace/runs_newpod/s1/results/imagine/*.json | head -2 | xargs -n1 basename | sed 's/.json$//' > /workspace/runs_newpod/dropped.txt
for j in $(cat /workspace/runs_newpod/dropped.txt); do mv /workspace/runs_newpod/s1/results/imagine/$j.json /workspace/runs_newpod/$j.orig.json; done
(ls /workspace/runs_newpod/s1/results/cf; ls /workspace/runs_newpod/s1/results/imagine) | sed 's/.json$//' > /workspace/runs_newpod/done_ids.txt
rm -rf /workspace/runs_newpod/s1/results  # the new pod starts empty except for the merged done list
$VP /workspace/code/b_runner.py run --manifest $M --stage s1 --root /workspace/runs_newpod --cpu 1 --done /workspace/runs_newpod/done_ids.txt > /workspace/logs/smoke_restart_2.log 2>&1
grep runner /workspace/logs/smoke_restart_2.log
$VP - <<'PY'
import json, os
d = [l.strip() for l in open("/workspace/runs_newpod/dropped.txt") if l.strip()]
for j in d:
    a = json.load(open(f"/workspace/runs_newpod/{j}.orig.json"))
    b = json.load(open(f"/workspace/runs_newpod/s1/results/imagine/{j}.json"))
    cfa = json.load(open(f"/workspace/runs_restart/s1/results/cf/{a['dep']}.json"))
    cfb = json.load(open(f"/workspace/runs_newpod/s1/results/cf/{b['dep']}.json"))
    same_cf = all(cfa[k] == cfb[k] for k in ("r_settled", "success_settled", "success_ever", "T", "r_lift"))
    dv = max(abs(a[k]["v_end"] - b[k]["v_end"]) for k in ("k1", "k2", "k3", "k4") if k in a)
    print(j, "cf identical:", same_cf, "max |dV| imagine:", round(dv, 6), "v_real", a.get("v_real_k1"), b.get("v_real_k1"))
n_new = len(os.listdir("/workspace/runs_newpod/s1/results/imagine"))
print("imagine jobs redone on the new pod:", n_new, "(expected", len(d), ")")
PY
echo SMOKE_RESTART_DONE
