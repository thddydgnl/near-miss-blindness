"""Check that the runner's judge (a) CNN (b_judges.judge_a_cnn) reproduces judge_a.py's held logits."""
import glob, json, os
import numpy as np
import b_judges as J
TRAIN, HELD = set(range(0, 17)), set(range(17, 20))
rows = []
for jp in sorted(glob.glob("/workspace/e0b/cf/t*_d*.json")):
    r = json.load(open(jp))
    if r.get("skipped") or r["task"] not in J.TASKS_A or r["demo"] not in (TRAIN | HELD):
        continue
    if r["demo"] in HELD:
        rows.append(r)
ref = np.load("/workspace/results/judge_a/cnn_held_logits_d0-16.npy")
assert len(ref) == len(rows), (len(ref), len(rows))
diffs = []
for i in range(0, len(rows), max(1, len(rows) // 40)):
    z = np.load(f"/workspace/e0b/cf/{rows[i]['cf_id']}.npz")
    a = J.judge_a_cnn(rows[i]["task"], z["Slift_primary"], z["Slift_wrist"])
    diffs.append(abs(a - float(ref[i])))
    if (a > 0) != (ref[i] > 0):
        print("sign mismatch", rows[i]["cf_id"], a, ref[i])
print("checked", len(diffs), "max |logit diff|", max(diffs), "mean", float(np.mean(diffs)))
