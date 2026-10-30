"""Summarize DP quick evaluations (dp_eval.py JSONL files; standard library only).
Per task x checkpoint: success rate (LIBERO, any step), settled success, near-miss share among failures and among all
episodes (1 <= r_settled < 2), far share (r >= 3), and the spread across checkpoints.
Usage: python3 analyze_dp_eval.py <dir with t*.jsonl> [<dir2> ...]
"""
import glob
import json
import os
import sys
from collections import defaultdict

rows = []
for d in sys.argv[1:]:
    for p in glob.glob(os.path.join(d, "t*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
g = defaultdict(list)
for r in rows:
    g[(r["task"], r["ckpt"])].append(r)
out = {}
for (task, ck), rs in sorted(g.items(), key=lambda x: (x[0][0], int(x[0][1].split("_")[-1].split(".")[0]))):
    n = len(rs)
    fails = [r for r in rs if not r["success"]]
    near = [r for r in fails if 1 <= r["r_settled"] < 2]
    far = [r for r in fails if r["r_settled"] >= 3]
    out.setdefault(str(task), {})[ck] = dict(
        n=n, success=sum(r["success"] for r in rs) / n, success_settled=sum(r["success_settled"] for r in rs) / n,
        near_share_of_failures=(len(near) / len(fails)) if fails else None, near_share_all=len(near) / n,
        far_share_all=len(far) / n, n_fail=len(fails),
        ever_vs_settled_disagree=sum(r["success"] != r["success_settled"] for r in rs),
        r_fail=sorted(round(r["r_settled"], 2) for r in fails))
for task, cks in out.items():
    srs = [v["success"] for v in cks.values()]
    print(f"t{task}: success by stage {[round(x, 2) for x in srs]}  spread {max(srs) - min(srs):.2f}  "
          f"near/all {[round(v['near_share_all'], 2) for v in cks.values()]}")
print(json.dumps(out, indent=1))
