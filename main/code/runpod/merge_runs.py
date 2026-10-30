"""Merge runner results pulled from one or more pods (standard library only; runs on the Mac).

  python3 merge_runs.py --manifest m.jsonl --pulled <dir1> [<dir2> ...] --out <merged dir>
Each <dir> is a pulled copy of /workspace/runs/<stage>/results (and optionally small/). Writes
<out>/<kind>.jsonl (one record per job_id; if a job was finished on two pods, the records must agree on the key
outcome fields, otherwise both are kept in conflicts.jsonl), <out>/done_ids.txt (push to the next pod as --done),
and prints completeness per kind against the manifest.
"""
import argparse
import glob
import json
import os

KEY_FIELDS = ("r_settled", "success_settled", "success", "T", "n_chunks")

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--pulled", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
jobs = [json.loads(l) for l in open(args.manifest) if l.strip()]
kind_of = {j["job_id"]: j["kind"] for j in jobs}
merged, conflicts = {}, []
for d in args.pulled:
    for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        jid = os.path.basename(p)[:-5]
        if jid not in kind_of:
            continue
        rec = json.load(open(p))
        if jid in merged:
            a, b = merged[jid], rec
            if any(a.get(k) != b.get(k) for k in KEY_FIELDS if k in a or k in b):
                conflicts.append({"job_id": jid, "a": {k: a.get(k) for k in KEY_FIELDS}, "b": {k: b.get(k) for k in KEY_FIELDS}})
            continue
        merged[jid] = rec
by_kind = {}
for jid, rec in merged.items():
    by_kind.setdefault(kind_of[jid], []).append(rec)
for k, recs in by_kind.items():
    with open(os.path.join(args.out, f"{k}.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r.get("job_id", "")):
            f.write(json.dumps(r) + "\n")
with open(os.path.join(args.out, "done_ids.txt"), "w") as f:
    f.write("\n".join(sorted(merged)) + "\n")
with open(os.path.join(args.out, "conflicts.jsonl"), "w") as f:
    for c in conflicts:
        f.write(json.dumps(c) + "\n")
total = {}
for j in jobs:
    t = total.setdefault(j["kind"], [0, 0])
    t[0] += 1
    t[1] += j["job_id"] in merged
print(json.dumps({k: {"jobs": v[0], "done": v[1], "missing": v[0] - v[1]} for k, v in total.items()}), "conflicts", len(conflicts))
