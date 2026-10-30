"""Compare two sets of counterfactual records (standard library only): regenerated vs reference, matched by id.
Inputs are directories of <cf_id>.json files or JSONL files (records with cf_id or job_id).
Usage: python3 compare_cf_records.py <new dir|jsonl> <reference dir|jsonl> [--strip_prefix cf_]"""
import glob
import json
import os
import sys

FIELDS_NUM = ("r_settled", "r_lift")
FIELDS_EQ = ("success_settled", "success_lift", "success_ever", "T", "skipped", "aux_settled")


def load(p):
    recs = {}
    files = sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else None
    rows = [json.load(open(f)) for f in files] if files is not None else [json.loads(l) for l in open(p) if l.strip()]
    for r in rows:
        cid = r.get("cf_id") or r["job_id"]
        recs[cid[3:] if cid.startswith("cf_") else cid] = r
    return recs


new, ref = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(new) & set(ref))
max_d = {f: 0.0 for f in FIELDS_NUM}
mism = []
for c in common:
    a, b = new[c], ref[c]
    for f in FIELDS_NUM:
        if a.get(f) is not None and b.get(f) is not None:
            max_d[f] = max(max_d[f], abs(a[f] - b[f]))
        elif (a.get(f) is None) != (b.get(f) is None):
            mism.append((c, f, a.get(f), b.get(f)))
    for f in FIELDS_EQ:
        if a.get(f) != b.get(f):
            mism.append((c, f, a.get(f), b.get(f)))
print(json.dumps(dict(new=len(new), reference=len(ref), common=len(common), max_abs_diff=max_d, n_mismatch=len(mism),
                      mismatch_examples=mism[:10])))
