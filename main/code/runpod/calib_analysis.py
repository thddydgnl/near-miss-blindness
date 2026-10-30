"""Severity-grid calibration for M2 E1 (standard library only; runs on pulled E0b counterfactual JSONs).

Calibration demos d00-d09 -> per task and family: median r (settled) per severity and the severity window whose
median r lies in [0.8, 3]. Validation on d10-d19 (same grids): fraction of counterfactuals with r in [0.8, 3], the
r-bin histogram, and the expected number of candidate rollouts per task needed to fill E1's bins.
E1 plan used here: per task 500 counterfactuals with r in [0.8, 3] in 11 equal bins of width 0.2 (~45 each), plus
100 successes with r < 0.8 (nominal and mild severities) for the success class; 6 tasks -> 3,000 + 600.
Usage: python3 calib_analysis.py <dir with cf JSONs> > calib.json
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

rows = []
for p in glob.glob(os.path.join(sys.argv[1], "t*_d*.json")):
    r = json.load(open(p))
    if not r.get("skipped"):
        rows.append(r)
CAL, VAL = set(range(0, 10)), set(range(10, 20))
BINS = [(0.8 + 0.2 * i, 1.0 + 0.2 * i) for i in range(11)]
out = {"n": len(rows), "tasks": {}}
for task in sorted({r["task"] for r in rows}):
    rt = [r for r in rows if r["task"] == task]
    fams = sorted({r["family"] for r in rt if r["family"] != "nominal"})
    tinfo = {"pred": rt[0]["pred"], "families": {}}
    for fam in fams:
        rf = [r for r in rt if r["family"] == fam]
        by_s = defaultdict(list)
        for r in rf:
            if r["demo"] in CAL:
                by_s[r["severity"]].append(r["r_settled"])
        med = {s: statistics.median(v) for s, v in sorted(by_s.items())}
        window = [s for s, m in med.items() if 0.8 <= m <= 3.0]
        val = [r for r in rf if r["demo"] in VAL and (not window or min(window) <= r["severity"] <= max(window))]
        val_all = [r for r in rf if r["demo"] in VAL]
        rv = [r["r_settled"] for r in val]
        tinfo["families"][fam] = dict(
            median_r_by_severity={str(round(s, 4)): round(m, 3) for s, m in med.items()},
            window=[min(window), max(window)] if window else None,
            val_n_in_window=len(val), val_frac_r_0p8_3=(sum(0.8 <= x < 3 for x in rv) / len(rv)) if rv else None,
            val_frac_r_0p8_3_all_severities=(sum(0.8 <= r["r_settled"] < 3 for r in val_all) / len(val_all)) if val_all else None,
            val_bins=[sum(lo <= x < hi for x in rv) for lo, hi in BINS],
            val_success_settled=sum(r["success_settled"] for r in val_all), val_n=len(val_all),
            lift_changes_outcome=sum(r.get("lift_changes_outcome", False) for r in rf),
            ever_vs_settled_disagree=sum(r["success_ever"] != r["success_settled"] for r in rf))
    # expected candidates to fill the E1 bins for this task: pool all families' validation rollouts inside their windows
    pool = []
    for fam, fi in tinfo["families"].items():
        for r in rt:
            if r["family"] == fam and r["demo"] in VAL and fi["window"] and fi["window"][0] <= r["severity"] <= fi["window"][1]:
                pool.append(r["r_settled"])
    pool_cw = [r["r_settled"] for r in rt if r["demo"] in VAL and r["family"] not in ("nominal", "place_offset", "under_actuation", "pull_truncate")]
    hist = [sum(lo <= x < hi for x in pool) for lo, hi in BINS]
    n_pool = len(pool)
    if n_pool and min(hist) > 0:
        # rollouts needed so that the rarest bin reaches 45 (rejection sampling with this pool's bin frequencies)
        need = max(45 * n_pool / h for h in hist)
    else:
        need = None
    tinfo["pool_val_n"] = n_pool
    tinfo["pool_val_bins"] = hist
    tinfo["rollouts_needed_for_500_in_bins"] = round(need) if need else None
    tinfo["curewm_only_val_bins"] = [sum(lo <= x < hi for x in pool_cw) for lo, hi in BINS]
    tinfo["curewm_only_val_n"] = len(pool_cw)
    tinfo["nominal_val_success"] = [r["success_settled"] for r in rt if r["family"] == "nominal" and r["demo"] in VAL].count(True)
    tinfo["disagree_ever_vs_settled_all"] = sum(r["success_ever"] != r["success_settled"] for r in rt)
    tinfo["lift_changes_outcome_all"] = sum(r.get("lift_changes_outcome", False) for r in rt)
    tinfo["n_all"] = len(rt)
    out["tasks"][task] = tinfo
print(json.dumps(out, indent=1))
