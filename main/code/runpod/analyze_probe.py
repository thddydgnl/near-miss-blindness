"""Summarize the CureWM-protocol probe of the released checkpoint (probe_cf.py output) and compare with CureWM's
released-checkpoint numbers on LIBERO-Goal held-out (analysis/probe_prefa_ho*.jsonl: optimism 79.13%, n = 484).
Standard library only (runs on the Mac on pulled results in seconds).
Metrics as in CureWM analysis/grid44.py: optimism = % failures with V(o, a-) > 0.5; dSF = mean V(a+) - mean V(a-)
over failures; AUROC = P(V(a-) of a survivor > V(a-) of a failure); false alarm = % survivors with V(a-) <= 0.5.
CI: cluster bootstrap over source demonstrations (task, demo), 5,000 resamples, seed 0.
Usage: python3 analyze_probe.py <probe.jsonl> [<cf_margins.jsonl>] [<curewm_prefa_dir>] > summary.json
"""
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

probe = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
margins = {}
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        if l.strip():
            m = json.loads(l)
            margins[m["key"]] = m
ref_dir = sys.argv[3] if len(sys.argv) > 3 else None


def auroc(fails, survs):
    if not fails or not survs:
        return None
    allv = sorted([(v, 1) for v in survs] + [(v, 0) for v in fails])
    rsum, k = 0.0, 0
    while k < len(allv):
        j = k
        while j + 1 < len(allv) and allv[j + 1][0] == allv[k][0]:
            j += 1
        avg = (k + j) / 2.0 + 1.0
        rsum += sum(avg for t in range(k, j + 1) if allv[t][1] == 1)
        k = j + 1
    n1, n0 = len(survs), len(fails)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def metrics(rs, vkey="value_fail_action"):
    fail = [r for r in rs if r["outcome"] is False and r.get(vkey) is not None]
    surv = [r for r in rs if r["outcome"] is True and r.get(vkey) is not None]
    d = dict(n_fail=len(fail), n_surv=len(surv))
    if fail:
        vf = [r[vkey] for r in fail]
        d["optimism_pct"] = 100.0 * sum(v > 0.5 for v in vf) / len(vf)
        d["mean_v_fail"] = sum(vf) / len(vf)
        if vkey == "value_fail_action":
            d["dSF"] = sum(r["value_nominal_action"] for r in fail) / len(fail) - d["mean_v_fail"]
            vo = [r["visually_optimistic"] for r in fail if "visually_optimistic" in r]
            d["visual_optimism_pct"] = 100.0 * sum(vo) / len(vo) if vo else None
    if surv:
        d["false_alarm_pct"] = 100.0 * sum(r[vkey] <= 0.5 for r in surv) / len(surv)
        d["mean_v_surv"] = sum(r[vkey] for r in surv) / len(surv)
    if fail and surv:
        d["auroc"] = auroc([r[vkey] for r in fail], [r[vkey] for r in surv])
    return d


def demo_of(r):
    m = re.match(r"(d\d+)", r["name"])
    return f"{r['key'].split('/')[0]}/{m.group(1) if m else r['name']}"


def boot_ci(rs, fn, n=5000):
    g = defaultdict(list)
    for r in rs:
        g[demo_of(r)].append(r)
    ks = sorted(g)
    rng = random.Random(0)
    vals = []
    for _ in range(n):
        s = []
        for _ in ks:
            s.extend(g[ks[rng.randrange(len(ks))]])
        v = fn(s)
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


out = {"n_rows": len(probe), "n_demos": len({demo_of(r) for r in probe})}
out["released_ours"] = metrics(probe)
out["released_ours"]["optimism_ci95"] = boot_ci(probe, lambda s: metrics(s).get("optimism_pct"))
out["released_ours"]["auroc_ci95"] = boot_ci(probe, lambda s: metrics(s).get("auroc"))
if any("value_real_future" in r for r in probe):
    out["value_head_on_real_future"] = metrics(probe, "value_real_future")
    out["value_head_on_real_future"]["optimism_ci95"] = boot_ci(probe, lambda s: metrics(s, "value_real_future").get("optimism_pct"))
chk = [r for r in probe if "value_fail_action_act5" in r]
out["act1_vs_act5_max_abs_diff"] = max((abs(r["value_fail_action"] - r["value_fail_action_act5"]) for r in chk), default=None)
out["by_family"] = {f: metrics([r for r in probe if r["family"] == f]) for f in sorted({r["family"] for r in probe})}
out["by_task"] = {t: metrics([r for r in probe if r["key"].split("/")[0] == t]) for t in sorted({r["key"].split("/")[0] for r in probe})}
out["by_severity"] = {str(s): metrics([r for r in probe if r["severity"] == s]) for s in sorted({r["severity"] for r in probe})}

# time-to-go: Cosmos value targets are gamma^(T - t) * success with gamma = 0.99
def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                rk[o[t]] = (i + j) / 2.0
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


disc = [0.99 ** max(0, r["T_nominal"] - (r["t0"] + 16)) for r in probe]
out["spearman_V_fail_action_vs_discount_time_to_go"] = spearman([r["value_fail_action"] for r in probe], disc)
out["spearman_V_fail_action_vs_V_nominal_action"] = spearman([r["value_fail_action"] for r in probe],
                                                             [r["value_nominal_action"] for r in probe])

# exploratory: optimism by final failure distance r (from test_margin.py) for the 1-chunk CureWM-style query
if margins:
    bands = {"near_1_2": (1, 2), "mid_2_3": (2, 3), "far_ge3": (3, 1e9)}
    fails = [r for r in probe if r["outcome"] is False and r["key"] in margins]
    out["exploratory_by_r_band_failures"] = {}
    for b, (lo, hi) in bands.items():
        rs = [r for r in fails if lo <= margins[r["key"]]["r_final"] < hi]
        vf = [r["value_fail_action"] for r in rs]
        vr = [r["value_real_future"] for r in rs if "value_real_future" in r]
        out["exploratory_by_r_band_failures"][b] = dict(
            n=len(rs), optimism_pct=(100.0 * sum(v > 0.5 for v in vf) / len(vf)) if vf else None,
            optimism_real_future_pct=(100.0 * sum(v > 0.5 for v in vr) / len(vr)) if vr else None)

if ref_dir:
    ref = []
    for sh in ["ho_A", "ho_B", "ho2_A", "ho2_B"]:
        p = os.path.join(ref_dir, f"probe_prefa_{sh}.jsonl")
        if os.path.exists(p):
            ref += [json.loads(l) for l in open(p) if l.strip()]
    if ref:
        for r in ref:
            r["key"] = "ref/" + r["name"]
        out["curewm_released_goal_ho"] = metrics(ref)
        out["curewm_released_goal_ho_by_family"] = {f: metrics([r for r in ref if r["family"] == f])
                                                    for f in sorted({r["family"] for r in ref})}
print(json.dumps(out, indent=1))
