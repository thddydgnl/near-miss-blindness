"""H3 for M2 on its pre-registered primary cell (DESIGN_main v2.2 section 7: H3 primary cell = the H1 cell; for M2 the v1.1
cell = judge (a), k = 1, the 6 E2 tasks pooled). Imagination share = (P_imag(near) - P_real(near)) / P_imag(near).

Same data, bands, judge threshold and cluster bootstrap as analyze_e1.py (clusters = start states (task, demo), 2,000
resamples, seed 0, percentile CIs). The primary computation uses records where both the real-frame and the imagined (k = 1)
score exist; the unpaired version (each side on its own valid records, as in the E1 tables) is reported as a check.

usage: python3 h3_m2.py --imagine ../../results/E1_m2/merged/imagine.jsonl --out ../../results/E1_m2/analysis/h3_m2.json
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--imagine", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

E2_TASKS = (1, 2, 4, 5, 6, 8)
recs = [json.loads(l) for l in open(args.imagine) if l.strip()]
recs = [r for r in recs if not r.get("skipped") and "k1" in r and r["task"] in E2_TASKS]
task = np.array([r["task"] for r in recs])
demo = np.array([int(r["dep"].split("_d")[1][:2]) for r in recs])
clu = task * 100 + demo
y = np.array([bool(r["success_settled"]) for r in recs])
rr = np.array([float(r["r_settled"]) for r in recs])
near = ~y & (rr >= 1) & (rr < 2)


def num(v):
    return np.nan if v is None else float(v)


real = np.array([num(r.get("a_logit_real")) for r in recs])
imag = np.array([num(r["k1"].get("a_logit")) for r in recs])
THR = 0.0  # judge (a) logit threshold, as in analyze_e1.py


def share(ix, paired):
    m_near = near[ix]
    rv, iv = real[ix], imag[ix]
    if paired:
        v = m_near & ~np.isnan(rv) & ~np.isnan(iv)
        p_real, p_imag = (rv[v] > THR).mean(), (iv[v] > THR).mean()
    else:
        vr, vi = m_near & ~np.isnan(rv), m_near & ~np.isnan(iv)
        p_real, p_imag = (rv[vr] > THR).mean(), (iv[vi] > THR).mean()
    return p_real, p_imag, (p_imag - p_real) / p_imag if p_imag > 0 else np.nan


idx = np.arange(len(recs))
uniq = np.unique(clu)
members = [idx[clu == c] for c in uniq]
out = dict(cell="a|pooled6", k=1, n_records=len(recs), n_near=int(near.sum()), clusters=len(uniq), boot=args.boot,
           seed=args.seed, definition="share = (P_imag(near) - P_real(near)) / P_imag(near); H3 cell holds if share > 0.5")
for name, paired in (("paired", True), ("unpaired", False)):
    pr, pi, sh = share(idx, paired)
    rng = np.random.default_rng(args.seed)
    reps = []
    for _ in range(args.boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        reps.append(share(np.concatenate([members[i] for i in pick]), paired))
    reps = np.array(reps, dtype=float)
    out[name] = dict(p_real_near=float(pr), p_imag_near=float(pi), share=float(sh),
                     share_ci95=[float(np.nanpercentile(reps[:, 2], 2.5)), float(np.nanpercentile(reps[:, 2], 97.5))],
                     p_real_near_ci95=[float(np.percentile(reps[:, 0], 2.5)), float(np.percentile(reps[:, 0], 97.5))],
                     p_imag_near_ci95=[float(np.percentile(reps[:, 1], 2.5)), float(np.percentile(reps[:, 1], 97.5))],
                     cell_holds=bool(sh > 0.5))
json.dump(out, open(args.out, "w"), indent=2)
print(json.dumps(out, indent=2))
