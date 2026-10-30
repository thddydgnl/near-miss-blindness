"""B E1 (M2) analysis on the merged imagine records: failure-distance curves, band probabilities, boundary resolution
(AUROC real vs imagined), and the H1a / H1b verdicts of the pre-registered primary cell, plus every other cell.
Plan fixed before any E1 imagination: worklogs/E1_m2.md section 0.4 (DESIGN_main v1.1: M2 primary cell = primary
judge, k = 1, the 6 E2 tasks pooled; t0 reported separately; slopes over k in {1, 2, 3, 4}; cluster = start state).

Definitions
  success     success_settled (LIBERO predicate at the settled end state; r < 1 and aux)
  bands       failures: near 1 <= r < 2, mid 2 <= r < 3, far r >= 3 (r = r_settled); failures with r < 1 (aux
              failures) are in no band
  judges      a: CNN logit > 0 (real = real post-lift frames, imagined = imagined post-lift frames)
              c: V > 0.5 (imagined = value at the end of the k-th chunk, real = value of the real end state)
              d: VLM P(Yes) > 0.2018 (threshold fixed in E0b on the training split; (d) is auxiliary)
              c_lift: value after the imagined lift chunk (auxiliary)
  H1a         (i) logistic slope of S_hat on log r over failures with r >= 1 (pooled, as worded), cluster-bootstrap
              95% CI < 0; (ii) P(S_hat = 1 | near) >= 0.3; (iii) near / far >= 3
  H1b         (i) AUROC_real - AUROC_imag(k=1) (success vs near) >= 0.05; (ii) the same for success vs far <= 0.02;
              (iii) OLS slope over k of [AUROC_imag(k) - AUROC_real] (success vs near), cluster-bootstrap 95% CI < 0
              (AUROC_real is computed on the same items as AUROC_imag(k): paired)
  curves      logistic of S_hat on log max(r, 0.01) over all items, real and imagined: r50, r10, dr10 = r10(imag) -
              r10(real); binned P(S_hat = 1) over r < 0.8, 11 bins of 0.2 in [0.8, 3), r >= 3
  bootstrap   clusters = start states (task, demo), 2,000 resamples, seed 0, percentile CIs (also 98.33% =
              Holm-conservative for a family of 3)
  sensitivity task-intercept slope, start-state random-intercept GLMM slope (Gauss-Hermite), per task
  D3          information ceiling: per-task L2 logistic regression (success vs failure) on standardized DINOv2
              features of the judged frame (real or imagined at k), grouped 5-fold CV by start state (C in {0.01, 0.1}
              chosen by inner grouped 3-fold), out-of-fold logits pooled over tasks: AUROC success vs near / far
Runs on a pod (numpy + scipy; scikit-learn for D3). Usage:
  $VP analyze_e1.py --imagine merged/imagine.jsonl --primary a --out out_dir [--small small_dir] [--boot 2000]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.special import expit
from scipy.special import logit as sp_logit
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--imagine", required=True)
ap.add_argument("--primary", choices=["a", "c"], required=True, help="M2 primary judge (decided on (a) held accuracy)")
ap.add_argument("--out", required=True)
ap.add_argument("--small", default=None, help="dir with small/<job_id>.npz feature files (D3)")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_d3", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

E2_TASKS = (1, 2, 4, 5, 6, 8)
KS = (1, 2, 3, 4)
D_THR = 0.2018132209777832  # E0b judge_d_v2/judge_d_7b.json compare2 threshold (training split, fixed before E1)
JUDGES = {
    "a": ("a_logit_real", "a_logit", 0.0),
    "c": ("v_real_k1", "v_end", 0.5),
    "d": ("d_real", "d", D_THR),
    "c_lift": ("v_real_k1", "v_lift", 0.5),
}
R_FLOOR = 0.01
BINS = [(0.0, 0.8)] + [(0.8 + 0.2 * i, 1.0 + 0.2 * i) for i in range(11)] + [(3.0, np.inf)]
BAND_CODES = (("success", 0), ("near", 1), ("mid", 2), ("far", 3))

# ------------------------------------------------------------------ data
recs = [json.loads(l) for l in open(args.imagine) if l.strip()]
n_all = len(recs)
recs = [r for r in recs if not r.get("skipped") and "k1" in r]
N = len(recs)
task = np.array([r["task"] for r in recs])
demo = np.array([int(r["dep"].split("_d")[1][:2]) for r in recs])
clu = task * 100 + demo
y = np.array([bool(r["success_settled"]) for r in recs])
rr = np.array([float(r["r_settled"]) for r in recs])
band = np.full(N, -1)
band[y] = 0
band[~y & (rr >= 1) & (rr < 2)] = 1
band[~y & (rr >= 2) & (rr < 3)] = 2
band[~y & (rr >= 3)] = 3
lr_all = np.log(np.maximum(rr, R_FLOOR))


def num(v):
    return np.nan if v is None else float(v)


S = {}
for j, (rk, ik, thr) in JUDGES.items():
    S[j] = dict(real=np.array([num(r.get(rk)) for r in recs]), thr=thr,
                imag={k: np.array([num(r[f"k{k}"].get(ik)) if r.get(f"k{k}") else np.nan for r in recs]) for k in KS})


# ------------------------------------------------------------------ estimators
def auroc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    rk = rankdata(np.concatenate([pos, neg]))
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logfit(X, z, lam=1e-3):
    """Ridge-stabilized (lam on all coefficients) logistic regression by damped Newton; X includes the intercept."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = expit(X @ b)
        grad = X.T @ (z - p) - lam * b
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        step = np.clip(step, -5, 5)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def slope_fit(lr, z):
    if len(z) < 3 or z.min() == z.max():
        # all-equal outcomes: the (ridge) slope is 0 by symmetry; report it as such
        return 0.0 if len(z) >= 3 else np.nan
    return float(logfit(np.column_stack([np.ones_like(lr), lr]), z.astype(float))[1])


def curve(lr, z):
    if len(z) < 3:
        return dict(alpha=np.nan, beta=np.nan, r50=np.nan, r10=np.nan)
    a, b = logfit(np.column_stack([np.ones_like(lr), lr]), z.astype(float))
    res = dict(alpha=float(a), beta=float(b))
    for q, name in ((0.5, "r50"), (0.1, "r10")):
        res[name] = float(np.exp((sp_logit(q) - a) / b)) if b < 0 else np.nan
    return res


def side_stats(score, thr, bd, rv, lr):
    v = ~np.isnan(score)
    h = score > thr
    out = {}
    for name, code in BAND_CODES:
        m = v & (bd == code)
        out[name] = float(h[m].mean()) if m.any() else np.nan
        out[f"mean_score_{name}"] = float(score[m].mean()) if m.any() else np.nan
        out[f"n_{name}"] = int(m.sum())
    m = v & (bd >= 1)
    out["fail_r>=1"] = float(h[m].mean()) if m.any() else np.nan
    out["near_over_far"] = (out["near"] / out["far"] if out["far"] > 0 else (np.inf if out["near"] > 0 else np.nan))
    out["slope_fail"] = slope_fit(lr[m], h[m]) if m.sum() >= 3 else np.nan
    out["curve"] = curve(lr[v], h[v]) if v.sum() >= 3 else dict(alpha=np.nan, beta=np.nan, r50=np.nan, r10=np.nan)
    out["bins"] = [float(h[v & (rv >= lo) & (rv < hi)].mean()) if (v & (rv >= lo) & (rv < hi)).any() else np.nan
                   for lo, hi in BINS]
    return out


def stats_for(idx, j):
    Sj, thr = S[j], S[j]["thr"]
    bd, rv, lr = band[idx], rr[idx], lr_all[idx]
    real = Sj["real"][idx]
    out = {"real": side_stats(real, thr, bd, rv, lr)}
    out["real"]["auroc_near"] = auroc(real[bd == 0], real[bd == 1])
    out["real"]["auroc_far"] = auroc(real[bd == 0], real[bd == 3])
    for k in KS:
        im = Sj["imag"][k][idx]
        v = ~np.isnan(im)
        if not v.any():
            continue
        d = side_stats(im, thr, bd, rv, lr)
        vv = v & ~np.isnan(real)
        d["auroc_near"] = auroc(im[vv & (bd == 0)], im[vv & (bd == 1)])
        d["auroc_far"] = auroc(im[vv & (bd == 0)], im[vv & (bd == 3)])
        d["auroc_real_near_paired"] = auroc(real[vv & (bd == 0)], real[vv & (bd == 1)])
        d["auroc_real_far_paired"] = auroc(real[vv & (bd == 0)], real[vv & (bd == 3)])
        d["d_auroc_near"] = d["auroc_real_near_paired"] - d["auroc_near"]
        d["d_auroc_far"] = d["auroc_real_far_paired"] - d["auroc_far"]
        d["d_r10"] = d["curve"]["r10"] - out["real"]["curve"]["r10"]
        d["d_r50"] = d["curve"]["r50"] - out["real"]["curve"]["r50"]
        out[f"k{k}"] = d
    for which in ("near", "far"):
        ks = [k for k in KS if f"k{k}" in out and np.isfinite(out[f"k{k}"][f"d_auroc_{which}"])]
        out[f"horizon_slope_{which}"] = (float(np.polyfit(ks, [-out[f"k{k}"][f"d_auroc_{which}"] for k in ks], 1)[0])
                                         if len(ks) >= 2 else np.nan)
    return out


def flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                out[f"{key}[{i}]"] = x
        else:
            out[key] = v
    return out


def unflatten_ci(point_flat, reps, qs):
    ci = {}
    for key in point_flat:
        vals = np.array([r.get(key, np.nan) for r in reps], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0 or key.split(".")[-1].startswith("n_"):
            continue
        ci[key] = [float(np.percentile(vals, q, method="nearest")) for q in qs]  # nearest: ratios can be inf
    return ci


def bootstrap(idx, fn, B, seed):
    rng = np.random.default_rng(seed)
    cl = clu[idx]
    uniq = np.unique(cl)
    members = [idx[cl == c] for c in uniq]
    reps = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        reps.append(flatten(fn(np.concatenate([members[i] for i in pick]))))
    return reps, len(uniq)


def task_intercept_slope(idx, j, k):
    Sj = S[j]
    im = Sj["imag"][k][idx]
    m = ~np.isnan(im) & (band[idx] >= 1)
    ii = idx[m]
    T = sorted(set(task[ii]))
    X = np.column_stack([(task[ii] == t).astype(float) for t in T] + [lr_all[ii]])
    return float(logfit(X, (im[m] > Sj["thr"]).astype(float))[-1])


def glmm_slope(idx, j, k, n_gh=20):
    """Random-intercept (start state) logistic of S_hat on log r over failures with r >= 1; Gauss-Hermite marginal
    likelihood, Wald CI from a central-difference Hessian."""
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.optimize import minimize
    Sj = S[j]
    im = Sj["imag"][k][idx]
    m = ~np.isnan(im) & (band[idx] >= 1)
    ii = idx[m]
    x, z = lr_all[ii], (im[m] > Sj["thr"]).astype(float)
    uniq, inv = np.unique(clu[ii], return_inverse=True)
    nodes, w = hermegauss(n_gh)
    w = w / w.sum()

    def nll(th):
        a, b, ls = th
        E = (a + b * x)[:, None] + np.exp(ls) * nodes[None, :]
        ll = z[:, None] * E - np.logaddexp(0, E)
        L = np.zeros((len(uniq), n_gh))
        np.add.at(L, inv, ll)
        mx = L.max(1, keepdims=True)
        return -float((mx[:, 0] + np.log((np.exp(L - mx) * w).sum(1))).sum())

    r = minimize(nll, x0=np.array([0.0, 0.0, np.log(0.5)]), method="Nelder-Mead",
                 options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    r = minimize(nll, x0=r.x, method="BFGS")
    th, h = r.x, 1e-4
    H = np.zeros((3, 3))
    for a_ in range(3):
        for b_ in range(3):
            e1, e2 = np.eye(3)[a_] * h, np.eye(3)[b_] * h
            H[a_, b_] = (nll(th + e1 + e2) - nll(th + e1 - e2) - nll(th - e1 + e2) + nll(th - e1 - e2)) / (4 * h * h)
    try:
        se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    except np.linalg.LinAlgError:
        se = np.nan
    return dict(alpha=float(th[0]), beta=float(th[1]), sigma=float(np.exp(th[2])), beta_se=se,
                beta_ci95=[float(th[1] - 1.96 * se), float(th[1] + 1.96 * se)], n=int(m.sum()), clusters=len(uniq),
                converged=bool(r.success))


def verdict(point, ci):
    k1 = point.get("k1", {})
    g = lambda key: ci.get(key, [np.nan] * 4)
    h1a = dict(slope=k1.get("slope_fail"), slope_ci95=g("k1.slope_fail")[1:3], slope_ci_holm=[g("k1.slope_fail")[0], g("k1.slope_fail")[3]],
               near=k1.get("near"), near_ci95=g("k1.near")[1:3], far=k1.get("far"), far_ci95=g("k1.far")[1:3],
               near_over_far=k1.get("near_over_far"), near_over_far_ci95=g("k1.near_over_far")[1:3])
    h1a["i_monotone"] = bool(np.isfinite(h1a["slope_ci95"][1]) and h1a["slope_ci95"][1] < 0)
    h1a["ii_near>=0.3"] = bool(k1.get("near", np.nan) >= 0.3)
    h1a["iii_near/far>=3"] = bool(k1.get("near_over_far", np.nan) >= 3)
    h1a["holds"] = h1a["i_monotone"] and h1a["ii_near>=0.3"] and h1a["iii_near/far>=3"]
    h1b = dict(d_auroc_near=k1.get("d_auroc_near"), d_auroc_near_ci95=g("k1.d_auroc_near")[1:3],
               d_auroc_far=k1.get("d_auroc_far"), d_auroc_far_ci95=g("k1.d_auroc_far")[1:3],
               horizon_slope=point.get("horizon_slope_near"), horizon_slope_ci95=g("horizon_slope_near")[1:3],
               horizon_slope_ci_holm=[g("horizon_slope_near")[0], g("horizon_slope_near")[3]])
    h1b["i_d_near>=0.05"] = bool(k1.get("d_auroc_near", np.nan) >= 0.05)
    h1b["ii_d_far<=0.02"] = bool(k1.get("d_auroc_far", np.nan) <= 0.02)
    h1b["iii_slope_ci<0"] = bool(np.isfinite(h1b["horizon_slope_ci95"][1]) and h1b["horizon_slope_ci95"][1] < 0)
    h1b["holds"] = h1b["i_d_near>=0.05"] and h1b["ii_d_far<=0.02"] and h1b["iii_slope_ci<0"]
    return dict(H1a=h1a, H1b=h1b, go=bool(h1a["holds"] or h1b["holds"]))


# ------------------------------------------------------------------ cells
t_start = time.time()
groups = {"pooled6": np.where(np.isin(task, E2_TASKS))[0]}
for t in sorted(set(task)):
    groups[f"t{t}"] = np.where(task == t)[0]
QS = (100 / 120, 2.5, 97.5, 100 - 100 / 120)  # [Holm-lo, 95-lo, 95-hi, Holm-hi]
results = dict(n_records=n_all, n_used=N, primary_judge=args.primary, boot=args.boot, bins=[list(b) for b in BINS],
               composition={g: {name: int((band[ix] == code).sum()) for name, code in BAND_CODES} |
                            {"fail_r<1": int((band[ix] == -1).sum()), "n": int(len(ix)), "start_states": int(len(np.unique(clu[ix])))}
                            for g, ix in groups.items()},
               cells={})
for j in JUDGES:
    for gname, idx in groups.items():
        if np.isnan(S[j]["real"][idx]).all() and all(np.isnan(S[j]["imag"][k][idx]).all() for k in KS):
            continue  # e.g. judge (a) on t0 (no head)
        point = stats_for(idx, j)
        B = args.boot if gname == "pooled6" else max(args.boot // 2, 200)
        reps, ncl = bootstrap(idx, lambda ix: stats_for(ix, j), B, args.seed)
        ci = unflatten_ci(flatten(point), reps, QS)
        results["cells"][f"{j}|{gname}"] = dict(point=point, ci=ci, clusters=ncl, boot=B, verdict=verdict(point, ci))
        print(f"[{time.time() - t_start:.0f}s] {j}|{gname} done", flush=True)

# sensitivity for the primary judge (pooled6, every k)
sens = {}
idx = groups["pooled6"]
for j in ("a", "c", "d"):
    if np.isnan(S[j]["imag"][1][idx]).all():
        continue
    for k in KS:
        reps = []
        rng = np.random.default_rng(args.seed)
        cl = clu[idx]
        uniq = np.unique(cl)
        members = [idx[cl == c] for c in uniq]
        for _ in range(max(args.boot // 2, 200)):
            pick = rng.integers(0, len(uniq), len(uniq))
            reps.append(task_intercept_slope(np.concatenate([members[i] for i in pick]), j, k))
        sens[f"{j}|k{k}"] = dict(task_intercept_slope=task_intercept_slope(idx, j, k),
                                 task_intercept_slope_ci95=[float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
                                 glmm=glmm_slope(idx, j, k))
        print(f"[{time.time() - t_start:.0f}s] sensitivity {j}|k{k}", flush=True)
results["sensitivity_pooled6"] = sens

# ------------------------------------------------------------------ D3 information ceiling
if args.small and not args.no_d3:
    from sklearn.linear_model import LogisticRegression
    feats = {}
    keys = ["feat_real"] + [f"feat_imag_k{k}" for k in KS]
    for i, r in enumerate(recs):
        p = os.path.join(args.small, r["job_id"] + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            for key in keys:
                if key in z:
                    feats.setdefault(key, {})[i] = z[key].astype(np.float32)

    def grouped_folds(groups_, n, seed):
        u = np.unique(groups_)
        rng = np.random.default_rng(seed)
        u = u[rng.permutation(len(u))]
        return [np.isin(groups_, u[f::n]) for f in range(n)]

    def fit_predict(Xtr, ytr, Xte, C):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        m = LogisticRegression(C=C, max_iter=3000)
        m.fit((Xtr - mu) / sd, ytr)
        return m.decision_function((Xte - mu) / sd)

    d3 = {}
    for key in keys:
        if key not in feats:
            continue
        oof = np.full(N, np.nan)
        for t in sorted(set(task)):
            ii = np.array([i for i in np.where(task == t)[0] if i in feats[key]])
            if len(ii) < 20 or len(set(y[ii])) < 2:
                continue
            X = np.stack([feats[key][i] for i in ii])
            yy = y[ii].astype(int)
            for fm in grouped_folds(clu[ii], 5, args.seed):
                tr, te = ~fm, fm
                if len(set(yy[tr])) < 2 or te.sum() == 0:
                    continue
                best = None
                for C in (0.01, 0.1):  # inner grouped 3-fold choice of the L2 strength (AUROC success vs fail)
                    sc = []
                    for im_ in grouped_folds(clu[ii][tr], 3, args.seed + 1):
                        a_, b_ = ~im_, im_
                        if len(set(yy[tr][a_])) < 2 or len(set(yy[tr][b_])) < 2:
                            continue
                        s = fit_predict(X[tr][a_], yy[tr][a_], X[tr][b_], C)
                        sc.append(auroc(s[yy[tr][b_] == 1], s[yy[tr][b_] == 0]))
                    m_ = np.nanmean(sc) if sc else -1
                    if best is None or m_ > best[1]:
                        best = (C, m_)
                oof[ii[te]] = fit_predict(X[tr], yy[tr], X[te], best[0])
        res = {}
        for gname, gidx in groups.items():
            o = oof[gidx]
            bd = band[gidx]
            res[gname] = dict(auroc_near=auroc(o[bd == 0], o[bd == 1]), auroc_far=auroc(o[bd == 0], o[bd == 3]),
                              auroc_fail=auroc(o[bd == 0], o[bd != 0]), n=int((~np.isnan(o)).sum()))
        d3[key] = res
        print(f"[{time.time() - t_start:.0f}s] D3 {key}", flush=True)
    results["D3"] = d3

results["primary_cell"] = dict(cell=f"{args.primary}|pooled6", k=1,
                               verdict=results["cells"][f"{args.primary}|pooled6"]["verdict"])
results["t0_cells"] = {c: v["verdict"] for c, v in results["cells"].items() if c.endswith("|t0")}
results["seconds"] = time.time() - t_start


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "inf" if o > 0 else "-inf"
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(results), open(os.path.join(args.out, "e1_analysis.json"), "w"), indent=1)

# ------------------------------------------------------------------ figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = [0.4] + [0.9 + 0.2 * i for i in range(11)] + [3.5]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, j in zip(axes[0], ("a", "c", "d", "c_lift")):
        c = results["cells"].get(f"{j}|pooled6")
        if c is None:
            ax.set_visible(False)
            continue
        ax.plot(centers, c["point"]["real"]["bins"], "k-o", ms=3, label="real")
        for k, col in zip(KS, ("C0", "C1", "C2", "C3")):
            if f"k{k}" in c["point"]:
                ax.plot(centers, c["point"][f"k{k}"]["bins"], "-o", color=col, ms=3, label=f"imag k={k}")
        ax.axvline(1, color="gray", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"judge {j}, 6 tasks: P(S_hat=1 | r bin)")
        ax.set_xlabel("r (bin centre)")
        ax.legend(fontsize=7)
    for ax, j in zip(axes[1], ("a", "c", "d")):
        c = results["cells"].get(f"{j}|pooled6")
        if c is None:
            ax.set_visible(False)
            continue
        for which, col in (("near", "C3"), ("far", "C0")):
            ks = [k for k in KS if f"k{k}" in c["point"]]
            ax.plot(ks, [c["point"][f"k{k}"][f"auroc_{which}"] for k in ks], "-o", color=col, label=f"imag success vs {which}")
            ax.axhline(c["point"]["real"][f"auroc_{which}"], color=col, ls="--", lw=1, label=f"real success vs {which}")
        ax.set_xticks(list(KS))
        ax.set_xlabel("k (imagined chunks)")
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"judge {j}, 6 tasks: AUROC")
        ax.legend(fontsize=7)
    ax = axes[1][3]
    for j, col in (("c", "C1"), ("d", "C2")):
        c = results["cells"].get(f"{j}|t0")
        if c is None:
            continue
        ax.plot(centers, c["point"]["real"]["bins"], "--o", color=col, ms=3, label=f"t0 {j} real")
        ax.plot(centers, c["point"]["k1"]["bins"], "-o", color=col, ms=3, label=f"t0 {j} imag k=1")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.axvline(1, color="gray", lw=0.8, ls="--")
    ax.set_title("t0 (drawer): P(S_hat=1 | r bin)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "e1_curves.png"), dpi=110)
except Exception as e:  # figures are optional
    print("figure error", e)
print(json.dumps(clean(results["primary_cell"]), indent=1))
print(f"ANALYSIS_DONE {time.time() - t_start:.0f}s")
