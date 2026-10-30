"""B E1 M3 analysis (DESIGN_main v1 + v1.1): per p (3 seeds pooled, primary p = 0.03) and per model:
perception floor (real-frame classifier), P(S^=1 | r) on the 49-point grid, band probabilities (success r < 1,
near [1, 2), mid [2, 3), far r = 3.00 only; R2-9 band [1, 1.2) too), logistic fits on log r (all points -> r50, r10;
failures only -> slope), AUROC success vs near / far on real, imagined and ceiling scores, scene-cluster bootstrap
(B = 1000, scenes resampled jointly for the 3 seeds). H1a/H1b verdict at p = 0.03 pooled; H1b horizon clause N/A.
Runs in the CPU analysis env. usage: python m3_e1_analyze.py [B]"""
import os, sys, json, time
import numpy as np
from scipy.special import expit, logit

RES = os.path.expanduser(os.environ.get('R29_OUT', '~/pmx/B/results/E1/m3'))
PS = ('0.03', '0.1', '0.3')
SEEDS = (0, 1, 2)


def wauc(sp, wp, sn, wn):
    if len(sp) == 0 or len(sn) == 0 or wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    o = np.argsort(sn)
    s, w = sn[o], wn[o]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    lo, hi = np.searchsorted(s, sp, 'left'), np.searchsorted(s, sp, 'right')
    return float(np.sum(wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))) / (wp.sum() * wn.sum()))


def wlogit(x, y, w, ridge=1e-4, iters=80):
    X = np.stack([np.ones_like(x), x], 1)
    b = np.zeros(2)
    for _ in range(iters):
        p = expit(X @ b)
        g = X.T @ (w * (y - p)) - ridge * b
        Hm = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(Hm, g)
        b = b + np.clip(step, -5, 5)
        if np.abs(step).max() < 1e-8:
            break
    return b


def r_at(b, q):
    if not np.isfinite(b).all() or b[1] >= 0:
        return np.nan
    return float(np.exp((float(logit(q)) - b[0]) / b[1]))


def emp_cross(rg, p, q):
    """first grid r where the empirical curve drops below q (linear interpolation); robust to perfect separation."""
    for i in range(1, len(rg)):
        if p[i - 1] >= q > p[i]:
            return float(rg[i - 1] + (p[i - 1] - q) / (p[i - 1] - p[i]) * (rg[i] - rg[i - 1]))
    return float('nan')


def load_model(name):
    z = np.load(f'{RES}/e1_{name}.npz')
    return {k: z[k] for k in z.files}


def records(names):
    """flatten models -> arrays r, scene, model index, scores; only usable (scene, r) points."""
    R_, S_, M_, sr, si, sc = [], [], [], [], [], []
    for mi, n in enumerate(names):
        d = load_model(n)
        g_, s_ = np.where(d['use'])
        R_.append(d['grid_r'][g_]); S_.append(s_); M_.append(np.full(len(g_), mi))
        sr.append(d['real_logit'][g_, s_]); si.append(d['imag_logit'][g_, s_]); sc.append(d['ceil_logit'][g_, s_])
    return dict(r=np.concatenate(R_), scene=np.concatenate(S_), model=np.concatenate(M_), real=np.concatenate(sr),
                imag=np.concatenate(si), ceil=np.concatenate(sc))


def metrics(D, w, full=True):
    r = D['r']
    lr = np.log(r)
    sm, nm, fm, mm = r < 1, (r >= 1) & (r < 2), r >= 3, (r >= 2) & (r < 3)
    r29 = (r >= 1) & (r < 1.2)
    out = {}
    for inp in ('real', 'imag', 'ceil'):
        s = D[inp]
        f = (s > 0).astype(float)
        for nmk, mk in (('success', sm), ('near', nm), ('mid', mm), ('far', fm), ('near_r29', r29)):
            out[f'p_{nmk}_{inp}'] = float((f[mk] * w[mk]).sum() / w[mk].sum()) if w[mk].sum() > 0 else np.nan
        out[f'ratio_near_far_{inp}'] = (out[f'p_near_{inp}'] / out[f'p_far_{inp}'] if out[f'p_far_{inp}'] > 0
                                        else (np.inf if out[f'p_near_{inp}'] > 0 else np.nan))
        out[f'auc_sn_{inp}'] = wauc(s[sm], w[sm], s[nm], w[nm])
        out[f'auc_sf_{inp}'] = wauc(s[sm], w[sm], s[fm], w[fm])
        if inp != 'ceil':
            b = wlogit(lr, f, w)
            out[f'r50_{inp}'], out[f'r10_{inp}'] = r_at(b, 0.5), r_at(b, 0.1)
            bf = wlogit(lr[~sm], f[~sm], w[~sm])
            out[f'slope_fail_{inp}'] = float(bf[1])
    out['delta_auc_sn'] = out['auc_sn_real'] - out['auc_sn_imag']
    out['delta_auc_sf'] = out['auc_sf_real'] - out['auc_sf_imag']
    out['delta_r10'] = out['r10_imag'] - out['r10_real']
    if full:
        rg = np.unique(r)
        cur = {inp: [float(((D[inp][r == x] > 0) * w[r == x]).sum() / w[r == x].sum()) for x in rg] for inp in ('real', 'imag', 'ceil')}
        out['curve'] = dict(r=rg.tolist(), **cur)
        out['r50_imag_empirical'] = emp_cross(rg, np.array(cur['imag']), 0.5)
        out['r10_imag_empirical'] = emp_cross(rg, np.array(cur['imag']), 0.1)
        out['r10_real_empirical'] = emp_cross(rg, np.array(cur['real']), 0.1)
        out['delta_r10_empirical'] = out['r10_imag_empirical'] - out['r10_real_empirical']
        out['n'] = dict(success=int(sm.sum()), near=int(nm.sum()), mid=int(mm.sum()), far=int(fm.sum()), near_r29=int(r29.sum()))
    return out


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t0 = time.time()
    out = dict(B=B, pooled={}, per_model={})
    keys = ['p_near_imag', 'p_near_real', 'p_far_imag', 'p_success_imag', 'p_near_r29_imag', 'auc_sn_real', 'auc_sn_imag',
            'auc_sn_ceil', 'auc_sf_real', 'auc_sf_imag', 'delta_auc_sn', 'delta_auc_sf', 'r50_imag', 'r10_imag', 'r10_real',
            'delta_r10', 'slope_fail_imag', 'ratio_near_far_imag']
    for p in PS:
        names = [f'p{p}_std_1x_s{s}' for s in SEEDS if os.path.exists(f'{RES}/e1_p{p}_std_1x_s{s}.npz')]
        if not names:
            continue
        D = records(names)
        nsc = int(D['scene'].max()) + 1
        pt = metrics(D, np.ones(len(D['r'])))
        rng = np.random.RandomState(31337)
        bs = []
        for _ in range(B):
            mult = np.bincount(rng.randint(0, nsc, nsc), minlength=nsc).astype(float)
            bs.append(metrics(D, mult[D['scene']], full=False))
        pt['ci'] = {k: ci([b[k] for b in bs]) for k in keys}
        sf = np.array([b['slope_fail_imag'] for b in bs], float)
        pt['p_boot_slope_fail_imag_ge0'] = float(np.mean(sf[np.isfinite(sf)] >= 0)) if np.isfinite(sf).any() else None
        pt['models'] = names
        out['pooled'][p] = pt
        print(f"p={p} pooled ({len(names)} seeds): near imag {pt['p_near_imag']:.3f} real {pt['p_near_real']:.3f} far imag "
              f"{pt['p_far_imag']:.3f} | AUROC sn real {pt['auc_sn_real']:.3f} imag {pt['auc_sn_imag']:.3f} ceil {pt['auc_sn_ceil']:.3f}"
              f" | sf real {pt['auc_sf_real']:.3f} imag {pt['auc_sf_imag']:.3f} | r50 {pt['r50_imag']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        for n in names:
            Dn = records([n])
            out['per_model'][n] = metrics(Dn, np.ones(len(Dn['r'])))
    P = out['pooled'].get('0.03')
    if P:
        h1a = dict(p_near=P['p_near_imag'], p_far=P['p_far_imag'], ratio=P['ratio_near_far_imag'], slope_fail=P['slope_fail_imag'],
                   slope_fail_ci=P['ci']['slope_fail_imag'], near_ge_0p3=bool(P['p_near_imag'] >= 0.3),
                   ratio_ge_3=bool((P['p_far_imag'] == 0 and P['p_near_imag'] > 0) or (P['p_far_imag'] > 0 and P['p_near_imag'] / P['p_far_imag'] >= 3)),
                   monotone=bool(P['ci']['slope_fail_imag'] is not None and P['ci']['slope_fail_imag'][1] < 0),
                   perception_floor_p_near_real=P['p_near_real'], far_band='r = 3.00 only (w <= 0.13 scenes)',
                   p_boot_monotone=P.get('p_boot_slope_fail_imag_ge0'))
        h1a['H1a_M3'] = bool(h1a['near_ge_0p3'] and h1a['ratio_ge_3'] and h1a['monotone'])
        h1b = dict(delta_auc_sn=P['delta_auc_sn'], delta_auc_sn_ci=P['ci']['delta_auc_sn'], delta_auc_sf=P['delta_auc_sf'],
                   delta_auc_sf_ci=P['ci']['delta_auc_sf'], sn_ge_0p05=bool(P['delta_auc_sn'] >= 0.05),
                   sf_le_0p02=bool(P['delta_auc_sf'] <= 0.02), horizon_clause='N/A (fixed 4-step imagination)')
        h1b['H1b_M3_without_horizon_clause'] = bool(h1b['sn_ge_0p05'] and h1b['sf_le_0p02'])
        out['verdict_M3'] = dict(cell='p=0.03, seeds 0-2 pooled, frozen R2-9 classifier', H1a=h1a, H1b=h1b)
        print('VERDICT M3', json.dumps(out['verdict_M3']))
    out['time_s'] = time.time() - t0
    json.dump(out, open(f'{RES}/e1_m3_analysis.json', 'w'), indent=1)
    print('M3_ANALYZE_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
