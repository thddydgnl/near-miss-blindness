"""B E1 M1 analysis (DESIGN_main v1 + v1.1 operational cells). Runs in the CPU analysis env (b-ana).
Per cell (set, H, definition d; the grid targeted at d) and judge (a = K=5 ensemble of a_full / a_obj, b = d_vis with
the frozen tau of d; information ceiling ai = imagined-feature ensemble):
  perception floor  P(S~=1 | near) on real frames (reported first)
  band probabilities P(S^=1 | success / near / mid / far / expert), real and imagined, near/far ratio
  logistic fits on log r (Newton): all counterfactuals -> r50, r10 (imagined, real), delta r10 = r10_imag - r10_real;
      failures only (r >= 1) -> slope (H1a monotonicity, v1.1)
  logistic GLMM with a start-state random intercept (statsmodels BinomialBayesMixedGLM, variational Bayes): point fits
  AUROC success vs near / success vs far: real, imagined, ceiling; delta = real - imagined
  cluster bootstrap over source trajectories (B = 1000): CIs of everything above; horizon slope of
      [AUROC_imag - AUROC_real](success vs near) over H in {3,5,7,10} (same resampled trajectories for all H)
Verdict cells (v1.1): M1 primary = val, H = 5, d = r, judge a.
usage: python e1_m1_analyze.py [B]"""
import os, sys, json, glob, time
import numpy as np
from scipy.special import expit, logit

RES = os.path.expanduser(os.environ.get('B_RES', '~/pmx/B/results/E1/m1'))
TASK = os.environ.get('B_TASK', 'pusht')                 # 'pusht' (E1 M1) or 'pointmaze' (E7, DESIGN_main section 12)
HS = (3, 5, 7, 10)
DEFS = ('r', 'r_obj') if TASK == 'pusht' else ('r',)
SETS = ('val', 'trainsplit')
OUT = 'e1_m1_analysis.json' if TASK == 'pusht' else 'e7_analysis.json'
if TASK == 'pusht':
    TAU = {'r': 0.2125, 'r_obj': 0.345}                  # frozen (DESIGN_main 0.2)
else:
    TAU = {'r': json.load(open(f'{RES}/judges_e7.json'))['b_full']['tau_used']}   # fixed before evaluation (12.7)
AKEY = {'r': 'p_full', 'r_obj': 'p_obj'}
AIKEY = {'r': 'pi_full_imag', 'r_obj': 'pi_obj_imag'}
BINS = ('success', 'near', 'mid', 'far')
CURVE_EDGES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 1e9]
L10 = float(logit(0.1))


def load(st, H, d):
    p = f'{RES}/eval_{st}_H{H}_{d}.jsonl'
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def binof(x):
    return np.where(x < 1, 0, np.where(x < 2, 1, np.where(x < 3, 2, 3)))


def wauc(sp, wp, sn, wn):
    if len(sp) == 0 or len(sn) == 0 or wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    o = np.argsort(sn)
    s, w = sn[o], wn[o]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    lo, hi = np.searchsorted(s, sp, 'left'), np.searchsorted(s, sp, 'right')
    return float(np.sum(wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))) / (wp.sum() * wn.sum()))


def wlogit(x, y, w, ridge=1e-4, iters=60):
    X = np.stack([np.ones_like(x), x], 1)
    b = np.zeros(2)
    for _ in range(iters):
        p = expit(X @ b)
        g = X.T @ (w * (y - p)) - ridge * b
        Hm = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(Hm, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def r_at(b, q):
    """r where the fitted curve equals probability q (decreasing curve only)."""
    if not np.isfinite(b).all() or b[1] >= 0:
        return np.nan
    return float(np.exp((float(logit(q)) - b[0]) / b[1]))


class Cell:
    def __init__(self, recs, d):
        self.d = d
        cf = [x for x in recs if x['kind'] == 'cf']
        ex = [x for x in recs if x['kind'] == 'expert']
        g = lambda L, k: np.array([x[k] for x in L], dtype=float)
        self.r = g(cf, d)
        self.lr = np.log(np.maximum(self.r, 1e-4))
        self.bin = binof(self.r)
        self.traj = np.array([x['traj'] for x in cf])
        self.start = np.array([x['start_id'] for x in cf])
        self.score = dict(a_real=g(cf, f'{AKEY[d]}_real'), a_imag=g(cf, f'{AKEY[d]}_imag'), ai_imag=g(cf, AIKEY[d]),
                          b_real=-g(cf, 'dv_real'), b_imag=-g(cf, 'dv_imag'))
        self.flag = dict(a_real=self.score['a_real'] > 0.5, a_imag=self.score['a_imag'] > 0.5,
                         ai_imag=self.score['ai_imag'] > 0.5,
                         b_real=g(cf, 'dv_real') < TAU[d], b_imag=g(cf, 'dv_imag') < TAU[d])
        self.mem = dict(real=np.array([x[f'{AKEY[d]}_real_m'] for x in cf]), imag=np.array([x[f'{AKEY[d]}_imag_m'] for x in cf]))
        self.ex_traj = np.array([x['traj'] for x in ex])
        self.ex = dict(a_real=g(ex, f'{AKEY[d]}_real') > 0.5, a_imag=g(ex, f'{AKEY[d]}_imag') > 0.5,
                       b_real=g(ex, 'dv_real') < TAU[d], b_imag=g(ex, 'dv_imag') < TAU[d])
        if d == 'r':
            comp = np.where(g(cf, 'r_ang') > g(cf, 'r_pos'), 'angle', np.where(g(cf, 'agent') > g(cf, 'block'), 'agent', 'block'))
        else:
            comp = np.where(g(cf, 'r_ang') > g(cf, 'r_obj_pos'), 'angle', 'block')
        self.comp = comp
        self.fam = np.array([x['fam'] for x in cf])
        self.n = len(cf)

    def set_index(self, trajs):
        trajs = np.asarray(trajs)
        self.tix = np.searchsorted(trajs, self.traj)
        self.ex_tix = np.searchsorted(trajs, self.ex_traj)

    def weights(self, mult):
        """per-record weights from per-trajectory bootstrap multiplicities (array aligned with the set's trajectory list)."""
        if mult is None:
            return np.ones(self.n), np.ones(len(self.ex_traj))
        return mult[self.tix], mult[self.ex_tix]

    def metrics(self, j, tw=None, full=True):
        w, we = self.weights(tw)
        out = {}
        for inp in ('real', 'imag'):
            f = self.flag[f'{j}_{inp}'].astype(float)
            for k, b in enumerate(BINS):
                mk = self.bin == k
                out[f'p_{b}_{inp}'] = float((f[mk] * w[mk]).sum() / w[mk].sum()) if w[mk].sum() > 0 else np.nan
            fe = self.ex[f'{j}_{inp}'].astype(float)
            out[f'p_expert_{inp}'] = float((fe * we).sum() / we.sum()) if we.sum() > 0 else np.nan
            out[f'ratio_near_far_{inp}'] = (out[f'p_near_{inp}'] / out[f'p_far_{inp}'] if out[f'p_far_{inp}'] > 0
                                            else (np.inf if out[f'p_near_{inp}'] > 0 else np.nan))
            s = self.score[f'{j}_{inp}']
            sm, nm, fm = self.bin == 0, self.bin == 1, self.bin == 3
            out[f'auc_sn_{inp}'] = wauc(s[sm], w[sm], s[nm], w[nm])
            out[f'auc_sf_{inp}'] = wauc(s[sm], w[sm], s[fm], w[fm])
            bfit = wlogit(self.lr, f, w)
            out[f'b0_{inp}'], out[f'b1_{inp}'] = float(bfit[0]), float(bfit[1])
            out[f'r50_{inp}'], out[f'r10_{inp}'] = r_at(bfit, 0.5), r_at(bfit, 0.1)
            fail = self.r >= 1
            bf = wlogit(self.lr[fail], f[fail], w[fail])
            out[f'slope_fail_{inp}'] = float(bf[1])
        out['delta_auc_sn'] = out['auc_sn_real'] - out['auc_sn_imag']
        out['delta_auc_sf'] = out['auc_sf_real'] - out['auc_sf_imag']
        out['delta_r10'] = out['r10_imag'] - out['r10_real']
        out['near_shift'] = out['p_near_imag'] - out['p_near_real']
        out['succ_shift'] = out['p_success_imag'] - out['p_success_real']
        if j == 'a':
            s = self.score['ai_imag']
            sm, nm, fm = self.bin == 0, self.bin == 1, self.bin == 3
            out['auc_sn_ceiling'] = wauc(s[sm], w[sm], s[nm], w[nm])
            out['auc_sf_ceiling'] = wauc(s[sm], w[sm], s[fm], w[fm])
            fc = self.flag['ai_imag'].astype(float)
            out['p_near_ceiling'] = float((fc[nm] * w[nm]).sum() / w[nm].sum()) if w[nm].sum() > 0 else np.nan
        if full:
            curve = []
            for lo, hi in zip(CURVE_EDGES[:-1], CURVE_EDGES[1:]):
                mk = (self.r >= lo) & (self.r < hi)
                if mk.sum() < 10:
                    continue
                curve.append(dict(lo=lo, hi=hi, n=int(mk.sum()), r_med=float(np.median(self.r[mk])),
                                  p_imag=float(self.flag[f'{j}_imag'][mk].mean()), p_real=float(self.flag[f'{j}_real'][mk].mean()),
                                  **({'p_ceiling': float(self.flag['ai_imag'][mk].mean())} if j == 'a' else {})))
            out['curve'] = curve
            out['n'] = {b: int((self.bin == k).sum()) for k, b in enumerate(BINS)}
            out['n_expert'] = int(len(self.ex_traj))
            nm = self.bin == 1
            comp = {}
            for c in np.unique(self.comp[nm]):
                mk = nm & (self.comp == c)
                comp[c] = dict(n=int(mk.sum()), p_imag=float(self.flag[f'{j}_imag'][mk].mean()),
                               p_real=float(self.flag[f'{j}_real'][mk].mean()))
            out['near_composition'] = comp
            out['near_by_family'] = {fm_: dict(n=int((nm & (self.fam == fm_)).sum()),
                                               p_imag=float(self.flag[f'{j}_imag'][nm & (self.fam == fm_)].mean()) if (nm & (self.fam == fm_)).any() else None)
                                     for fm_ in ('offset', 'iid')}
            if j == 'a':
                mem = {}
                for inp in ('real', 'imag'):
                    P = self.mem[inp]
                    pn = (P[nm] > 0.5).mean(0)
                    mem[f'p_near_{inp}_members'] = pn.tolist()
                    mem[f'p_near_{inp}_sd'] = float(pn.std(ddof=1))
                    mem[f'auc_sn_{inp}_members'] = [wauc(P[self.bin == 0, k], np.ones((self.bin == 0).sum()), P[nm, k], np.ones(nm.sum()))
                                                    for k in range(P.shape[1])]
                out['members'] = mem
        return out

    def glmm(self, j, inp):
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            import scipy.sparse as sp
            y = self.flag[f'{j}_{inp}'].astype(float)
            u, gi = np.unique(self.start, return_inverse=True)
            X = np.stack([np.ones(self.n), self.lr], 1)
            Z = sp.csr_matrix((np.ones(self.n), (np.arange(self.n), gi)), shape=(self.n, len(u))).toarray()
            fit = BinomialBayesMixedGLM(y, X, Z, np.zeros(len(u), int)).fit_vb()
            b = np.asarray(fit.fe_mean)
            return dict(b0=float(b[0]), b1=float(b[1]), b1_sd=float(fit.fe_sd[1]), re_sd=float(np.exp(fit.vcp_mean[0])),
                        n_groups=int(len(u)), r50=r_at(b, 0.5), r10=r_at(b, 0.1))
        except Exception as e:  # noqa
            return dict(error=str(e)[:200])


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    t0 = time.time()
    cells = {}
    for st in SETS:
        for H in HS:
            for d in DEFS:
                recs = load(st, H, d)
                if recs:
                    cells[(st, H, d)] = Cell(recs, d)
    out = dict(B=B, cells={})
    keys_boot = ['p_near_imag', 'p_near_real', 'p_far_imag', 'p_success_imag', 'p_success_real', 'ratio_near_far_imag',
                 'auc_sn_real', 'auc_sn_imag', 'auc_sf_real', 'auc_sf_imag', 'delta_auc_sn', 'delta_auc_sf',
                 'r50_imag', 'r50_real', 'r10_imag', 'r10_real', 'delta_r10', 'slope_fail_imag', 'slope_fail_real',
                 'near_shift', 'succ_shift', 'auc_sn_ceiling', 'p_near_ceiling']
    for st in SETS:
        trajs = sorted({int(t) for (s, H, d), c in cells.items() if s == st for t in c.traj})
        if not trajs:
            continue
        rng = np.random.RandomState(20261022)
        boots = [np.bincount(rng.randint(0, len(trajs), len(trajs)), minlength=len(trajs)).astype(float) for _ in range(B)]
        for (s_, H_, d_), c_ in cells.items():
            if s_ == st:
                c_.set_index(trajs)
        for d in DEFS:
            slope_pts = {j: [] for j in ('a', 'b')}
            per_H = {}
            for H in HS:
                c = cells.get((st, H, d))
                if c is None:
                    continue
                cell = {}
                for j in ('a', 'b'):
                    pt = c.metrics(j)
                    bs = [c.metrics(j, tw, full=False) for tw in boots]
                    pt['ci'] = {k: ci([b_[k] for b_ in bs]) for k in keys_boot if k in pt}
                    sf = np.array([b_['slope_fail_imag'] for b_ in bs], float)
                    pt['p_boot_slope_fail_imag_ge0'] = float(np.mean(sf[np.isfinite(sf)] >= 0)) if np.isfinite(sf).any() else None
                    pt['glmm_imag'], pt['glmm_real'] = c.glmm(j, 'imag'), c.glmm(j, 'real')
                    pt['_boot_gap'] = [b_['auc_sn_imag'] - b_['auc_sn_real'] for b_ in bs]
                    cell[j] = pt
                per_H[H] = cell
                print(f'{st} H{H} {d}: a near imag {cell["a"]["p_near_imag"]:.3f} real {cell["a"]["p_near_real"]:.3f} far imag '
                      f'{cell["a"]["p_far_imag"]:.3f} | AUROC sn real {cell["a"]["auc_sn_real"]:.3f} imag {cell["a"]["auc_sn_imag"]:.3f} '
                      f'ceil {cell["a"]["auc_sn_ceiling"]:.3f} | sf real {cell["a"]["auc_sf_real"]:.3f} imag {cell["a"]["auc_sf_imag"]:.3f} '
                      f'| r50 {cell["a"]["r50_imag"]:.2f} dr10 {cell["a"]["delta_r10"]:.2f} ({time.time() - t0:.0f}s)', flush=True)
            # horizon slope of [AUROC_imag - AUROC_real](success vs near), same bootstrap resamples across H
            hs = [H for H in HS if H in per_H]
            for j in ('a', 'b'):
                if len(hs) >= 3:
                    gap_pt = [per_H[H][j]['auc_sn_imag'] - per_H[H][j]['auc_sn_real'] for H in hs]
                    slope_pt = float(np.polyfit(hs, gap_pt, 1)[0])
                    G = np.array([per_H[H][j]['_boot_gap'] for H in hs])          # (nH, B)
                    sl = [float(np.polyfit(hs, G[:, b_], 1)[0]) for b_ in range(G.shape[1]) if np.isfinite(G[:, b_]).all()]
                    out['cells'][f'{st}/{d}/horizon_slope_{j}'] = dict(H=hs, gap=gap_pt, slope=slope_pt, ci=ci(sl),
                                                                        p_boot_ge0=float(np.mean(np.array(sl) >= 0)) if sl else None)
            for H in hs:
                for j in ('a', 'b'):
                    per_H[H][j].pop('_boot_gap', None)
                out['cells'][f'{st}/H{H}/{d}'] = per_H[H]
    # pooled-grid sensitivity (both targeted grids of a (set, H) measured on definition d; point estimates)
    pooled = {}
    for st in SETS:
        for H in HS:
            recs = (load(st, H, 'r') or []) + (load(st, H, 'r_obj') or [])
            if not recs:
                continue
            for d in DEFS:
                c = Cell(recs, d)
                m_ = c.metrics('a')
                pooled[f'{st}/H{H}/{d}'] = {k: m_[k] for k in ('p_near_imag', 'p_near_real', 'p_far_imag', 'auc_sn_real', 'auc_sn_imag',
                                                                'auc_sf_real', 'auc_sf_imag', 'n')}
    out['pooled_grids'] = pooled
    # verdicts (v1.1): M1 primary = val, H5, d = r, judge a
    P = out['cells'].get('val/H5/r', {}).get('a')
    if P:
        sl = out['cells'].get('val/r/horizon_slope_a', {})
        h1a = dict(p_near=P['p_near_imag'], p_far=P['p_far_imag'], ratio=P['ratio_near_far_imag'],
                   slope_fail=P['slope_fail_imag'], slope_fail_ci=P['ci'].get('slope_fail_imag'),
                   near_ge_0p3=bool(P['p_near_imag'] >= 0.3),
                   ratio_ge_3=bool(P['p_far_imag'] == 0 and P['p_near_imag'] > 0 or (P['p_far_imag'] > 0 and P['p_near_imag'] / P['p_far_imag'] >= 3)),
                   monotone=bool(P['ci'].get('slope_fail_imag') is not None and P['ci']['slope_fail_imag'][1] < 0),
                   p_boot_monotone=P.get('p_boot_slope_fail_imag_ge0'),
                   perception_floor_p_near_real=P['p_near_real'])
        h1a['H1a_M1'] = bool(h1a['near_ge_0p3'] and h1a['ratio_ge_3'] and h1a['monotone'])
        h1b = dict(delta_auc_sn=P['delta_auc_sn'], delta_auc_sn_ci=P['ci'].get('delta_auc_sn'), delta_auc_sf=P['delta_auc_sf'],
                   delta_auc_sf_ci=P['ci'].get('delta_auc_sf'), horizon_slope=sl.get('slope'), horizon_slope_ci=sl.get('ci'),
                   sn_ge_0p05=bool(P['delta_auc_sn'] >= 0.05), sf_le_0p02=bool(P['delta_auc_sf'] <= 0.02),
                   slope_neg=bool(sl.get('ci') is not None and sl['ci'][1] < 0), p_boot_slope=sl.get('p_boot_ge0'))
        h1b['H1b_M1'] = bool(h1b['sn_ge_0p05'] and h1b['sf_le_0p02'] and h1b['slope_neg'])
        if TASK == 'pusht':
            out['verdict_M1'] = dict(cell='val/H5/r/a', H1a=h1a, H1b=h1b)
            print('VERDICT M1', json.dumps(out['verdict_M1']))
        else:   # E7 (DESIGN_main 12.8): H1b clauses + H4b at the primary cell; H1a-type quantities are descriptive only
            h1b['H1b_E7'] = h1b.pop('H1b_M1')
            ceil = P.get('auc_sn_ceiling')
            h4b = dict(auc_sn_real=P['auc_sn_real'], auc_sn_ceiling=ceil,
                       gap=(P['auc_sn_real'] - ceil) if ceil is not None else None,
                       real_ge_0p95=bool(P['auc_sn_real'] >= 0.95),
                       gap_ge_0p05=bool(ceil is not None and P['auc_sn_real'] - ceil >= 0.05))
            h4b['H4b_E7'] = bool(h4b['real_ge_0p95'] and h4b['gap_ge_0p05'])
            desc = {k: P.get(k) for k in ('p_near_real', 'p_near_imag', 'p_far_imag', 'ratio_near_far_imag', 'slope_fail_imag',
                                           'p_success_real', 'p_success_imag', 'succ_shift', 'near_shift', 'r50_real', 'r50_imag',
                                           'r10_real', 'r10_imag', 'delta_r10', 'auc_sf_real', 'auc_sf_imag', 'n')}
            desc['ci'] = {k: P['ci'].get(k) for k in ('p_near_imag', 'succ_shift', 'near_shift', 'slope_fail_imag', 'delta_r10')}
            desc['h1a_type_reference'] = {k: h1a[k] for k in ('near_ge_0p3', 'ratio_ge_3', 'monotone')}
            cc = f'{RES}/convcheck_e7.json'
            out['verdict_E7'] = dict(cell='val/H5/r/a', H1b=h1b, H4b=h4b, descriptive=desc,
                                     convention_check=(json.load(open(cc)) if os.path.exists(cc) else None))
            print('VERDICT E7', json.dumps({k: out['verdict_E7'][k] for k in ('H1b', 'H4b')}))
    out['time_s'] = time.time() - t0
    json.dump(out, open(f'{RES}/{OUT}', 'w'), indent=1, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else str(o))
    print('M1_ANALYZE_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
