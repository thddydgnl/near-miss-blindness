"""B E3 analysis: judge x input decomposition of near-miss misjudgments (DESIGN_main v1 section 6.3, v2.2 section 7).
Runs in the CPU analysis env (b-ana).
For every set / cell and judge j: P(S^_j = 1 | near) on the real final frame (perception floor) and on the imagined final
frame; imagination share = (P_imag - P_real) / P_imag; also P(S^ = 1 | success) and AUROC success vs near (real, imag).
  M1 E1 curve cells (counterfactual records of the cell's own grid, bands of the cell's definition):
      (a) a_full / a_obj ensemble (> 0.5), (b) d_vis < frozen tau, (d) VLM P(Yes) > rule-set threshold (cells run)
      cluster bootstrap over source trajectories (B = 10,000)
  M1 DP set (E2 windows, r_obj vs the standard goal): (a) a_std ensemble, (d) VLM (std rule); bootstrap over episodes
  M3 (E1, p = 0.03 seeds pooled, frozen R2-9 classifier): from the E1 M3 records.
H3 verdict: primary cell = H1 cell (M1 val H5 r (a); M3 p = 0.03): share > 0.5.
usage: python e3_m1_analyze.py [B]"""
import os, sys, json, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_m1_analyze as E2A

E1 = os.path.expanduser('~/pmx/B/results/E1/m1')
E1M3 = os.path.expanduser('~/pmx/B/results/E1/m3')
E2 = os.path.expanduser('~/pmx/B/results/E2/m1')
OUT = os.path.expanduser('~/pmx/B/results/E3/m1')
FEAT3 = os.path.expanduser('~/pmx/B/feat/E3/m1')
FEAT2 = os.path.expanduser('~/pmx/B/feat/E2/m1')
VLM = f'{OUT}/vlm'
TAU = {'r': 0.2125, 'r_obj': 0.345}
AKEY = {'r': 'p_full', 'r_obj': 'p_obj'}
RULE = {'r': 'dset_r', 'r_obj': 'dset_robj'}


def auroc(sp, sn):
    sp, sn = np.asarray(sp, float), np.asarray(sn, float)
    if len(sp) == 0 or len(sn) == 0:
        return None
    o = np.sort(sn)
    lo, hi = np.searchsorted(o, sp, 'left'), np.searchsorted(o, sp, 'right')
    return float((lo + 0.5 * (hi - lo)).sum() / (len(sp) * len(sn)))


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None


def boot_w(n_units, B, seed):
    rng = np.random.RandomState(seed)
    W = np.zeros((B, n_units))
    np.add.at(W, (np.arange(B)[:, None], rng.randint(0, n_units, (B, n_units))), 1.0)
    return W


def decomp(jr, ji, band, unit, Wu, sr=None, si=None):
    """jr, ji: judged-success flags (real / imagined) per record; band: array of 'success'/'near'/... ; unit: cluster
    index per record; Wu: (B, n_units) bootstrap multiplicities. sr / si: continuous scores (AUROC)."""
    nm, sm = band == 'near', band == 'success'
    pr, pi = float(jr[nm].mean()), float(ji[nm].mean())
    W = Wu[:, unit[nm]]
    den = W.sum(1)
    prb, pib = (W @ jr[nm]) / den, (W @ ji[nm]) / den
    with np.errstate(invalid='ignore', divide='ignore'):
        shb = np.where(pib > 0, (pib - prb) / pib, np.nan)
    out = dict(n_near=int(nm.sum()), n_success=int(sm.sum()), p_near_real=pr, p_near_imag=pi,
               increment=pi - pr, share=(pi - pr) / pi if pi > 0 else None,
               ci_p_near_real=ci(prb), ci_p_near_imag=ci(pib), ci_increment=ci(pib - prb), ci_share=ci(shb),
               p_success_real=float(jr[sm].mean()), p_success_imag=float(ji[sm].mean()),
               p_far_real=float(jr[band == 'far'].mean()) if (band == 'far').any() else None,
               p_far_imag=float(ji[band == 'far'].mean()) if (band == 'far').any() else None)
    if sr is not None:
        out.update(auc_sn_real=auroc(sr[sm], sr[nm]), auc_sn_imag=auroc(si[sm], si[nm]),
                   mean_score_near_real=float(sr[nm].mean()), mean_score_near_imag=float(si[nm].mean()))
    return out


def bands(x):
    return np.where(x < 1, 'success', np.where(x < 2, 'near', np.where(x < 3, 'mid', 'far')))


def e1_cells(B, rules):
    res = {}
    for f in sorted(glob.glob(f'{E1}/eval_*_H*_*.jsonl')):
        tag = os.path.basename(f)[5:-6]
        st, Hs, d = tag.split('_', 2)
        recs = [json.loads(l) for l in open(f)]
        cf = np.array([r['kind'] == 'cf' for r in recs])
        g = lambda k: np.array([r[k] for r in recs], dtype=float)
        x = g(d)
        band = bands(x)
        trajs = np.array([r['traj'] for r in recs])
        ut = np.unique(trajs)
        unit = np.searchsorted(ut, trajs)
        Wu = boot_w(len(ut), B, 777)
        sel = lambda a: a[cf]
        cell = {}
        pa_r, pa_i = g(f'{AKEY[d]}_real'), g(f'{AKEY[d]}_imag')
        cell['a'] = decomp(sel(pa_r > 0.5).astype(float), sel(pa_i > 0.5).astype(float), sel(band), sel(unit), Wu, sel(pa_r), sel(pa_i))
        dv_r, dv_i = g('dv_real'), g('dv_imag')
        cell['b'] = decomp(sel(dv_r < TAU[d]).astype(float), sel(dv_i < TAU[d]).astype(float), sel(band), sel(unit), Wu, -sel(dv_r), -sel(dv_i))
        vr, vi = f'{VLM}/e1_{tag}_real.npy', f'{VLM}/e1_{tag}_imag.npy'
        if os.path.exists(vr) and os.path.exists(vi) and RULE[d] in rules:
            Z = np.load(f'{FEAT3}/e1dec_{tag}.npz')
            line = Z['line']
            sr, si = np.full(len(recs), np.nan), np.full(len(recs), np.nan)
            sr[line], si[line] = np.load(vr), np.load(vi)
            assert np.isfinite(sr).all() and np.isfinite(si).all()
            th = rules[RULE[d]]['threshold']
            cell['d'] = decomp(sel(sr > th).astype(float), sel(si > th).astype(float), sel(band), sel(unit), Wu, sel(sr), sel(si))
            cell['d'].update(threshold=th, rule_set=RULE[d], is_judge=rules[RULE[d]]['passes_0p80'],
                             p_expert_real=float((sr[~cf] > th).mean()), p_expert_imag=float((si[~cf] > th).mean()))
        res[tag] = cell
        print(tag, ' '.join(f"{j}: near {c['p_near_real']:.3f}->{c['p_near_imag']:.3f} share {c['share'] if c['share'] is None else round(c['share'], 3)}"
                            for j, c in cell.items()), flush=True)
    return res


def dp_set(B, rules):
    recs = [json.loads(l) for l in open(f'{E2}/dp_windows.jsonl')]
    res = {}
    for H in (3, 5, 7, 10):
        rr = [r for r in recs if r['H'] == H]
        x = np.array([r['r_obj'] for r in rr])
        band = bands(x)
        unit = np.array([r['ep'] for r in rr])
        Wu = boot_w(100, B, 778)
        pr, pi = np.array([r['p_std_real'] for r in rr]), np.array([r['p_std_imag'] for r in rr])
        cell = dict(a_std=decomp((pr > 0.5).astype(float), (pi > 0.5).astype(float), band, unit, Wu, pr, pi))
        vr, vi = f'{VLM}/dp_H{H}_real.npy', f'{VLM}/dp_H{H}_imag.npy'
        if os.path.exists(vr) and os.path.exists(vi) and 'std' in rules:
            Z = np.load(f'{FEAT2}/dp_dec_H{H}.npz')
            uid = [str(u) for u in Z['uid']]
            pos = {u: k for k, u in enumerate(uid)}
            k = np.array([pos[f"{r['policy']}/{r['ep']}"] for r in rr])
            sr, si = np.load(vr)[k], np.load(vi)[k]
            th = rules['std']['threshold']
            cell['d'] = decomp((sr > th).astype(float), (si > th).astype(float), band, unit, Wu, sr, si)
            cell['d'].update(threshold=th, rule_set='std', is_judge=rules['std']['passes_0p80'])
            # secondary (not in the H2 verdict): the E2 policy-level statistics with judge (d) instead of a_std
            names = sorted({r['policy'] for r in rr}, key=lambda q: (int(q[1:3]), int(q.split('ep')[1])))
            pk = {q: i for i, q in enumerate(names)}
            shp = (len(names), 100)
            ro_, jr_, ji_ = np.full(shp, np.nan), np.zeros(shp), np.zeros(shp)
            for q, r in enumerate(rr):
                a, b = pk[r['policy']], r['ep']
                ro_[a, b], jr_[a, b], ji_[a, b] = r['r_obj'], float(sr[q] > th), float(si[q] > th)
            Sm, Nm = (ro_ < 1).astype(float), ((ro_ >= 1) & (ro_ < 2)).astype(float)
            cell['d_policy'] = E2A.analyze(Sm, Nm, ji_, jr_, E2A.boot_weights(100, B, 20261022), names)
        res[f'H{H}'] = cell
        print(f'DP H{H}', ' '.join(f"{j}: near {c['p_near_real']:.3f}->{c['p_near_imag']:.3f} share {c['share'] if c['share'] is None else round(c['share'], 3)}"
                                   for j, c in cell.items() if 'p_near_real' in c), flush=True)
    return res


def m3_cell(B):
    out = {}
    for p in ('0.03', '0.1', '0.3'):
        jr, ji, band, unit = [], [], [], []
        for s in (0, 1, 2):
            z = np.load(f'{E1M3}/e1_p{p}_std_1x_s{s}.npz')
            g_, sc_ = np.where(z['use'])
            x = z['grid_r'][g_]
            jr.append(z['real_logit'][g_, sc_] > 0); ji.append(z['imag_logit'][g_, sc_] > 0)
            band.append(np.where(x < 1, 'success', np.where(x < 2, 'near', np.where(x < 3, 'mid', 'far'))))
            unit.append(sc_)
        jr, ji, band, unit = (np.concatenate(v) for v in (jr, ji, band, unit))
        Wu = boot_w(int(unit.max()) + 1, B, 779)
        out[p] = decomp(jr.astype(float), ji.astype(float), band, unit, Wu)
        print(f"M3 p={p}: near {out[p]['p_near_real']:.3f}->{out[p]['p_near_imag']:.3f} share {out[p]['share']}", flush=True)
    return out


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    t0 = time.time()
    rules = json.load(open(f'{OUT}/d_rules.json')) if os.path.exists(f'{OUT}/d_rules.json') else {}
    out = dict(B=B, rules={k: v for k, v in rules.items() if isinstance(v, dict)})
    out['e1_cells'] = e1_cells(B, rules)
    out['dp'] = dp_set(B, rules)
    out['m3'] = m3_cell(B)
    a = out['e1_cells']['val_H5_r']['a']
    m3 = out['m3']['0.03']
    out['verdict_H3'] = dict(
        M1=dict(cell='val, H = 5, r, judge (a) ensemble (= H1 cell, v2.2 section 7)', p_near_real=a['p_near_real'],
                p_near_imag=a['p_near_imag'], share=a['share'], ci_share=a['ci_share'], holds=bool(a['share'] is not None and a['share'] > 0.5),
                note='the value of this cell was already known from E1 when the cell was fixed (v2.2 section 7)'),
        M3=dict(cell='p = 0.03, seeds 0-2 pooled, frozen R2-9 classifier', p_near_real=m3['p_near_real'], p_near_imag=m3['p_near_imag'],
                share=m3['share'], ci_share=m3['ci_share'], holds=bool(m3['share'] is not None and m3['share'] > 0.5)))
    print('VERDICT H3', json.dumps({k: v['holds'] for k, v in out['verdict_H3'].items()}), flush=True)
    out['time_s'] = time.time() - t0
    with open(f'{OUT}/e3_m1_analysis.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('E3_ANALYZE_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
