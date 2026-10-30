"""B E2 M1 analysis (DESIGN_main v1 sections 4.3 / 6.2 / 6.7, v2.2 section 7). Runs in the CPU analysis env (b-ana).
Primary (H2 cell): the 24 DP checkpoints, H = 5, judge a_std ensemble (mean probability > 0.5, as E1), truth r_obj vs the
fixed goal pose: s = P(r_obj < 1), n = P(1 <= r_obj < 2), sh = imagined judged success, sh_real = the same judge on the
real final frame of the same window; o = sh - s = o_imag (sh - sh_real) + o_judge (sh_real - s). H = 3, 7, 10 auxiliary.
Episode cluster bootstrap B = 10,000: the 100 initial states are shared by all checkpoints and resampled jointly.
H2a p-value: one-sided policy-level permutation test of Spearman(n, o_imag) (10,000 permutations).
Control (reported separately, not in the H2 verdict): R2-12 main set, 8 perturbation policies, original r, judges
a = a_full ensemble and b = d_vis < tau_full (frozen), cluster bootstrap over the source trajectories (B = 10,000).
usage: python e2_m1_analyze.py [B]"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2stats as S2

RES = os.path.expanduser(os.environ.get('B_RES', '~/pmx/B/results/E2/m1'))
HS = (3, 5, 7, 10)
TAU_FULL = 0.2125
CTRL_POL = ['far_L', 'mixA_L', 'mixB_L', 'near_L', 'iid_L', 'far_H', 'near_H', 'iid_H']
KEYS_CI = ['spearman_n_oimag', 'spearman_n_o', 'spearman_n_ojudge', 'pearson_sh_s', 'spearman_sh_s', 'pearson_shreal_s',
           'mmrv', 'mmrv_realjudge', 'fgre', 'fgre_realjudge', 'h2b_n_pairs', 'h2b_max_gap_similar', 'tau_similar',
           'tau_similar_realjudge', 'mean_o', 'mean_abs_o', 'mean_oimag', 'mean_ojudge']


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None


def fl(x):
    x = float(x)
    return x if np.isfinite(x) else (None if np.isnan(x) else (1e9 if x > 0 else -1e9))


def analyze(Sm, Nm, Ji, Jr, W, names, extra=None):
    """Sm, Nm, Ji, Jr: (N_policies, n_units) per-episode indicators (or fractions); W: (B, n_units) bootstrap weights
    (multiplicities of the resampled units). Returns point estimates, CIs, p-values and the per-policy table."""
    s, n, sh, shr = Sm.mean(1), Nm.mean(1), Ji.mean(1), Jr.mean(1)
    pt = {k: fl(v[0]) for k, v in S2.policy_stats(s[None], sh[None], shr[None], n[None]).items()}
    den = W.sum(1, keepdims=True)
    sb, nb, shb, shrb = (W @ Sm.T) / den, (W @ Nm.T) / den, (W @ Ji.T) / den, (W @ Jr.T) / den
    bs = S2.policy_stats(sb, shb, shrb, nb)
    out = dict(point=pt, ci={k: ci(bs[k]) for k in KEYS_CI})
    out['p_boot_spearman_n_oimag_le0'] = float(np.mean(bs['spearman_n_oimag'] <= 0))
    rho, p_perm, kind = S2.perm_spearman_p(n, sh - shr)
    out['perm_spearman_n_oimag'] = dict(rho=rho, p_one_sided=p_perm, kind=kind)
    q, mx = S2.h2b_pairs(s[None], sh[None])
    i, j = np.triu_indices(len(s), 1)
    qual = [(names[a], names[b], float(s[a]), float(s[b]), float(sh[a]), float(sh[b])) for a, b in zip(i, j)
            if abs(s[a] - s[b]) <= S2.DELTA + S2.EPS and abs(sh[a] - sh[b]) >= S2.GAP - S2.EPS]
    out['h2b_pairs'] = qual
    tw, idx, nties = S2.tau_window(s, sh)
    out['tau_densest_window'] = dict(tau_b=fl(tw), policies=[names[k] for k in idx], s=[float(s[k]) for k in idx],
                                     sh=[float(sh[k]) for k in idx], n_tied_windows=nties)
    twr, _, _ = S2.tau_window(s, shr)
    out['tau_densest_window_realjudge'] = fl(twr)
    tab = []
    for k, nm in enumerate(names):
        row = dict(policy=nm, s=float(s[k]), n=float(n[k]), sh=float(sh[k]), sh_real=float(shr[k]), o=float(sh[k] - s[k]),
                   o_imag=float(sh[k] - shr[k]), o_judge=float(shr[k] - s[k]),
                   ci_o_imag=ci(shb[:, k] - shrb[:, k]), ci_sh=ci(shb[:, k]), ci_s=ci(sb[:, k]))
        if extra:
            row.update({kk: float(vv[k]) for kk, vv in extra.items()})
        tab.append(row)
    out['table'] = tab
    out['verdict_inputs'] = dict(H2a=pt['spearman_n_oimag'], H2b_pairs=pt['h2b_n_pairs'], H2b_tau=pt['tau_similar'],
                                 H2c_pearson=pt['pearson_sh_s'], H2c_mmrv=pt['mmrv'])
    return out


def boot_weights(n_units, B, seed):
    rng = np.random.RandomState(seed)
    W = np.zeros((B, n_units))
    idx = rng.randint(0, n_units, (B, n_units))
    np.add.at(W, (np.arange(B)[:, None], idx), 1.0)
    return W


def bands(x, lo, hi):
    return ((x >= lo) & (x < hi)).astype(float)


def dp_part(B):
    recs = [json.loads(l) for l in open(f'{RES}/dp_windows.jsonl')]
    names = sorted({r['policy'] for r in recs}, key=lambda p: (int(p[1:3]), int(p.split('ep')[1])))
    pidx = {p: k for k, p in enumerate(names)}
    out = dict(policies=names)
    W = boot_weights(100, B, 20261022)
    for H in HS:
        rr = [r for r in recs if r['H'] == H]
        shp = (len(names), 100)
        ro, ji, jr, sdp, pad = np.full(shp, np.nan), np.zeros(shp), np.zeros(shp), np.zeros(shp), 0
        for r in rr:
            a, b = pidx[r['policy']], r['ep']
            ro[a, b] = r['r_obj']
            ji[a, b] = float(r['p_std_imag'] > 0.5)
            jr[a, b] = float(r['p_std_real'] > 0.5)
            sdp[a, b] = float(r['success_dp'])
            pad += int(r['pad'] > 0)
        assert np.isfinite(ro).all(), f'H{H}: missing windows'
        Sm, Nm = (ro < 1).astype(float), bands(ro, 1, 2)
        A = analyze(Sm, Nm, ji, jr, W, names, extra=dict(mid=bands(ro, 2, 3).mean(1), far=(ro >= 3).mean(1), success_dp=sdp.mean(1)))
        # perception floor and imagined misjudgment on the pooled windows (E3 secondary decomposition, DP set)
        nb_, sb_, fb_ = Nm > 0, Sm > 0, ro >= 3
        A['pooled'] = dict(n_windows=int(ro.size), padded=pad,
                           p_judged_success_given_near_real=float(jr[nb_].mean()), p_judged_success_given_near_imag=float(ji[nb_].mean()),
                           p_judged_success_given_success_real=float(jr[sb_].mean()), p_judged_success_given_success_imag=float(ji[sb_].mean()),
                           p_judged_success_given_far_real=float(jr[fb_].mean()) if fb_.any() else None,
                           p_judged_success_given_far_imag=float(ji[fb_].mean()) if fb_.any() else None)
        pn_r, pn_i = A['pooled']['p_judged_success_given_near_real'], A['pooled']['p_judged_success_given_near_imag']
        A['pooled']['imagination_share_near'] = float((pn_i - pn_r) / pn_i) if pn_i > 0 else None
        out[f'H{H}'] = A
        p = A['point']
        print(f"DP H{H}: Spearman(n, o_imag) {p['spearman_n_oimag']:.3f} {A['ci']['spearman_n_oimag']} perm p {A['perm_spearman_n_oimag']['p_one_sided']:.4f}"
              f" | pairs {p['h2b_n_pairs']:.0f} tau_sim {p['tau_similar']} | Pearson {p['pearson_sh_s']:.3f} MMRV {p['mmrv']:.3f} FGRE {p['fgre']:.3f}"
              f" | near real/imag {pn_r:.3f}/{pn_i:.3f}", flush=True)
    P = out['H5']['point']
    out['verdict_M1'] = dict(
        cell='DP 24 checkpoints, H = 5, a_std ensemble, n = P(1 <= r_obj < 2)',
        H2a=dict(spearman=P['spearman_n_oimag'], ci=out['H5']['ci']['spearman_n_oimag'], ge_0p6=bool(P['spearman_n_oimag'] >= 0.6),
                 p_perm_one_sided=out['H5']['perm_spearman_n_oimag']['p_one_sided'], p_boot_le0=out['H5']['p_boot_spearman_n_oimag_le0']),
        H2b=dict(n_pairs=P['h2b_n_pairs'], pairs=out['H5']['h2b_pairs'], tau_similar=P['tau_similar'],
                 tau_le_0p5=bool(P['tau_similar'] is not None and P['tau_similar'] <= 0.5),
                 exists_pair=bool(P['h2b_n_pairs'] >= 1), tau_densest_window=out['H5']['tau_densest_window']),
        H2c=dict(pearson=P['pearson_sh_s'], mmrv=P['mmrv'], pearson_ge_0p8=bool(P['pearson_sh_s'] >= 0.8),
                 mmrv_le_0p05=bool(P['mmrv'] <= 0.05), fgre=P['fgre']))
    v = out['verdict_M1']
    v['H2a']['holds'] = v['H2a']['ge_0p6']
    v['H2b']['holds'] = bool(v['H2b']['exists_pair'] and v['H2b']['tau_le_0p5'])
    v['H2c']['holds'] = bool(v['H2c']['pearson_ge_0p8'] and v['H2c']['mmrv_le_0p05'])
    print('VERDICT M1', json.dumps({k: (vv.get('holds') if isinstance(vv, dict) else vv) for k, vv in v.items()}), flush=True)
    return out


def ctrl_part(B):
    recs = [json.loads(l) for l in open(f'{RES}/ctrl_main.jsonl')]
    eps = sorted({r['ep'] for r in recs})
    traj_of = {r['ep']: r['traj'] for r in recs}
    ut = sorted(set(traj_of.values()))
    tix = np.array([ut.index(traj_of[e]) for e in eps])
    rng = np.random.RandomState(20261023)
    Wt = np.zeros((B, len(ut)))
    np.add.at(Wt, (np.arange(B)[:, None], rng.randint(0, len(ut), (B, len(ut)))), 1.0)
    W = Wt[:, tix]
    shp = (len(CTRL_POL), len(eps))
    rv, pa_i, pa_r, dv_i, dv_r = (np.full(shp, np.nan) for _ in range(5))
    pk = {p: k for k, p in enumerate(CTRL_POL)}
    ek = {e: k for k, e in enumerate(eps)}
    for r in recs:
        if r['policy'] not in pk:
            continue
        a, b = pk[r['policy']], ek[r['ep']]
        rv[a, b], pa_i[a, b], pa_r[a, b], dv_i[a, b], dv_r[a, b] = r['r'], r['p_full_imag'], r['p_full_real'], r['dv_imag'], r['dv_real']
    assert np.isfinite(rv).all()
    Sm, Nm = (rv < 1).astype(float), bands(rv, 1, 2)
    out = dict(policies=CTRL_POL, n_episodes=len(eps), n_clusters=len(ut))
    for j, (ji, jr) in dict(a=((pa_i > 0.5), (pa_r > 0.5)), b=((dv_i < TAU_FULL), (dv_r < TAU_FULL))).items():
        A = analyze(Sm, Nm, ji.astype(float), jr.astype(float), W, CTRL_POL, extra=dict(far=(rv >= 3).mean(1)))
        out[j] = A
        p = A['point']
        print(f"CTRL judge {j}: Spearman(n, o_imag) {p['spearman_n_oimag']:.3f} {A['ci']['spearman_n_oimag']} | Spearman(n, o) {p['spearman_n_o']:.3f}"
              f" | pairs {p['h2b_n_pairs']:.0f} tau_sim {p['tau_similar']} | Pearson {p['pearson_sh_s']:.3f} MMRV {p['mmrv']:.3f} FGRE {p['fgre']:.3f}", flush=True)
    return out


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    t0 = time.time()
    out = dict(B=B)
    out['dp'] = dp_part(B)
    if os.path.exists(f'{RES}/ctrl_main.jsonl'):
        out['ctrl'] = ctrl_part(B)
    out['time_s'] = time.time() - t0
    with open(f'{RES}/e2_m1_analysis.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('E2_M1_ANALYZE_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
