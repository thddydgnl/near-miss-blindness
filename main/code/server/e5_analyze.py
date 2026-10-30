"""B E5 analysis (DESIGN_main v3.1 section 9; CPU analysis env).
H5a, three settings: M1 (val H5 r, judge a_full; fine-tunes from the public checkpoint), M3 p = 0.03 and M3 p = 0.1
(E1 grid, frozen R2-9 classifier; retrained from scratch). Prescriptions (0) reference, (i) rw, (ii) near, (iii) cp,
(iv) full, (v) bw; 3 seeds pooled per record. In a setting: NM(ii) < NM(i) and NM(ii) < NM(iv), each with the paired 95%
CI of the difference below 0, and SM(ii) - SM(0) <= 0.05 (point). H5a holds in >= 2 of 3 settings.
H5b (M1 DP set, H5, public model): MC dropout K = 16; judged success iff the agreement a >= q*, q* in {1/16..16/16}
chosen on the dev rollouts (minimum mean |o_pi| over the 24 dev policies, ties -> smaller q). On the E2 evaluation set:
mean |o_pi| and FGRE vs the E2 deterministic judgments; paired episode bootstrap (10,000, the 100 shared initial states).
Abstention (0 < a < 1 windows dropped from the judged rate) is reported as auxiliary.
Writes results/E5/e5_analysis.json. usage: python e5_analyze.py [B]"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e45lib as L
import e2stats as S2
from e2_m1_analyze import analyze, boot_weights as boot_w

R5 = os.path.expanduser('~/pmx/B/results/E5')
E2 = os.path.expanduser('~/pmx/B/results/E2/m1')
PRESC = dict(ref='(0) reference', rw='(i) uniform reweighting', near='(ii) near-boundary counterfactuals',
             cp='(iii) contrast pairs', full='(iv) CureWM-style full range', bw='(v) boundary-weighted loss')


def setting_result(J, band, unit, W):
    res = {}
    for k, Jk in J.items():
        pt, bs = L.nm_sm(Jk, band, unit, W)
        res[k] = dict(NM=pt['NM'], SM=pt['SM'], ci_NM=L.ci(bs['NM']), ci_SM=L.ci(bs['SM']))
    res['near_minus_rw'] = L.paired(J['near'], J['rw'], band, unit, W)
    res['near_minus_full'] = L.paired(J['near'], J['full'], band, unit, W)
    res['near_minus_ref'] = L.paired(J['near'], J['ref'], band, unit, W)
    for k in ('rw', 'full', 'cp', 'bw'):
        if k in J:
            res[f'{k}_minus_ref'] = L.paired(J[k], J['ref'], band, unit, W)
    a, b = res['near_minus_rw']['NM'], res['near_minus_full']['NM']
    dsm = res['near']['SM'] - res['ref']['SM']
    res['criteria'] = dict(nm_ii_below_i=bool(a['diff'] < 0 and a['ci'][1] < 0), nm_ii_below_iv=bool(b['diff'] < 0 and b['ci'][1] < 0),
                           sm_increase=dsm, sm_increase_le_0p05=bool(dsm <= 0.05))
    res['holds'] = bool(all(v for k, v in res['criteria'].items() if k != 'sm_increase'))
    return res


def h5a(B):
    out = {}
    C = L.M1Cell('val_H5_r')
    W = L.boot_w(C.n_units, B, 5501)
    J = {'ref': C.judged('S97')}
    for k in ('rw', 'near', 'cp', 'full', 'bw'):
        if os.path.exists(f'{L.E4M1}/imag_val_H5_r_{k}_s2.npz'):
            J[k] = C.judged(k)
    out['M1'] = setting_result(J, C.band, C.unit, W)
    G = L.M3Grid()
    Wg = L.boot_w(G.n_units, B, 5502)
    for p in ('0.03', '0.1'):
        J = {'ref': G.judged_e5(f'p{p}_std_1x_s{{s}}')}
        for k in ('rw', 'near', 'cp', 'full', 'bw'):
            J[k] = G.judged_e5(f'e5_p{p}_{k}_s{{s}}')
        out[f'M3_p{p}'] = setting_result(J, G.band, G.unit, Wg)
    for k, v in out.items():
        print(f"H5a {k}: NM ref/rw/near/full/cp/bw " + '/'.join(f"{v[q]['NM']:.3f}" for q in ('ref', 'rw', 'near', 'full', 'cp', 'bw') if q in v)
              + f" | SM " + '/'.join(f"{v[q]['SM']:.3f}" for q in ('ref', 'rw', 'near', 'full', 'cp', 'bw') if q in v) + f" -> {v['holds']} {v['criteria']}", flush=True)
    n = sum(v['holds'] for v in out.values())
    out['verdict'] = dict(settings_holding=n, H5a=bool(n >= 2))
    # post hoc (not part of the verdict): M3 with the R2-9 near band [1, 1.2) for NM (the band of the (ii) failures)
    ph = {}
    z = np.load(f'{L.E1M3}/e1_p0.03_std_1x_s0.npz')
    r = z['grid_r'][G.g]
    band29 = np.where(r < 1, 'success', np.where(r < 1.2, 'near', 'other'))
    for p in ('0.03', '0.1'):
        J = {'ref': G.judged_e5(f'p{p}_std_1x_s{{s}}')}
        for k in ('rw', 'near', 'cp', 'full', 'bw'):
            J[k] = G.judged_e5(f'e5_p{p}_{k}_s{{s}}')
        ph[f'M3_p{p}_band_1_1.2'] = setting_result(J, band29, G.unit, Wg)
    out['posthoc_r29_band'] = ph
    for k, v in ph.items():
        print(f"post hoc {k}: NM ref/rw/near/full " + '/'.join(f"{v[q]['NM']:.3f}" for q in ('ref', 'rw', 'near', 'full')) + f" -> {v['criteria']}", flush=True)
    return out


def policy_arrays(D, judged):
    names = sorted(set(D['policy']), key=lambda p: (int(p[1:3]), int(p.split('ep')[1])))
    pk = {p: i for i, p in enumerate(names)}
    shp = (len(names), 100)
    ro, J = np.full(shp, np.nan), np.full(shp, np.nan)
    for i, (p, e) in enumerate(zip(D['policy'], D['ep'])):
        ro[pk[p], e] = D['r_obj'][i]
        J[pk[p], e] = judged[i]
    return names, ro, J


def h5b(B):
    dev = np.load(f'{R5}/m1/mcdrop_dev.npz')
    ev = np.load(f'{R5}/m1/mcdrop_eval.npz')
    out = {}
    # q* on the dev rollouts (25 episodes per policy)
    a_dev = (dev['p_mc'] > 0.5).mean(0)
    pols = sorted(set(dev['policy']))
    s_dev = np.array([(dev['r_obj'][dev['policy'] == p] < 1).mean() for p in pols])
    scores = {}
    for k in range(1, 17):
        q = k / 16
        sh = np.array([(a_dev[dev['policy'] == p] >= q - 1e-9).mean() for p in pols])
        scores[k] = float(np.abs(sh - s_dev).mean())
    best = min(scores.values())
    kq = min(k for k, v in scores.items() if v <= best + 1e-12)
    out['dev'] = dict(mean_abs_o_by_q={f'{k}/16': v for k, v in scores.items()}, q_star=f'{kq}/16',
                      mean_abs_o_deterministic=float(np.abs(np.array([(dev['p_det'][dev['policy'] == p] > 0.5).mean() for p in pols]) - s_dev).mean()))
    q = kq / 16
    # evaluation set: E2 deterministic baseline (dp_windows.jsonl, H5) vs MC-dropout discount
    recs = [json.loads(l) for l in open(f'{E2}/dp_windows.jsonl')]
    base = {(r['policy'], r['ep']): float(r['p_std_imag'] > 0.5) for r in recs if r['H'] == 5}
    real = {(r['policy'], r['ep']): float(r['p_std_real'] > 0.5) for r in recs if r['H'] == 5}
    keys = list(zip(ev['policy'], ev['ep']))
    j_base = np.array([base[(p, int(e))] for p, e in keys])
    j_real = np.array([real[(p, int(e))] for p, e in keys])
    a_ev = (ev['p_mc'] > 0.5).mean(0)
    j_disc = (a_ev >= q - 1e-9).astype(float)
    j_major = (a_ev >= 0.5).astype(float)
    D = dict(policy=ev['policy'], ep=ev['ep'].astype(int), r_obj=ev['r_obj'])
    names, ro, Jb = policy_arrays(D, j_base)
    _, _, Jd = policy_arrays(D, j_disc)
    _, _, Jm = policy_arrays(D, j_major)
    _, _, Jr = policy_arrays(D, j_real)
    Sm, Nm = (ro < 1).astype(float), ((ro >= 1) & (ro < 2)).astype(float)
    W = boot_w(100, B, 20261022)
    res = {}
    for k, Jx in (('deterministic', Jb), ('mc_discount', Jd), ('mc_majority', Jm)):
        res[k] = analyze(Sm, Nm, Jx, Jr, W, names)
    # paired difference of mean |o| (discount - deterministic)
    den = W.sum(1, keepdims=True)
    sb = (W @ Sm.T) / den
    mabs = {k: np.abs((W @ Jx.T) / den - sb).mean(1) for k, Jx in (('deterministic', Jb), ('mc_discount', Jd))}
    dlt = mabs['mc_discount'] - mabs['deterministic']
    pt = {k: res[k]['point']['mean_abs_o'] for k in ('deterministic', 'mc_discount', 'mc_majority')}
    out['eval'] = dict(q_star=f'{kq}/16', results=res, mean_abs_o=pt, diff_mean_abs_o=pt['mc_discount'] - pt['deterministic'],
                       ci_diff_mean_abs_o=L.ci(dlt), p_boot_ge0=float(np.mean(dlt >= 0)),
                       fgre=dict(deterministic=res['deterministic']['point']['fgre'], mc_discount=res['mc_discount']['point']['fgre']))
    # abstention (auxiliary): windows with 0 < a < 1 dropped from the judged rate
    keep = (a_ev == 0) | (a_ev == 1)
    _, _, Ja = policy_arrays(D, np.where(keep, a_ev, np.nan))
    sh_ab = np.nanmean(Ja, 1)
    s_pt = Sm.mean(1)
    out['eval']['abstention'] = dict(fraction_abstained=float(1 - keep.mean()), mean_abs_o=float(np.abs(sh_ab - s_pt).mean()),
                                     fgre=float(S2.fgre(s_pt[None], sh_ab[None], Nm.mean(1)[None])[0]))
    e = out['eval']
    e['H5b_holds'] = bool(e['diff_mean_abs_o'] < 0 and e['ci_diff_mean_abs_o'][1] < 0 and e['fgre']['mc_discount'] < e['fgre']['deterministic'])
    e['check_p_det_vs_E2'] = int(((ev['p_det'] > 0.5).astype(float) != j_base).sum())
    print(f"H5b: q* {kq}/16; mean |o| det {pt['deterministic']:.4f} -> discount {pt['mc_discount']:.4f} (diff {e['diff_mean_abs_o']:+.4f} {e['ci_diff_mean_abs_o']}),"
          f" FGRE {e['fgre']['deterministic']:.4f} -> {e['fgre']['mc_discount']:.4f}; abstention {e['abstention']} -> {e['H5b_holds']}", flush=True)
    return out


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else L.B_BOOT
    t0 = time.time()
    out = dict(B=B, prescriptions=PRESC)
    out['H5a'] = h5a(B)
    if os.path.exists(f'{R5}/m1/mcdrop_eval.npz'):
        out['H5b'] = h5b(B)
    out['verdict'] = dict(H5a=out['H5a']['verdict']['H5a'], H5b=out.get('H5b', {}).get('eval', {}).get('H5b_holds'))
    out['time_s'] = time.time() - t0
    json.dump(out, open(f'{R5}/e5_analysis.json', 'w'), indent=1)
    print('VERDICT E5', json.dumps(out['verdict']), flush=True)
    print('E5_ANALYZE_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
