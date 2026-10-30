"""Apply the pre-registered R2-6 criteria (PLAN.md 6장) and write summary tables."""
import csv
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results'
STATE_SETTINGS = [('Nf200', 's_n20k_T1', 's_n20k_T4'), ('Nf1k', 's_n100k_T1', 's_n100k_T4'), ('Nf3k', 's_n300k_T1', 's_n300k_T4')]
VIS_SETTINGS = [('Nf1k', 'v_T1_std', 'v_T4_std')]


def load(run, f='analysis.json'):
    p = os.path.join(RES, run, f)
    return json.load(open(p)) if os.path.exists(p) else None


def sel(rows, p, mode='std'):
    return [r for r in rows if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == mode]


def mean(rows, key, sub=None):
    vals = [(r[key][sub] if sub else r[key]) for r in rows]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def t_half(steps, q):
    """first step (after the maximum, steps >= 5k) where q <= q_end + 0.5 (q_max - q_end); inf if q never decreases."""
    pts = [(s, v) for s, v in zip(steps, q) if s >= 5000]
    if len(pts) < 3:
        pts = list(zip(steps, q))
    ss, vv = [s for s, _ in pts], [v for _, v in pts]
    i_max = int(np.argmax(vv))
    q_max, q_end = vv[i_max], vv[-1]
    if q_max <= q_end:
        return float('inf')
    thr = q_end + 0.5 * (q_max - q_end)
    for s, v in zip(ss[i_max:], vv[i_max:]):
        if v <= thr:
            return s
    return float('inf')


def timeseries(run, p=0.01):
    ts = load(run, 'timeseries.json')
    if ts is None:
        return None
    ts = [r for r in ts if abs(r['p'] - p) < 1e-9 and r['mode'] == 'std']
    if not ts:
        return None
    steps = sorted(set(r['step'] for r in ts))
    n50 = [float(np.mean([r['n50'] for r in ts if r['step'] == s])) for s in steps]
    opt = [float(np.mean([r['opt'] for r in ts if r['step'] == s])) for s in steps]
    l3 = [float(np.mean([r['layer_share'][3] for r in ts if r['step'] == s])) for s in steps]
    return dict(steps=steps, n50=n50, opt=opt, l3=l3, t_half_n50=t_half(steps, n50), t_half_opt=t_half(steps, opt))


def s1_version(run):
    A = load(run)
    if A is None or not sel(A, 0.01) or not sel(A, 0.3):
        return dict(PASS=False, note='missing')
    lo, hi = sel(A, 0.01), sel(A, 0.3)
    near, far = mean(lo, 'opt_near'), mean(lo, 'opt_far')
    rs_lo, rs_hi = mean(lo, 'r_star'), mean(hi, 'r_star')
    ratio = near / max(far, 0.01)
    d = dict(opt_near=near, opt_far=far, near_far_ratio=ratio, r_star_p001=rs_lo, r_star_p03=rs_hi, r_shift=rs_lo - rs_hi,
             opt_p001=mean(lo, 'opt'), opt_p03=mean(hi, 'opt'), opt_slip_p001=mean(lo, 'opt_slip'),
             mse_s_p001=mean(lo, 'mse_s'), mse_s_p03=mean(hi, 'mse_s'))
    d['PASS'] = bool(ratio >= 3 and d['r_shift'] >= 0.2)
    d['H3_slip_le_third_of_near'] = bool(d['opt_slip_p001'] <= near / 3)
    return d


def s2_setting(name, r1, r4, p=0.01):
    A1, A4 = load(r1), load(r4)
    if A1 is None or A4 is None:
        return None
    a1, a4 = sel(A1, p), sel(A4, p)
    if not a1 or not a4:
        return None
    o1, o4 = mean(a1, 'opt'), mean(a4, 'opt')
    n1, n4 = mean(a1, 'n50'), mean(a4, 'n50')
    d = dict(setting=name, opt_1x=o1, opt_4x=o4, rel_drop=(o1 - o4) / o1 if o1 > 0 else float('nan'), n50_1x=n1, n50_4x=n4,
             l3_1x=float(np.mean([r['layer_share'][3] for r in a1])), l3_4x=float(np.mean([r['layer_share'][3] for r in a4])),
             n50_dact_1x=mean(a1, 'n50_dact'), n50_dact_4x=mean(a4, 'n50_dact'),
             n50_untrained=float(np.mean([r['untrained']['n50'] for r in a1])))
    ts = timeseries(r4, p)
    d['pair_ok'] = bool(n4 < n1 and d['rel_drop'] >= 0.30)
    if ts:
        d['t_half_n50'], d['t_half_opt'] = ts['t_half_n50'], ts['t_half_opt']
        d['ts_ok'] = bool(ts['t_half_n50'] <= ts['t_half_opt'])
    else:
        d['ts_ok'] = False
    d['PASS'] = bool(d['pair_ok'] and d['ts_ok'])
    return d


def s3_model(run_cons, run_low, p_cons):
    A, L = load(run_cons), load(run_low)
    if A is None or L is None:
        return None
    a = sel(A, p_cons)
    if not a or not sel(L, 0.01):
        return None
    L_low = mean(sel(L, 0.01), 'opt')
    base = mean(a, 'opt')
    det = float(np.mean([r['ablation']['det_mean']['opt'] for r in a]))
    rnd = float(np.mean([r['ablation']['rand_mean']['opt'] for r in a]))
    detz = float(np.mean([r['ablation']['det_zero']['opt'] for r in a]))
    rndz = float(np.mean([r['ablation']['rand_zero']['opt'] for r in a]))
    gap = L_low - base
    d = dict(model=f'{run_cons} p={p_cons}', L_low=L_low, opt_base=base, opt_det=det, opt_rand=rnd, n_det=mean(a, 'n50'),
             restore_frac=(det - base) / gap if gap > 0 else float('nan'), rand_over_det=(rnd - base) / (det - base) if det > base else float('nan'),
             opt_det_zero=detz, opt_rand_zero=rndz)
    d['PASS'] = bool(gap > 0 and d['restore_frac'] >= 0.5 and (rnd - base) <= (det - base) / 3)
    return d


def s4_setting(run, p=0.01):
    A = load(run)
    if A is None:
        return None
    s, rw, hn = sel(A, p, 'std'), sel(A, p, 'rw'), sel(A, p, 'hn')
    if not rw or not hn:
        return None
    o_rw, o_hn = mean(rw, 'opt'), mean(hn, 'opt')
    e_rw, e_hn, e_std = mean(rw, 'mse_s'), mean(hn, 'mse_s'), mean(s, 'mse_s')
    d = dict(run=run, opt_std=mean(s, 'opt'), opt_rw=o_rw, opt_hn=o_hn, rel_diff=(o_rw - o_hn) / o_rw if o_rw > 0 else float('nan'),
             near_std=mean(s, 'opt_near'), near_rw=mean(rw, 'opt_near'), near_hn=mean(hn, 'opt_near'),
             mse_s_std=e_std, mse_s_rw=e_rw, mse_s_hn=e_hn, mse_inc_vs_rw=e_hn / e_rw - 1, mse_inc_vs_std=e_hn / e_std - 1,
             acc_s_rw=mean(rw, 'acc_s'), acc_s_hn=mean(hn, 'acc_s'))
    d['PASS'] = bool(d['rel_diff'] >= 0.30 and d['mse_inc_vs_rw'] <= 0.10)
    return d


def main():
    out = {}
    out['S1'] = dict(state=s1_version('s_n100k_T1'), visual=s1_version('v_T1_std'))
    out['S1']['PASS'] = bool(out['S1']['state']['PASS'] and out['S1']['visual']['PASS'])
    st = [s2_setting(*x) for x in STATE_SETTINGS]
    vi = [s2_setting(*x) for x in VIS_SETTINGS]
    st2 = [s2_setting('p0.03_Nf200', 's_p03_n6k_T1', 's_p03_n6k_T4', 0.03), s2_setting('p0.03_Nf1k', 's_p03_n33k_T1', 's_p03_n33k_T4', 0.03),
           s2_setting('p0.03_Nf3k', 's_n100k_T1', 's_n100k_T4', 0.03), s2_setting('p0.3_Nf30k', 's_n100k_T1', 's_n100k_T4', 0.3),
           s2_setting('vis_p0.03_Nf3k', 'v_T1_std', 'v_T4_std', 0.03)]
    out['S2'] = dict(state=st, visual=vi, secondary=[x for x in st2 if x],
                     n_state_pass=sum(1 for x in st if x and x['PASS']), n_visual_pass=sum(1 for x in vi if x and x['PASS']))
    out['S2']['PASS'] = bool(out['S2']['n_state_pass'] >= 2 and out['S2']['n_visual_pass'] >= 1)
    out['S3'] = dict(primary=s3_model('s_n100k_T4', 's_n100k_T1', 0.01),
                     secondary=[s3_model('s_n100k_T1', 's_n100k_T1', 0.3), s3_model('v_T4_std', 'v_T1_std', 0.01),
                                s3_model('v_T1_std', 'v_T1_std', 0.3), s3_model('s_n300k_T4', 's_n300k_T1', 0.01),
                                s3_model('s_n20k_T4', 's_n20k_T1', 0.01)])
    out['S3']['PASS'] = bool(out['S3']['primary'] and out['S3']['primary']['PASS'])
    out['S4'] = dict(primary=s4_setting('s_n100k_T1'),
                     secondary=[s4_setting('s_n20k_T1'), s4_setting('s_n300k_T1'), s4_setting('s_n100k_T4'),
                                s4_setting('s_n20k_T4'), s4_setting('s_n300k_T4'), s4_setting('v_T1_rem')])
    out['S4']['PASS'] = bool(out['S4']['primary'] and out['S4']['primary']['PASS'])
    s1, s2, s3 = out['S1']['PASS'], out['S2']['PASS'], out['S3']['PASS']
    out['verdict'] = '있음' if (s1 and s2 and s3) else '조건부 있음' if (s1 and (s2 or s3)) else '없음'
    # flat table of all models
    rows = []
    for run in sorted(os.listdir(RES)):
        A = load(run)
        if not A:
            continue
        for r in A:
            c = r['cfg']
            rows.append(dict(run=run, version=c['version'], N=c['N'], T=c['T'], p=c['p'], mode=c['mode'], seed=c['seed'],
                             opt=r['opt'], opt_near=r.get('opt_near'), opt_far=r.get('opt_far'), opt_slip=r.get('opt_slip'),
                             acc_s=r['acc_s'], mse_s=r['mse_s'], r_star=r['r_star'], n50=r['n50'], n50_dact=r['n50_dact'],
                             n50_untrained=r['untrained']['n50'], l3=r['layer_share'][3], l0=r['layer_share'][0],
                             succ_n50=r['succ_n50'], ov_dact=r['overlap_dact']['ratio'], ov_dact_untrained=r['untrained']['overlap_dact']['ratio'],
                             ov_gain=r['overlap_gain']['ratio'], ov_gain_untrained=r['untrained']['overlap_gain']['ratio'],
                             abl_det=r['ablation']['det_mean']['opt'], abl_rand=r['ablation']['rand_mean']['opt'],
                             abl_det_zero=r['ablation']['det_zero']['opt'], abl_rand_zero=r['ablation']['rand_zero']['opt'],
                             off_to_strong=r['regime_top']['neg->strong+'] + r['regime_top']['weak+->strong+'],
                             strong_strong=r['regime_top']['strong+->strong+']))
    with open(os.path.join(RES, 'summary_models.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(out, open(os.path.join(RES, 'criteria.json'), 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
