"""Apply the pre-registered criteria (PLAN.md 6장) and write summary tables.
Reads <res>/main/{final_metrics.json, analysis.json, finetune.json} (+ optional n20k/n300k final metrics)."""
import csv
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results'
PS = [0.3, 0.1, 0.03, 0.01]
LOWP = 0.01


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float('nan')


def sel(rows, p, mode='std'):
    return [r for r in rows if abs(r['p'] - p) < 1e-9 and r['mode'] == mode]


def main():
    out = {}
    fm = json.load(open(os.path.join(RES, 'main', 'final_metrics.json')))['finals']
    # ------------------------------------------------------------------ S1
    tab1 = []
    for N_name in ('main', 'n20k', 'n300k'):
        path = os.path.join(RES, N_name, 'final_metrics.json')
        if not os.path.exists(path):
            continue
        F = json.load(open(path))['finals']
        for mode in ('std', 'rw'):
            for p in PS:
                rs = sel(F, p, mode)
                if not rs:
                    continue
                row = dict(run=N_name, N=rs[0]['N'], mode=mode, p=p, n_seeds=len(rs))
                for k in ('bal_opt', 'bal_opt_miss', 'bal_opt_slip', 'bal_opt_cont', 'bal_acc_s', 'bal_pes_cont',
                          'bal_mse_s', 'bal_mse_s_obj', 'near_opt', 'near_acc_s', 'n_fail', 'n_miss', 'n_slip'):
                    vals = [r[k] for r in rs if k in r]
                    row[k] = mean(vals)
                    if k in ('bal_opt', 'bal_mse_s'):
                        row[k + '_min'] = float(np.min(vals))
                        row[k + '_max'] = float(np.max(vals))
                tab1.append(row)
    with open(os.path.join(RES, 'summary_S1.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tab1[0].keys()))
        w.writeheader()
        w.writerows(tab1)
    g = {(r['run'], r['mode'], r['p']): r for r in tab1}
    o_low, o_hi = g[('main', 'std', LOWP)]['bal_opt'], g[('main', 'std', 0.3)]['bal_opt']
    e_low, e_hi = g[('main', 'std', LOWP)]['bal_mse_s'], g[('main', 'std', 0.3)]['bal_mse_s']
    ratio = o_low / max(o_hi, 0.01)
    rel = abs(e_low - e_hi) / e_hi
    o_rw = g[('main', 'rw', LOWP)]['bal_opt']
    excess = o_low - o_hi
    out['S1'] = dict(opt_p001=o_low, opt_p03=o_hi, ratio_floor1pct=ratio, ratio_raw=o_low / max(o_hi, 1e-9),
                     mse_s_p001=e_low, mse_s_p03=e_hi, mse_rel_diff=rel,
                     pass_ratio=ratio >= 3, pass_mse=rel <= 0.2, PASS=bool(ratio >= 3 and rel <= 0.2),
                     rw_opt_p001=o_rw, rw_fraction_of_excess_removed=(o_low - o_rw) / excess if excess > 0 else float('nan'),
                     rw_acc_s_p001=g[('main', 'rw', LOWP)]['bal_acc_s'], std_acc_s_p001=g[('main', 'std', LOWP)]['bal_acc_s'])
    # ------------------------------------------------------------------ S2
    A = json.load(open(os.path.join(RES, 'main', 'analysis.json')))

    def curves(p, mode='std'):
        rs = [r for r in A if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == mode]
        alphas = [c['alpha'] for c in rs[0]['weak_curve']]
        res = {}
        for src in ('target', 'random'):
            acc_s = np.array([[c[src]['acc_s'] for c in r['weak_curve']] for r in rs]).mean(0)
            acc_f = np.array([[c[src]['acc_f'] for c in r['weak_curve']] for r in rs]).mean(0)
            res[src] = dict(D_s=(acc_s[0] - acc_s).tolist(), D_f=(acc_f[0] - acc_f).tolist(),
                            acc_s=acc_s.tolist(), acc_f=acc_f.tolist())
        res['frac_entries'] = np.array([[c['frac_entries'] for c in r['weak_curve']] for r in rs]).mean(0).tolist()
        res['frac_mass'] = np.array([[c['frac_mass'] for c in r['weak_curve']] for r in rs]).mean(0).tolist()
        return alphas, res

    s2 = {}
    tab2 = []
    for mode in ('std', 'rw'):
        for p in PS:
            alphas, cv = curves(p, mode)
            s2[(mode, p)] = cv
            for i, a in enumerate(alphas):
                tab2.append(dict(mode=mode, p=p, alpha=a, frac_entries=cv['frac_entries'][i], frac_mass=cv['frac_mass'][i],
                                 weak_acc_s=cv['target']['acc_s'][i], weak_acc_f=cv['target']['acc_f'][i],
                                 weak_D_s=cv['target']['D_s'][i], weak_D_f=cv['target']['D_f'][i],
                                 rand_acc_s=cv['random']['acc_s'][i], rand_acc_f=cv['random']['acc_f'][i],
                                 rand_D_s=cv['random']['D_s'][i], rand_D_f=cv['random']['D_f'][i]))
    with open(os.path.join(RES, 'summary_S2_weak_ablation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tab2[0].keys()))
        w.writeheader()
        w.writerows(tab2)
    for lowp in (0.01, 0.03):
        lo, hi = s2[('std', lowp)]['target'], s2[('std', 0.3)]['target']
        adm = [i for i, a in enumerate(alphas) if a > 0 and lo['D_s'][i] <= 0.02]
        if adm:
            i_star = max(adm, key=lambda i: lo['D_f'][i])
            sel_lo = lo['D_f'][i_star] - lo['D_s'][i_star]
            sel_hi = hi['D_f'][i_star] - hi['D_s'][i_star]
            rl = s2[('std', lowp)]['random']
            sel_rand = rl['D_f'][i_star] - rl['D_s'][i_star]
            d = dict(alpha_star=alphas[i_star], D_f=lo['D_f'][i_star], D_s=lo['D_s'][i_star], sel_lowp=sel_lo,
                     sel_p03=sel_hi, D_f_p03=hi['D_f'][i_star], D_s_p03=hi['D_s'][i_star],
                     rand_D_f=rl['D_f'][i_star], rand_D_s=rl['D_s'][i_star], sel_random=sel_rand,
                     pass_a=lo['D_f'][i_star] >= 0.20, pass_b=sel_lo > sel_hi)
            d['PASS'] = bool(d['pass_a'] and d['pass_b'])
            d['aux_weak_more_selective_than_random'] = bool(sel_lo > sel_rand)
        else:
            d = dict(alpha_star=None, PASS=False, note='no admissible alpha')
        out[f'S2_p{lowp}'] = d
    out['S2'] = out['S2_p0.01']
    # regime ablations (seed means)
    reg = {}
    for mode in ('std', 'rw'):
        for p in PS:
            rs = [r for r in A if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == mode]
            base_s = mean([r['weak_curve'][0]['target']['acc_s'] for r in rs])
            base_f = mean([r['weak_curve'][0]['target']['acc_f'] for r in rs])
            for k in rs[0]['regime_ablation']:
                reg[f'{mode}_p{p}_{k}'] = dict(
                    frac_entries=mean([r['regime_ablation'][k]['frac_entries'] for r in rs]),
                    frac_mass=mean([r['regime_ablation'][k]['frac_mass'] for r in rs]),
                    D_s=base_s - mean([r['regime_ablation'][k]['target']['acc_s'] for r in rs]),
                    D_f=base_f - mean([r['regime_ablation'][k]['target']['acc_f'] for r in rs]),
                    rand_D_s=base_s - mean([r['regime_ablation'][k]['random']['acc_s'] for r in rs]),
                    rand_D_f=base_f - mean([r['regime_ablation'][k]['random']['acc_f'] for r in rs]))
    out['regime_ablation'] = reg
    # ------------------------------------------------------------------ S3
    tab3 = []
    keys = ['fail_succ', 'fail_succzero', 'fail_unrel', 'succ_unrel', 'failspec_succ', 'fail_copy', 'slip_succ',
            'slip_fail', 'slip_unrel', 'rot_fail_succ', 'untrained_fail_succ', 'dact_fail_succ', 'dact_slip_fail',
            'untrained_dact_fail_succ']
    for mode in ('std', 'rw'):
        for p in PS:
            rs = [r for r in A if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == mode]
            row = dict(mode=mode, p=p)
            for k in keys:
                J = mean([r['overlap'][k]['J'] for r in rs])
                E = mean([r['overlap'][k]['E_layer'] for r in rs])
                row[f'{k}_J'] = J
                row[f'{k}_E'] = E
                row[f'{k}_ratio'] = J / E
                row[f'{k}_ratio_unif'] = J / mean([r['overlap'][k]['E_unif'] for r in rs])
            for k in ('fail_n50', 'succ_n50', 'slip_n50', 'unrel_n50', 'fail_rot_n50'):
                row[k] = mean([r['concentration'][k] for r in rs])
            for nme in ('fail', 'succ', 'slip'):
                ls = np.array([r['layer_share'][nme] for r in rs]).mean(0)
                for l in range(len(ls)):
                    row[f'layershare_{nme}_L{l}'] = float(ls[l])
            for rk in ('regime_fail', 'regime_slip'):
                for t in rs[0][rk]['top']:
                    row[f'{rk}_top_{t}'] = mean([r[rk]['top'][t] for r in rs])
                    row[f'{rk}_all_{t}'] = mean([r[rk]['all'][t] for r in rs])
                row[f'{rk}_top_mean_u_succ'] = mean([r[rk]['top_mean_u_succ'] for r in rs])
                row[f'{rk}_top_mean_u_fail'] = mean([r[rk]['top_mean_u_fail'] for r in rs])
                row[f'{rk}_top_frac_pairs_regime_change'] = mean([r[rk]['top_frac_pairs_regime_change'] for r in rs])
            row['shared_regime_change'] = mean([r.get('shared_regime_change', float('nan')) for r in rs])
            row['shared_n'] = mean([r.get('shared_n', 0) for r in rs])
            row['copy_over_miss'] = mean([r['contrast']['copy_over_miss'] for r in rs])
            tab3.append(row)
    with open(os.path.join(RES, 'summary_S3_overlap.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(tab3[0].keys()))
        w.writeheader()
        w.writerows(tab3)
    g3 = {(r['mode'], r['p']): r for r in tab3}
    FT = json.load(open(os.path.join(RES, 'main', 'finetune.json')))
    tab4 = []
    for p in PS:
        rs = [r for r in FT if abs(r['cfg']['p'] - p) < 1e-9]
        for vn in ('full_sgd', 'ctrl_sgd', 'last1_sgd', 'last2_sgd', 'full_adam', 'ctrl_adam'):
            vs = [r[vn] for r in rs if vn in r]
            if not vs:
                continue
            row = dict(p=p, variant=vn, n=len(vs), lr_sgd=mean([r.get('lr_sgd') for r in rs]),
                       base_acc_miss=mean([r['base']['acc_miss'] for r in rs]),
                       base_acc_s=mean([r['base']['acc_s'] for r in rs]),
                       after_acc_miss=mean([v['after']['acc_miss'] for v in vs]),
                       after_acc_slip=mean([v['after']['acc_slip'] for v in vs]),
                       after_acc_s=mean([v['after']['acc_s'] for v in vs]))
            for key in ('w_sq_share', 'w_rel_share', 'act_share', 'grad0_share'):
                if key in vs[0]:
                    arr = np.array([v[key] for v in vs]).mean(0)
                    for l in range(len(arr)):
                        row[f'{key}_L{l}'] = float(arr[l])
            if 'transplant' in vs[0]:
                for t in vs[0]['transplant']:
                    row[f'transplant_{t}_recovery'] = mean([v['transplant'][t]['recovery'] for v in vs])
                    row[f'transplant_{t}_acc_s'] = mean([v['transplant'][t]['acc_s'] for v in vs])
            tab4.append(row)
    keys4 = []
    for r in tab4:
        for k in r:
            if k not in keys4:
                keys4.append(k)
    with open(os.path.join(RES, 'summary_finetune.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys4)
        w.writeheader()
        w.writerows(tab4)
    g4 = {(r['p'], r['variant']): r for r in tab4}
    fsg = g4.get((LOWP, 'full_sgd'), {})
    last1 = fsg.get('w_sq_share_L4', float('nan'))
    last2 = last1 + fsg.get('w_sq_share_L3', float('nan'))
    r3 = g3[('std', LOWP)]
    out['S3'] = dict(J_fail_succ=r3['fail_succ_J'], E_layer=r3['fail_succ_E'], ratio_layer=r3['fail_succ_ratio'],
                     ratio_unif=r3['fail_succ_ratio_unif'], pass_a=r3['fail_succ_ratio'] >= 3,
                     ft_share_last_layer=last1, ft_share_last_third=last2,
                     pass_b=bool((last1 >= 0.5) or (last2 >= 0.5)),
                     aux=dict(fail_unrel_ratio=r3['fail_unrel_ratio'], rot_ratio=r3['rot_fail_succ_ratio'],
                              untrained_ratio=r3['untrained_fail_succ_ratio'], failspec_succ_ratio=r3['failspec_succ_ratio'],
                              J_untrained=r3['untrained_fail_succ_J'], J_p03=g3[('std', 0.3)]['fail_succ_J'],
                              dact_ratio=r3['dact_fail_succ_ratio'], dact_untrained_ratio=r3['untrained_dact_fail_succ_ratio'],
                              dact_J=r3['dact_fail_succ_J'], dact_J_untrained=r3['untrained_dact_fail_succ_J'],
                              dact_J_p03=g3[('std', 0.3)]['dact_fail_succ_J'],
                              fail_succzero_ratio=r3['fail_succzero_ratio'],
                              ctrl_sgd_last_layer=g4.get((LOWP, 'ctrl_sgd'), {}).get('w_sq_share_L4', float('nan')),
                              ctrl_sgd_last_third=g4.get((LOWP, 'ctrl_sgd'), {}).get('w_sq_share_L4', float('nan'))
                              + g4.get((LOWP, 'ctrl_sgd'), {}).get('w_sq_share_L3', float('nan')),
                              full_adam_last_third=g4.get((LOWP, 'full_adam'), {}).get('w_sq_share_L4', float('nan'))
                              + g4.get((LOWP, 'full_adam'), {}).get('w_sq_share_L3', float('nan')),
                              transplant_L3L4_recovery=fsg.get('transplant_L3L4_recovery', float('nan')),
                              transplant_L4_recovery=fsg.get('transplant_L4_recovery', float('nan'))))
    out['S3']['PASS'] = bool(out['S3']['pass_a'] and out['S3']['pass_b'])
    s1, s2p, s3p = out['S1']['PASS'], out['S2']['PASS'], out['S3']['PASS']
    out['verdict'] = ('있음' if s1 and (s2p or s3p) else '조건부 있음(클래스 불균형과 구분 안 됨)' if s1 else '없음')
    json.dump(out, open(os.path.join(RES, 'criteria.json'), 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
