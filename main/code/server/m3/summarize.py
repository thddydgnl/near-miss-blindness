"""Apply the pre-registered R2-9 criteria (PLAN.md 6장) and write summary tables + figures."""
import csv
import glob
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/pmx/B/results/E0/m3')   # B E0 port
POL = ['pi1_precise', 'pi2_good', 'pi3_medium', 'pi4_sloppy', 'pi5_far_failer', 'pi6_near_failer']


def load_all():
    out = {}
    for f in glob.glob(os.path.join(RES, 'eval_*.json')):
        r = json.load(open(f))
        n = r['name']                      # p{p}_{mode}_{k}x_s{seed}
        p, mode, tm, s = n.split('_')
        out.setdefault((p, mode, tm), []).append(r)
    return out


def m(rs, f):
    v = [f(r) for r in rs]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def cond_summary(rs):
    d = dict(n=len(rs), mean_over=m(rs, lambda r: r['mean_over']), mean_abs_err=m(rs, lambda r: r['mean_abs_err']),
             r_star=m(rs, lambda r: r['r_star']), n50=m(rs, lambda r: r['n50']), n50_untrained=m(rs, lambda r: r['n50_untrained']),
             last_layer_share=m(rs, lambda r: r['swap']['layer_share'][-1]), err_succ=m(rs, lambda r: r['err_succ']),
             err_fail=m(rs, lambda r: r['err_fail']), err_succ_final=m(rs, lambda r: r['err_succ_final']))
    for p in POL:
        for k in ('true', 'imagined', 'real_frame_clf', 'over', 'near_share', 'opt_near', 'opt_far', 'pess'):
            d[f'{p}:{k}'] = m(rs, lambda r, p=p, k=k: r['policies'][p][k])
    d['over_seeds'] = [r['mean_over'] for r in rs]
    def kendall(r):
        a = [r['policies'][p]['true'] for p in POL]
        b = [r['policies'][p]['imagined'] for p in POL]
        num = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i in range(len(a)) for j in range(i + 1, len(a)))
        return num / (len(a) * (len(a) - 1) / 2)
    d['kendall_posthoc'] = m(rs, kendall)
    return d


def main():
    A = load_all()
    C = {f'{k[0]}_{k[1]}_{k[2]}': cond_summary(v) for k, v in sorted(A.items())}
    out = dict(conditions=C)
    base = C.get('p0.03_std_1x')
    # S1
    if base:
        shares = {p: base[f'{p}:near_share'] for p in POL}
        hi, lo = max(shares, key=shares.get), min(shares, key=shares.get)
        o_hi, o_lo = base[f'{hi}:over'], base[f'{lo}:over']
        ratio = o_hi / max(o_lo, 0.01)
        nm_corr = float(np.corrcoef([shares[p] for p in POL], [base[f'{p}:over'] for p in POL])[0, 1])
        out['S1'] = dict(mean_over=base['mean_over'], hi_policy=hi, lo_policy=lo, over_hi=o_hi, over_lo=o_lo, ratio_floor1pp=ratio,
                         near_share_over_corr=nm_corr, PASS=bool(base['mean_over'] >= 0.10 and ratio >= 2))
    else:
        out['S1'] = dict(PASS=False, note='missing')
    # S2
    l4 = C.get('p0.03_std_4x')
    if base and l4:
        rel = (base['mean_over'] - l4['mean_over']) / base['mean_over'] if base['mean_over'] > 0 else float('nan')
        out['S2'] = dict(over_1x=base['mean_over'], over_4x=l4['mean_over'], rel_drop=rel, n50_1x=base['n50'], n50_4x=l4['n50'],
                         r_star_1x=base['r_star'], r_star_4x=l4['r_star'], PASS=bool(rel >= 0.30 and l4['n50'] < base['n50']))
    else:
        out['S2'] = dict(PASS=False, note='missing')
    # S3
    cp, rw, nm = C.get('p0.03_cp_1x'), C.get('p0.03_rw_1x'), C.get('p0.03_nm_1x')
    if base and cp and rw:
        red = (base['mean_over'] - cp['mean_over']) / base['mean_over'] if base['mean_over'] > 0 else float('nan')
        out['S3'] = dict(over_std=base['mean_over'], over_cp=cp['mean_over'], over_rw=rw['mean_over'],
                         over_nm=nm['mean_over'] if nm else float('nan'), cp_reduction=red,
                         err_succ_std=base['err_succ'], err_succ_cp=cp['err_succ'], err_inc=cp['err_succ'] / base['err_succ'] - 1,
                         PASS=bool(red >= 0.5 and cp['mean_over'] < rw['mean_over'] and cp['err_succ'] <= 1.10 * base['err_succ']))
    else:
        out['S3'] = dict(PASS=False, note='missing')
    s1, s2, s3 = out['S1']['PASS'], out['S2']['PASS'], out['S3']['PASS']
    out['verdict'] = ('있음(처방 포함)' if s1 and s3 else '조건부(기전만)' if s1 and s2 else
                      '없음(현상만 재현)' if s1 else '없음')
    json.dump(out, open(os.path.join(RES, 'criteria.json'), 'w'), indent=1, default=float)
    rows = []
    for k, d in C.items():
        for p in POL:
            rows.append(dict(condition=k, policy=p, **{c: d[f'{p}:{c}'] for c in ('true', 'imagined', 'real_frame_clf', 'over', 'near_share',
                                                                                  'opt_near', 'opt_far', 'pess')}))
    with open(os.path.join(RES, 'summary_policies.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = ['n', 'mean_over', 'mean_abs_err', 'r_star', 'n50', 'n50_untrained', 'last_layer_share', 'err_succ', 'err_fail',
            'kendall_posthoc']
    with open(os.path.join(RES, 'summary_conditions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition'] + keys)
        for k, d in C.items():
            w.writerow([k] + [d[x] for x in keys])
    print(json.dumps({k: v for k, v in out.items() if k != 'conditions'}, indent=1, default=float))
    for k, d in C.items():
        print(k, {x: (round(d[x], 4) if isinstance(d[x], float) else d[x]) for x in keys})
    figures(C, out)


def figures(C, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
    INK = '#0b0b0b'
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e4e3df', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False})
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    lab = {'p0.3_std_1x': 'p=0.3', 'p0.03_std_1x': 'p=0.03 (1x)', 'p0.03_std_4x': 'p=0.03 (4x)', 'p0.03_rw_1x': 'reweight',
           'p0.03_cp_1x': 'contrast pairs', 'p0.03_nm_1x': 'near-miss unpaired', 'p0.03_rw_4x': 'reweight (4x)',
           'p0.03_cp_4x': 'contrast pairs (4x)'}
    cols = {'p0.3_std_1x': '#a9c8ef', 'p0.03_std_1x': C1, 'p0.03_std_4x': C4, 'p0.03_rw_1x': C2, 'p0.03_cp_1x': C3, 'p0.03_nm_1x': C5,
            'p0.03_rw_4x': '#9c3d16', 'p0.03_cp_4x': '#0d6b49'}
    conds = [c for c in lab if c in C]
    x = np.arange(len(POL))
    wd = 0.8 / max(len(conds), 1)
    for i, c in enumerate(conds):
        ax[0].bar(x + (i - (len(conds) - 1) / 2) * wd, [C[c][f'{p}:over'] for p in POL], wd * 0.95, color=cols[c], label=lab[c])
    ax[0].axhline(0, color=INK, lw=0.6)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([p.split('_', 1)[1] for p in POL], fontsize=7, rotation=15)
    ax[0].set_ylabel('imagined − true success rate')
    ax[0].set_title('(a) policy-evaluation overestimation', fontsize=9, color=INK)
    ax[0].legend(fontsize=7)
    b = C.get('p0.03_std_1x')
    if b:
        for c in conds:
            ax[1].scatter([C[c][f'{p}:near_share'] for p in POL], [C[c][f'{p}:over'] for p in POL], color=cols[c], s=22, label=lab[c])
        ax[1].set_xlabel("policy's near-miss share (r in [1, 1.2))")
        ax[1].set_ylabel('overestimation')
        ax[1].set_title('(b) overestimation vs near-miss share (colours as in a)', fontsize=9, color=INK)
    for c in conds:
        rs = C[c]
    ax[2].bar(np.arange(len(conds)), [C[c]['mean_over'] for c in conds], color=[cols[c] for c in conds])
    ax[2].set_xticks(np.arange(len(conds)))
    ax[2].set_xticklabels([lab[c] for c in conds], fontsize=7, rotation=20)
    ax[2].set_ylabel('mean overestimation (6 policies)')
    for i, c in enumerate(conds):
        ax[2].text(i, C[c]['mean_over'], f"n50={C[c]['n50']:.0f}", ha='center', va='bottom', fontsize=7, color=INK)
    ax[2].set_title('(c) conditions (label: predictor n50)', fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig1_policy_eval.png'), dpi=150)


if __name__ == '__main__':
    main()
