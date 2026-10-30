"""사후 분석 (post-hoc, after seeing main results): does a dedicated failure detector consolidate as the number of
distinct failure examples grows, and does that track the optimism bias? Uses main (N=100k), n20k and n300k runs.
Also: decision-boundary location r* and optimism over training steps."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results'
PS = [0.3, 0.1, 0.03, 0.01]
RUNS = ['n20k', 'main', 'n300k', 't40k']
MARK = {'n20k': '^', 'main': 'o', 'n300k': 's', 't40k': 'D'}
RLAB = {'n20k': 'N=20k', 'main': 'N=100k', 'n300k': 'N=300k', 't40k': 'N=100k, T=40k'}
C1, C2, INK, INK2 = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'


def boundary(curve_bal, curve_near, eb, en):
    xs = [0.5 * (a + b) for a, b in zip(eb[:-1], eb[1:])] + [0.5 * (a + b) for a, b in zip(en[:-1], en[1:])]
    ys = list(curve_bal) + list(curve_near)
    pts = sorted((x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y))
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 >= 0.5 > y1:
            return x0 + (y0 - 0.5) / (y0 - y1) * (x1 - x0)
    return float('nan') if pts[-1][1] < 0.5 else pts[-1][0]


rows = []
for run in RUNS:
    fm = os.path.join(RES, run, 'final_metrics.json')
    an = os.path.join(RES, run, 'analysis.json')
    if not (os.path.exists(fm) and os.path.exists(an)):
        continue
    F = json.load(open(fm))
    A = json.load(open(an))
    for mode in ('std', 'rw'):
        for p in PS:
            fr = [r for r in F['finals'] if abs(r['p'] - p) < 1e-9 and r['mode'] == mode]
            ar = [r for r in A if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == mode]
            row = dict(run=run, N=fr[0]['N'], T=fr[0]['T'], mode=mode, p=p, n_fail=np.mean([r['n_fail'] for r in fr]),
                       n_miss=np.mean([r['n_miss'] for r in fr]),
                       opt=np.mean([r['bal_opt'] for r in fr]), opt_miss=np.mean([r['bal_opt_miss'] for r in fr]),
                       r_star=np.nanmean([boundary(r['curve_bal'], r['curve_near'], F['edges_bal'], F['edges_near']) for r in fr]),
                       fail_n50=np.mean([r['concentration']['fail_n50'] for r in ar]),
                       succ_n50=np.mean([r['concentration']['succ_n50'] for r in ar]),
                       fail_L3_share=np.mean([r['layer_share']['fail'][3] for r in ar]),
                       fail_L0_share=np.mean([r['layer_share']['fail'][0] for r in ar]),
                       top_off_to_strong=np.mean([r['regime_fail']['top']['neg->strong+'] + r['regime_fail']['top']['weak+->strong+'] for r in ar]),
                       top_strong_to_off=np.mean([r['regime_fail']['top']['strong+->neg'] + r['regime_fail']['top']['strong+->weak+'] for r in ar]),
                       top_strong_strong=np.mean([r['regime_fail']['top']['strong+->strong+'] for r in ar]),
                       J_ratio=np.mean([r['overlap']['fail_succ']['J'] for r in ar]) / np.mean([r['overlap']['fail_succ']['E_layer'] for r in ar]),
                       dact_ratio=np.mean([r['overlap']['dact_fail_succ']['J'] for r in ar]) / np.mean([r['overlap']['dact_fail_succ']['E_layer'] for r in ar]),
                       weak01_D_f=np.mean([r['weak_curve'][0]['target']['acc_f'] - r['weak_curve'][4]['target']['acc_f'] for r in ar]),
                       weak01_D_s=np.mean([r['weak_curve'][0]['target']['acc_s'] - r['weak_curve'][4]['target']['acc_s'] for r in ar]))
            rows.append(row)
with open(os.path.join(RES, 'posthoc_consolidation.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(' '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()))

# optimism over training (main run, std): does the rare failure get learned late?
tc = list(csv.DictReader(open(os.path.join(RES, 'main', 'train_curves.csv'))))
dyn = []
for p in PS:
    for step in (1000, 2000, 5000, 10000):
        v = [float(r['bal_opt']) for r in tc if r['mode'] == 'std' and abs(float(r['p']) - p) < 1e-9 and int(r['step']) == step]
        dyn.append(dict(p=p, step=step, opt=float(np.mean(v))))
with open(os.path.join(RES, 'posthoc_opt_dynamics.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['p', 'step', 'opt'])
    w.writeheader()
    w.writerows(dyn)
print(dyn)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for mode, col, lab in (('std', C1, 'standard'), ('rw', C2, 'failures upsampled (rw)')):
    for run in RUNS:
        rr = [r for r in rows if r['mode'] == mode and r['run'] == run]
        if not rr:
            continue
        kw = dict(color=col, ms=5, ls='none', marker=MARK[run], label=f'{lab}, {RLAB[run]}')
        ax[0].plot([r['n_fail'] for r in rr], [r['opt'] for r in rr], **kw)
        ax[1].plot([r['n_fail'] for r in rr], [r['fail_n50'] for r in rr], **kw)
for a in ax:
    a.set_xscale('log')
    a.set_xlabel('distinct failure examples in training set')
    a.grid(True, color='#e4e3df', lw=0.6)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
ax[0].set_ylabel('optimism (balanced test)')
ax[0].set_title('(a) optimism vs failure count (N=20k/100k/300k)', fontsize=9, color=INK)
ax[1].set_yscale('log')
ax[1].set_ylabel('n50 (neurons for 50% of effect)')
ax[1].set_title('(b) consolidation of failure information', fontsize=9, color=INK)
ax[1].legend(fontsize=6.5, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(RES, 'fig4_posthoc_consolidation.png'), dpi=150)
print('posthoc done')
