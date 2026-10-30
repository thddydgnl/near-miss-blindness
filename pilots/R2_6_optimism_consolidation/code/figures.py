"""Figures for R2-6 (run on the pod after summarize.py)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from summarize import load, sel, timeseries

RES = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2 = '#0b0b0b', '#52514e'
BLUES = {0.3: '#a9c8ef', 0.03: '#2a78d6', 0.01: '#164a8c'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'lines.linewidth': 2})


def curves(run):
    F = load(run, 'final_metrics.json')
    eb, en = F['edges_bal'], F['edges_near']
    xs = np.array([0.5 * (a + b) for a, b in zip(eb[:-1], eb[1:])] + [0.5 * (a + b) for a, b in zip(en[:-1], en[1:])])
    out = {}
    for p in (0.3, 0.03, 0.01):
        rs = [r for r in F['finals'] if abs(r['p'] - p) < 1e-9 and r['mode'] == 'std']
        if rs:
            ys = np.nanmean(np.array([r['curve_bal'] + r['curve_near'] for r in rs], dtype=float), 0)
            o = np.argsort(xs)
            out[p] = (xs[o], ys[o])
    return out


def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for a, run, title in ((ax[0], 's_n100k_T1', '(a) state input'), (ax[1], 'v_T1_std', '(b) visual input (frozen DINOv2)')):
        for p, (x, y) in curves(run).items():
            a.plot(x, y, color=BLUES[p], lw=1.8, label=f'p={p}')
        a.axvspan(1.0, 1.2, color=C2, alpha=0.08, lw=0)
        a.axvspan(1.5, 2.0, color=C3, alpha=0.08, lw=0)
        a.axvline(1.0, color=INK2, lw=1, ls=':')
        a.set_xlabel('alignment ratio r = |e|/w  (r ≥ 1 fails; shaded: near / far miss)')
        a.set_title(title, color=INK, fontsize=9)
    ax[0].set_ylabel('P(pred lifted)')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig1_boundary_state_visual.png'), dpi=150)


def fig2():
    C = json.load(open(os.path.join(RES, 'criteria.json')))
    sets = [x for x in C['S2']['state'] if x] + [dict(x, setting='vis ' + x['setting']) for x in C['S2']['visual'] if x]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    x = np.arange(len(sets))
    ax[0].bar(x - 0.2, [s['opt_1x'] for s in sets], 0.38, color=C1, label='1× training')
    ax[0].bar(x + 0.2, [s['opt_4x'] for s in sets], 0.38, color=C2, label='4× training')
    ax[1].bar(x - 0.2, [s['n50_1x'] for s in sets], 0.38, color=C1, label='1×')
    ax[1].bar(x + 0.2, [s['n50_4x'] for s in sets], 0.38, color=C2, label='4×')
    for a in ax[:2]:
        a.set_xticks(x)
        a.set_xticklabels([s['setting'] for s in sets], fontsize=8)
    ax[0].set_ylabel('optimism (p=0.01, std)')
    ax[0].set_title('(a) same unique failures, 1× vs 4× training', color=INK, fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].set_yscale('log')
    ax[1].set_ylabel('n50 of failure-swap effect')
    ax[1].set_title('(b) consolidation (lower = fewer units)', color=INK, fontsize=9)
    for run, col, lab in (('s_n100k_T4', C1, 'state Nf=1k'), ('v_T4_std', C2, 'visual Nf=1k')):
        ts = timeseries(run, 0.01)
        if not ts:
            continue
        st = np.array(ts['steps']) / 1000
        n50, opt = np.array(ts['n50']), np.array(ts['opt'])
        w = st >= 5 if (st >= 5).sum() >= 3 else st >= 0
        ax[2].plot(st, opt / opt[w].max(), '-', color=col, label=f'optimism, {lab}')
        ax[2].plot(st, n50 / n50[w].max(), '--', color=col, label=f'n50, {lab}')
    ax[2].set_xlabel('training step (k), 4× run')
    ax[2].set_ylabel('normalised to max (≥5k)')
    ax[2].set_title('(c) time series: consolidation vs optimism', color=INK, fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig2_consolidation_S2.png'), dpi=150)


def fig3():
    C = json.load(open(os.path.join(RES, 'criteria.json')))
    ms = [C['S3']['primary']] + [m for m in C['S3']['secondary'] if m]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    x = np.arange(len(ms))
    ax[0].bar(x - 0.27, [m['opt_base'] for m in ms], 0.26, color=C1, label='no ablation')
    ax[0].bar(x, [m['opt_det'] for m in ms], 0.26, color=C2, label='top-n50 failure units (mean-ablated)')
    ax[0].bar(x + 0.27, [m['opt_rand'] for m in ms], 0.26, color=C3, label='same number of random units')
    ax[0].scatter(x, [m['L_low'] for m in ms], marker='_', s=300, color=INK, label='low-p 1× level', zorder=3)
    lab = lambda s: s.replace('s_n100k_', 'state ').replace('s_n300k_', 'state300k ').replace('s_n20k_', 'state20k ').replace('v_', 'vis ').replace('_std', '')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([lab(m['model']) for m in ms], fontsize=7, rotation=15)
    ax[0].set_ylabel('optimism')
    ax[0].set_title('(a) S3: ablating consolidated failure detectors', color=INK, fontsize=9)
    ax[0].legend(fontsize=7)
    s4 = [C['S4']['primary']] + [m for m in C['S4']['secondary'] if m]
    x = np.arange(len(s4))
    ax[1].bar(x - 0.27, [m['opt_std'] for m in s4], 0.26, color=C1, label='standard')
    ax[1].bar(x, [m['opt_rw'] for m in s4], 0.26, color=C2, label='uniform reweighting')
    ax[1].bar(x + 0.27, [m['opt_hn'] for m in s4], 0.26, color=C3, label='near-miss resampling')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([lab(m['run']) for m in s4], fontsize=7, rotation=15)
    ax[1].set_ylabel('optimism (p=0.01)')
    ax[1].set_title('(b) S4: remedies at equal unique-failure counts', color=INK, fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig3_ablation_remedies.png'), dpi=150)


if __name__ == '__main__':
    for f in (fig1, fig2, fig3):
        try:
            f()
        except Exception as e:  # keep going so partial results still produce the other figures
            print('figure failed:', f.__name__, repr(e))
    print('figures done')
