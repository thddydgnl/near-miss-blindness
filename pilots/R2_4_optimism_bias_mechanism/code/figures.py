"""Three small figures for results/ (run on the pod)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else '/workspace/results'
RUN = sys.argv[2] if len(sys.argv) > 2 else 'main'
PS = [0.3, 0.1, 0.03, 0.01]
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUES = ['#a9c8ef', '#6fa3e3', '#2a78d6', '#164a8c']   # light -> dark for p = 0.3 .. 0.01

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'lines.linewidth': 2})


def load(name):
    return json.load(open(os.path.join(RES, RUN, name)))


def fig1():
    F = load('final_metrics.json')
    fin, eb, en = F['finals'], F['edges_bal'], F['edges_near']
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for mode, col, lab in (('std', C1, 'standard'), ('rw', C2, 'failures upsampled (rw)')):
        ms, ss = [], []
        for p in PS:
            rs = [r for r in fin if abs(r['p'] - p) < 1e-9 and r['mode'] == mode]
            v = [r['bal_opt'] for r in rs]
            e = [r['bal_mse_s'] for r in rs]
            ax[0].scatter([p] * len(v), v, s=14, color=col, alpha=0.45, linewidths=0)
            ax[1].scatter([p] * len(e), e, s=14, color=col, alpha=0.45, linewidths=0)
            ms.append(np.mean(v))
            ss.append(np.mean(e))
        ax[0].plot(PS, ms, '-o', color=col, ms=5, label=lab)
        ax[1].plot(PS, ss, '-o', color=col, ms=5, label=lab)
    for a in ax[:2]:
        a.set_xscale('log')
        a.set_xticks(PS)
        a.set_xticklabels([str(p) for p in PS])
        a.set_xlabel('training failure rate p')
    ax[0].set_ylabel('optimism  P(pred lifted | true failure)')
    ax[0].set_title('(a) optimism bias, balanced test', color=INK, fontsize=9)
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel('success-case MSE (all outputs)')
    ax[1].set_yscale('log')
    ax[1].set_title('(b) success-case error', color=INK, fontsize=9)
    cb = [0.5 * (a + b) for a, b in zip(eb[:-1], eb[1:])]
    cn = [0.5 * (a + b) for a, b in zip(en[:-1], en[1:])]
    for p, col in zip(PS, BLUES):
        rs = [r for r in fin if abs(r['p'] - p) < 1e-9 and r['mode'] == 'std']
        yb = np.nanmean(np.array([r['curve_bal'] for r in rs], dtype=float), 0)
        yn = np.nanmean(np.array([r['curve_near'] for r in rs], dtype=float), 0)
        xs = np.array(cb + cn)
        ys = np.concatenate([yb, yn])
        o = np.argsort(xs)
        ax[2].plot(xs[o], ys[o], color=col, lw=1.6, label=f'p={p}')
    ax[2].axvline(1.0, color=INK2, lw=1, ls=':')
    ax[2].set_xlabel('alignment ratio r = |e| / w   (r ≥ 1 fails)')
    ax[2].set_ylabel('P(pred lifted)')
    ax[2].set_title('(c) predicted lift vs alignment (std)', color=INK, fontsize=9)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig1_optimism_vs_p.png'), dpi=150)


def fig2():
    A = load('analysis.json')
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for a, p in zip(ax, (0.01, 0.03, 0.3)):
        rs = [r for r in A if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == 'std']
        al = [c['alpha'] for c in rs[0]['weak_curve']]
        for src, ls, tag in (('target', '-', 'weak ablation'), ('random', '--', 'count-matched random')):
            s = np.array([[c[src]['acc_s'] for c in r['weak_curve']] for r in rs]).mean(0)
            f = np.array([[c[src]['acc_f'] for c in r['weak_curve']] for r in rs]).mean(0)
            a.plot(al, 100 * (f[0] - f), ls, color=C1, label=f'failure-acc drop, {tag}')
            a.plot(al, 100 * (s[0] - s), ls, color=C2, label=f'success-acc drop, {tag}')
        a.axhline(20, color=INK2, lw=0.8, ls=':')
        a.axhline(2, color=INK2, lw=0.8, ls=':')
        a.set_xlabel('α  (zero |a| < α·m_j)')
        a.set_title(f'p = {p} (std, 3 seeds)', color=INK, fontsize=9)
    ax[0].set_ylabel('accuracy drop (%p)')
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig2_weak_ablation.png'), dpi=150)


def fig3():
    A = load('analysis.json')
    FT = load('finetune.json')
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))
    x = np.arange(len(PS))
    wdt = 0.2
    for i, (k, col, lab) in enumerate((('fail_succ', C1, 'failure vs success'), ('fail_unrel', C2, 'failure vs unrelated'),
                                       ('rot_fail_succ', C3, 'rotated basis'), ('untrained_fail_succ', C4, 'untrained'))):
        vals = []
        for p in PS:
            rs = [r for r in A if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == 'std']
            vals.append(np.mean([r['overlap'][k]['J'] for r in rs]) / np.mean([r['overlap'][k]['E_layer'] for r in rs]))
        ax[0].bar(x + (i - 1.5) * wdt, vals, wdt * 0.9, color=col, label=lab)
    ax[0].axhline(3, color=INK2, lw=0.8, ls=':')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([str(p) for p in PS])
    ax[0].set_xlabel('training failure rate p (std)')
    ax[0].set_ylabel('Jaccard / layer-matched random')
    ax[0].set_title('(a) top-10% neuron overlap', color=INK, fontsize=9)
    ax[0].legend(fontsize=7, loc='upper center', ncol=2)
    ax[0].set_ylim(0, 8)
    groups = (('success-strong → off in failure', ('strong+->neg', 'strong+->weak+'), C1),
              ('off → strong in failure', ('neg->strong+', 'weak+->strong+'), C2),
              ('strong → strong (graded)', ('strong+->strong+',), C3),
              ('weak/neg both', ('neg->neg', 'neg->weak+', 'weak+->neg', 'weak+->weak+'), C4))
    bottom = np.zeros(len(PS))
    for lab, keys, col in groups:
        vals = []
        for p in PS:
            rs = [r for r in A if abs(r['cfg']['p'] - p) < 1e-9 and r['cfg']['mode'] == 'std']
            vals.append(np.mean([sum(r['regime_fail']['top'][k] for k in keys) for r in rs]))
        ax[1].bar(x, vals, 0.6, bottom=bottom, color=col, label=lab, edgecolor='#fcfcfb', linewidth=1)
        bottom += np.array(vals)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(p) for p in PS])
    ax[1].set_xlabel('training failure rate p (std)')
    ax[1].set_ylabel('share of failure-swap effect (top-10%)')
    ax[1].set_title('(b) regime: success run → failure run', color=INK, fontsize=9)
    ax[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[1].set_ylim(0, 1)
    lx = np.arange(5)
    for j, (vn, col, lab) in enumerate((('full_sgd', C1, '20 failures + replay'), ('ctrl_sgd', C2, '20 successes + replay (control)'))):
        rs = [r for r in FT if abs(r['cfg']['p'] - 0.01) < 1e-9 and vn in r]
        if not rs:
            continue
        v = np.array([r[vn]['w_sq_share'] for r in rs]).mean(0)
        ax[2].bar(lx + (j - 0.5) * 0.38, v, 0.36, color=col, label=lab)
    ax[2].set_xticks(lx)
    ax[2].set_xticklabels(['W0', 'W1', 'W2', 'W3', 'W4 (out)'])
    ax[2].set_ylabel('share of ‖ΔW‖² (SGD fine-tune)')
    ax[2].set_title('(c) fine-tune weight change, p=0.01', color=INK, fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig3_overlap_regime_finetune.png'), dpi=150)


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    print('figures done')
