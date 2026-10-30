"""B E1 figures (CPU analysis env): M1 curves (val), M1 AUROC vs horizon, M3 curves by p.
usage: python e1_figs.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

R1 = os.path.expanduser('~/pmx/B/results/E1/m1')
R3 = os.path.expanduser('~/pmx/B/results/E1/m3')
SURF, TXT, TXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': TXT2, 'axes.labelcolor': TXT, 'xtick.color': TXT2,
                     'ytick.color': TXT2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def style(ax, xlog=True, ticks=(0.2, 0.5, 1, 2, 3, 6)):
    if xlog:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def m1_curves(A, st='val'):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4), dpi=150, sharey=True)
    for i, d in enumerate(('r', 'r_obj')):
        for k, H in enumerate((3, 5, 7, 10)):
            ax = axes[i, k]
            c = A['cells'].get(f'{st}/H{H}/{d}')
            ax.axvspan(1, 2, color='#f0efec', zorder=0)
            ax.axvline(1, color=TXT2, lw=0.8, ls=':')
            if c:
                cu = c['a']['curve']
                x = [b['r_med'] for b in cu]
                ax.plot(x, [b['p_imag'] for b in cu], color=BLUE, lw=2, marker='o', ms=3.5, label='(a) imagined')
                ax.plot(x, [b['p_real'] for b in cu], color=BLUE, lw=1.6, ls='--', marker='o', ms=3.5, markerfacecolor=SURF,
                        label='(a) real frame')
                ax.plot(x, [b['p_ceiling'] for b in cu], color=AQUA, lw=1.6, ls=':', marker='s', ms=3, label='ceiling (imag.-trained)')
                cb = c['b']['curve']
                ax.plot([b['r_med'] for b in cb], [b['p_imag'] for b in cb], color=ORANGE, lw=1.6, marker='D', ms=3,
                        label='(b) imagined')
            style(ax)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f'{"original r" if d == "r" else "object-only r_obj"}, H = {H}', color=TXT, fontsize=8.5)
            if i == 1:
                ax.set_xlabel('achieved failure distance (log)')
            if k == 0:
                ax.set_ylabel('P(judged success)')
    axes[0, 0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle(f'M1 DINO-WM PushT: P(judged success) vs failure distance ({st} split; shaded = near-miss band)', color=TXT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f'{R1}/fig_e1_m1_curves_{st}.png', facecolor=SURF)
    plt.close(fig)


def m1_auroc(A, st='val'):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=150, sharey=True)
    for ax, d in zip(axes, ('r', 'r_obj')):
        hs = [H for H in (3, 5, 7, 10) if f'{st}/H{H}/{d}' in A['cells']]
        g = lambda k: [A['cells'][f'{st}/H{H}/{d}']['a'][k] for H in hs]
        ax.plot(hs, g('auc_sn_real'), color=BLUE, lw=1.6, ls='--', marker='o', markerfacecolor=SURF, label='success vs near, real frame')
        ax.plot(hs, g('auc_sn_imag'), color=BLUE, lw=2, marker='o', label='success vs near, imagined')
        ax.plot(hs, g('auc_sn_ceiling'), color=AQUA, lw=1.6, ls=':', marker='s', label='success vs near, ceiling')
        ax.plot(hs, g('auc_sf_imag'), color=ORANGE, lw=2, marker='D', label='success vs far, imagined')
        style(ax, xlog=False)
        ax.set_xticks(hs)
        ax.set_xlabel('imagination horizon H (x5 control steps)')
        ax.set_title('original r' if d == 'r' else 'object-only r_obj', color=TXT, fontsize=9)
    axes[0].set_ylabel('AUROC (judge (a) ensemble score)')
    axes[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle(f'M1 boundary resolution vs horizon ({st} split)', color=TXT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f'{R1}/fig_e1_m1_auroc_{st}.png', facecolor=SURF)
    plt.close(fig)


def m3_curves(M):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.axvspan(1, 2, color='#f0efec', zorder=0)
    ax.axvline(1, color=TXT2, lw=0.8, ls=':')
    cols = {'0.3': AQUA, '0.1': ORANGE, '0.03': BLUE}
    for p in ('0.3', '0.1', '0.03'):
        c = M['pooled'].get(p)
        if not c:
            continue
        cu = c['curve']
        ax.plot(cu['r'], cu['imag'], color=cols[p], lw=2, label=f'imagined, p = {p}')
    c = M['pooled'].get('0.03')
    if c:
        ax.plot(c['curve']['r'], c['curve']['real'], color=TXT, lw=1.4, ls='--', label='real frame (all p)')
    style(ax, ticks=(0.6, 1, 1.5, 2, 3))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('alignment ratio r = |e| / w (log)')
    ax.set_ylabel('P(judged success)')
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.set_title('M3 controlled model: P(judged success) vs r, 3 seeds pooled (r >= 2 is extrapolation)', color=TXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{R3}/fig_e1_m3_curves.png', facecolor=SURF)
    plt.close(fig)


def main():
    p1 = f'{R1}/e1_m1_analysis.json'
    if os.path.exists(p1):
        A = json.load(open(p1))
        for st in ('val', 'trainsplit'):
            if any(k.startswith(st + '/') for k in A['cells']):
                m1_curves(A, st)
                m1_auroc(A, st)
    p3 = f'{R3}/e1_m3_analysis.json'
    if os.path.exists(p3):
        m3_curves(json.load(open(p3)))
    print('FIGS_OK')


if __name__ == '__main__':
    main()
