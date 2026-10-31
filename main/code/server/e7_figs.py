"""B E7 figures (CPU): PointMaze P(judged success) vs failure distance for H in {3,5,7,10} (val and trainsplit) and
AUROC / two-sided shift vs horizon. Reads e7_analysis.json only. usage: python e7_figs.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

RES = os.path.expanduser(os.environ.get('B_RES', '~/pmx/B/results/E7/pm'))
HS = (3, 5, 7, 10)
SURF, TXT, TXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a86'
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


def curves(A, st):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), dpi=150, sharey=True)
    for k, H in enumerate(HS):
        ax = axes[k]
        c = A['cells'].get(f'{st}/H{H}/r')
        ax.axvspan(1, 2, color='#f0efec', zorder=0)
        ax.axvline(1, color=TXT2, lw=0.8, ls=':')
        if c:
            cu = c['a']['curve']
            x = [b['r_med'] for b in cu]
            ax.plot(x, [b['p_imag'] for b in cu], color=BLUE, lw=2, marker='o', ms=3.5, label='(a) imagined')
            ax.plot(x, [b['p_real'] for b in cu], color=BLUE, lw=1.6, ls='--', marker='o', ms=3.5, markerfacecolor=SURF, label='(a) real frame')
            ax.plot(x, [b['p_ceiling'] for b in cu], color=AQUA, lw=1.6, ls=':', marker='s', ms=3, label='probe (imag.-trained)')
            cb = c['b']['curve']
            ax.plot([b['r_med'] for b in cb], [b['p_imag'] for b in cb], color=ORANGE, lw=1.6, marker='D', ms=3, label='(b) imagined')
        style(ax)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'PointMaze, H = {H}', color=TXT, fontsize=8.5)
        ax.set_xlabel('achieved failure distance r (log)')
        if k == 0:
            ax.set_ylabel('P(judged success)')
    axes[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle(f'E7 DINO-WM PointMaze: P(judged success) vs failure distance ({st}; shaded = near-miss band)', color=TXT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f'{RES}/fig_e7_curves_{st}.png', facecolor=SURF)
    plt.close(fig)


def horizon(A, st):
    c = {H: A['cells'].get(f'{st}/H{H}/r', {}).get('a') for H in HS}
    hs = [H for H in HS if c[H]]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 3.0), dpi=150)
    for key, col, ls, lab in (('auc_sn_real', BLUE, '--', 'success vs near, real frame'), ('auc_sn_imag', BLUE, '-', 'success vs near, imagined'),
                              ('auc_sn_ceiling', AQUA, ':', 'success vs near, probe on imagined'), ('auc_sf_imag', ORANGE, '-', 'success vs far, imagined')):
        ax.plot(hs, [c[H][key] for H in hs], color=col, ls=ls, marker='o', ms=4, label=lab)
    ax.axvline(5, color=GRAY, lw=0.9, ls=(0, (4, 2)))
    ax.set_xticks(hs); ax.set_ylim(0.5, 1.01); ax.set_xlabel('imagination horizon H'); ax.set_ylabel('AUROC of success'); style(ax, xlog=False)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left')
    for key, col, lab in (('p_success', '#444', 'successes'), ('p_near', BLUE, 'near misses')):
        bx.plot(hs, [c[H][key + '_real'] for H in hs], color=col, ls='--', marker='o', ms=4, markerfacecolor=SURF, label=f'{lab}, real frame')
        bx.plot(hs, [c[H][key + '_imag'] for H in hs], color=col, ls='-', marker='o', ms=4, label=f'{lab}, imagined')
    bx.set_xticks(hs); bx.set_ylim(0, 1); bx.set_xlabel('imagination horizon H'); bx.set_ylabel('P(judged success)'); style(bx, xlog=False)
    bx.legend(frameon=False, fontsize=6.5, loc='center left')
    fig.suptitle(f'E7 DINO-WM PointMaze ({st}): boundary resolution and two-sided shift vs horizon', color=TXT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f'{RES}/fig_e7_horizon_{st}.png', facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    A = json.load(open(f'{RES}/e7_analysis.json'))
    for st in ('val', 'trainsplit'):
        if any(f'{st}/H{H}/r' in A['cells'] for H in HS):
            curves(A, st)
            horizon(A, st)
    print('E7_FIGS_DONE')
