"""B E0 figure: DP training stages, sim-only (calibration seeds), success share (r_obj < 1) and near-miss share
(1 <= r_obj < 2) vs epoch per training seed; chosen stages marked. Reads results/<stage>/dp_snapeval_summary.json.
usage: python fig_dp_stages.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

STAGES = [20, 25, 30, 40, 50, 80, 220, 1000]
SURF, TXT, TXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COL = {42: '#2a78d6', 43: '#eb6834', 44: '#1baf7a'}          # categorical slots 1-3 (validated all-pairs, light)

rows = json.load(open(f'{BP.RES}/dp_snapeval_summary.json'))
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': TXT2, 'axes.labelcolor': TXT, 'xtick.color': TXT2,
                     'ytick.color': TXT2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=150, sharex=True)
for ax, key, title in ((axes[0], 'success_robj', 'success share (r_obj < 1)'),
                       (axes[1], 'near_robj', 'near-miss share (1 <= r_obj < 2)')):
    for e in STAGES:
        ax.axvline(e, color=GRID, lw=3, zorder=0)
    for s in (42, 43, 44):
        rr = sorted([r for r in rows if r['seed'] == s], key=lambda r: r['epoch'])
        if not rr:
            continue
        x = [r['epoch'] for r in rr]
        y = [r[key] for r in rr]
        ax.plot(x, y, color=COL[s], lw=2, marker='o', ms=4.5, markeredgecolor=SURF, markeredgewidth=1.2,
                label=f'seed {s}', zorder=3)
    ax.set_xscale('log')
    tk = [8, 10, 20, 30, 50, 100, 200, 500, 1000]
    ax.xaxis.set_major_locator(FixedLocator(tk))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.02, 1.0 if key == 'success_robj' else 0.5)
    ax.set_xlabel('training epoch (EMA snapshot; grey bands = the 8 chosen stages)')
    ax.set_title(title, color=TXT, fontsize=9.5)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False, fontsize=8, loc='upper left')
fig.suptitle('PushT Diffusion Policy in the DINO-WM env: sim-only stage evaluation (50 calibration episodes each)',
             color=TXT, fontsize=9.5)
fig.tight_layout()
fig.savefig(f'{BP.RES}/fig_dp_stages.png', facecolor=SURF)
print('FIG_OK')
