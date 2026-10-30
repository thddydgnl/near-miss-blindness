"""B E4/E5 figures (CPU analysis env): E4a near-miss / success misjudgment by fine-tuning condition (M1) and by p (M3),
H5a prescriptions in the three settings, the M3 resolution sweep. usage: python e45_figs.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R4 = os.path.expanduser('~/pmx/B/results/E4')
R5 = os.path.expanduser('~/pmx/B/results/E5')
SURF, TXT, TXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#9b9a96'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': TXT2, 'axes.labelcolor': TXT, 'xtick.color': TXT2,
                     'ytick.color': TXT2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def style(ax):
    ax.grid(color=GRID, lw=0.6, axis='y')
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def bars(ax, labels, vals, cis, color, title, ylabel):
    x = np.arange(len(labels))
    err = np.array([[v - c[0], c[1] - v] if c else [0, 0] for v, c in zip(vals, cis)]).T
    ax.bar(x, vals, 0.6, color=color, yerr=err, ecolor=TXT2, capsize=2)
    top = max(max(v + e for v, e in zip(vals, err[1])), 1e-6)
    for k, v in enumerate(vals):
        ax.text(x[k], v + err[1][k] + 0.02 * top, f'{v:.3f}', ha='center', fontsize=6.5, color=TXT2)
    ax.set_ylim(0, top * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title, color=TXT, fontsize=8.5)
    ax.set_ylabel(ylabel)
    style(ax)


def e4():
    f = f'{R4}/e4_analysis.json'
    if not os.path.exists(f):
        return
    A = json.load(open(f))
    cells = [c for c in ('val_H5_r', 'val_H5_r_obj', 'trainsplit_H5_r') if c in A['4a_M1']]
    fig, axes = plt.subplots(2, len(cells) + 1, figsize=(3.0 * (len(cells) + 1), 5.2), dpi=150)
    axes = np.atleast_2d(axes)
    for k, c in enumerate(cells):
        d = A['4a_M1'][c]
        lab = ['public', 'S97', 'S50']
        keys = ['orig', 'S97', 'S50']
        bars(axes[0, k], lab, [d[q]['NM'] for q in keys], [d[q]['ci_NM'] for q in keys], BLUE, f'M1 {c}: NM', 'P(imagined success | near)')
        bars(axes[1, k], lab, [d[q]['SM'] for q in keys], [d[q]['ci_SM'] for q in keys], ORANGE, f'M1 {c}: SM', 'P(imagined failure | success)')
    m3 = A['4a_M3']
    lab = ['p=0.03', 'p=0.1', 'p=0.3']
    keys = ['p0.03', 'p0.1', 'p0.3']
    bars(axes[0, -1], lab, [m3[q]['NM'] for q in keys], [m3[q]['ci_NM'] for q in keys], BLUE, 'M3 E1 grid: NM', '')
    bars(axes[1, -1], lab, [m3[q]['SM'] for q in keys], [m3[q]['ci_SM'] for q in keys], ORANGE, 'M3 E1 grid: SM', '')
    fig.suptitle('E4a: success-skewed (S97) vs balanced (S50) fine-tuning of DINO-WM PushT, and M3 training failure rate p', color=TXT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f'{R4}/fig_e4a.png', facecolor=SURF)
    plt.close(fig)
    sw = A.get('4b_sweep', {})
    if sw:
        v = [k for k in ('base', 'R1', 'R2', 'R3', 'R4') if k in sw]
        names = dict(base='base\n64 tok x 64', R1='R1\n256 tok x 64', R2='R2\n64 tok x 384', R3='R3\nsmall pred.', R4='R4\nlarge pred.')
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), dpi=150)
        x = np.arange(len(v))
        for j, (key, col, lb) in enumerate((('auc_sn_real', TXT, 'real frame'), ('auc_sn_imag', BLUE, 'imagined (frozen clf)'),
                                             ('auc_sn_ceiling', AQUA, 'imagined ceiling'))):
            axes[0].bar(x + (j - 1) * 0.26, [sw[q][key] for q in v], 0.26, color=col, label=lb)
        axes[0].set_ylim(0.5, 1.01)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([names[q] for q in v], fontsize=7)
        axes[0].set_title('AUROC success vs near (p = 0.03, 3 seeds pooled)', color=TXT, fontsize=8.5)
        axes[0].legend(frameon=False, fontsize=7, loc='lower left')
        style(axes[0])
        bars(axes[1], [names[q] for q in v], [sw[q]['NM'] for q in v], [None] * len(v), BLUE, 'NM = P(imagined success | near)', '')
        fig.suptitle('E4b: M3 representation / predictor-size sweep', color=TXT, fontsize=9.5)
        fig.tight_layout()
        fig.savefig(f'{R4}/fig_e4b_sweep.png', facecolor=SURF)
        plt.close(fig)


def e5():
    f = f'{R5}/e5_analysis.json'
    if not os.path.exists(f):
        return
    A = json.load(open(f))['H5a']
    sets = [s for s in ('M1', 'M3_p0.03', 'M3_p0.1') if s in A]
    keys = ['ref', 'rw', 'near', 'cp', 'full', 'bw']
    lab = ['(0) ref', '(i) rw', '(ii) near', '(iii) cp', '(iv) full', '(v) bw']
    fig, axes = plt.subplots(2, len(sets), figsize=(3.6 * len(sets), 5.2), dpi=150)
    axes = np.atleast_2d(axes)
    for k, s in enumerate(sets):
        d = A[s]
        kk = [q for q in keys if q in d]
        ll = [lab[keys.index(q)] for q in kk]
        bars(axes[0, k], ll, [d[q]['NM'] for q in kk], [d[q]['ci_NM'] for q in kk], BLUE, f'{s}: NM (holds: {d["holds"]})', 'P(imagined success | near)')
        bars(axes[1, k], ll, [d[q]['SM'] for q in kk], [d[q]['ci_SM'] for q in kk], ORANGE, f'{s}: SM', 'P(imagined failure | success)')
    fig.suptitle('E5 (H5a): exposure-matched data-side prescriptions', color=TXT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f'{R5}/fig_e5_h5a.png', facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    e4(); e5()
    print('FIGS_OK')
