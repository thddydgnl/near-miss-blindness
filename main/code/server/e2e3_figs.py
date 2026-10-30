"""B E2/E3 figures (CPU analysis env): M1 DP policy law (H = 5), M3 policy law (p = 0.03), E3 judge x input
decomposition of near-miss misjudgments. usage: python e2e3_figs.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R2 = os.path.expanduser('~/pmx/B/results/E2')
R3 = os.path.expanduser('~/pmx/B/results/E3/m1')
SURF, TXT, TXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#9b9a96'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': TXT2, 'axes.labelcolor': TXT, 'xtick.color': TXT2,
                     'ytick.color': TXT2, 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def fmt_ci(c):
    return f'[{c[0]:.2f}, {c[1]:.2f}]' if c else ''


def policy_panels(A, title, out, label_fn, color_fn, nlabel):
    tab = A['table']
    n = np.array([t['n'] for t in tab]); oi = np.array([t['o_imag'] for t in tab])
    s = np.array([t['s'] for t in tab]); sh = np.array([t['sh'] for t in tab]); shr = np.array([t['sh_real'] for t in tab])
    cols = [color_fn(t['policy']) for t in tab]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=150)
    ax = axes[0]
    ax.axhline(0, color=TXT2, lw=0.8, ls=':')
    ax.scatter(n, oi, c=cols, s=28, zorder=3, edgecolor='white', linewidth=0.5)
    for t, x, y in zip(tab, n, oi):
        lb = label_fn(t['policy'])
        if lb:
            ax.annotate(lb, (x, y), xytext=(3, 3), textcoords='offset points', fontsize=6.5, color=TXT2)
    p = A['point']
    ax.set_title(f"Spearman(n, o_imag) = {p['spearman_n_oimag']:.2f} {fmt_ci(A['ci']['spearman_n_oimag'])}", color=TXT, fontsize=8.5)
    ax.set_xlabel(nlabel)
    ax.set_ylabel('imagination part of the over-estimate  o_imag = sh - sh_real')
    style(ax)
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=TXT2, lw=0.8, ls=':')
    ax.scatter(s, shr, facecolor=SURF, edgecolor=cols, s=24, zorder=2, label='judge on real final frame')
    ax.scatter(s, sh, c=cols, s=28, zorder=3, edgecolor='white', linewidth=0.5, label='judge on imagined final frame')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('real success rate s')
    ax.set_ylabel('judged success rate')
    ax.set_title(f"Pearson {p['pearson_sh_s']:.3f}, MMRV {p['mmrv']:.3f}, FGRE {p['fgre']:.3f}, tau_sim {p['tau_similar'] if p['tau_similar'] is None else round(p['tau_similar'], 2)}",
                 color=TXT, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    style(ax)
    fig.suptitle(title, color=TXT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


def m1():
    f = f'{R2}/m1/e2_m1_analysis.json'
    if not os.path.exists(f):
        return
    A = json.load(open(f))
    ep_col = {20: '#c6dbf5', 25: '#9cc2ee', 30: '#6fa6e4', 40: BLUE, 50: '#1f5fae', 80: ORANGE, 220: '#c4501f', 1000: AQUA}
    policy_panels(A['dp']['H5'], 'M1 DINO-WM PushT, 24 Diffusion-Policy checkpoints, H = 5, judge a_std (mode O)',
                  f'{R2}/m1/fig_e2_m1_dp.png', lambda p: p.split('_ep')[1].lstrip('0'),
                  lambda p: ep_col.get(int(p.split('ep')[1]), GREY), 'near-miss share n = P(1 <= r_obj < 2)')
    if 'ctrl' in A:
        policy_panels(A['ctrl']['a'], 'M1 control: R2-12 perturbation policies (val, H = 5), judge a_full ensemble',
                      f'{R2}/m1/fig_e2_m1_ctrl.png', lambda p: p, lambda p: BLUE if 'near' in p else (ORANGE if 'far' in p else GREY),
                      'near-miss share n = P(1 <= r < 2)')


def m3():
    f = f'{R2}/m3/e2_m3_analysis.json'
    if not os.path.exists(f):
        return
    A = json.load(open(f))
    policy_panels(A['0.03']['r29'], 'M3 controlled model, p = 0.03 (3 seeds pooled), frozen R2-9 classifier',
                  f'{R2}/m3/fig_e2_m3.png', lambda p: p.split('_')[0], lambda p: BLUE if 'near' in p else (ORANGE if 'far' in p else GREY),
                  'near-miss share n = P(1 <= r < 1.2)')


def e3():
    f = f'{R3}/e3_m1_analysis.json'
    if not os.path.exists(f):
        return
    A = json.load(open(f))
    rows = []
    for tag, lab in (('val_H5_r', 'M1 val H5, r'), ('val_H5_r_obj', 'M1 val H5, r_obj')):
        c = A['e1_cells'].get(tag, {})
        for j in ('a', 'b', 'd'):
            if j in c:
                rows.append((f'{lab}\njudge ({j})', c[j]))
    for j, c in A['dp'].get('H5', {}).items():
        if 'p_near_real' not in c:
            continue
        rows.append((f'M1 DP H5\njudge {"(a) a_std" if j == "a_std" else "(" + j + ")"}', c))
    if '0.03' in A['m3']:
        rows.append(('M3 p = 0.03\nR2-9 classifier', A['m3']['0.03']))
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(rows) + 2), 3.8), dpi=150)
    x = np.arange(len(rows))
    pr = [c['p_near_real'] for _, c in rows]; pi = [c['p_near_imag'] for _, c in rows]
    er = np.array([[c['p_near_real'] - c['ci_p_near_real'][0], c['ci_p_near_real'][1] - c['p_near_real']] if c.get('ci_p_near_real') else [0, 0] for _, c in rows]).T
    ei = np.array([[c['p_near_imag'] - c['ci_p_near_imag'][0], c['ci_p_near_imag'][1] - c['p_near_imag']] if c.get('ci_p_near_imag') else [0, 0] for _, c in rows]).T
    ax.bar(x - 0.19, pr, 0.36, color='#a9c8ef', label='real final frame (perception floor)', yerr=er, ecolor=TXT2, capsize=2)
    ax.bar(x + 0.19, pi, 0.36, color=BLUE, label='imagined final frame', yerr=ei, ecolor=TXT2, capsize=2)
    for k, (_, c) in enumerate(rows):
        if c.get('share') is not None:
            top = max(pi[k], c['ci_p_near_imag'][1] if c.get('ci_p_near_imag') else pi[k], pr[k], c['ci_p_near_real'][1] if c.get('ci_p_near_real') else pr[k])
            ax.text(x[k], top + 0.03, f"share {100 * c['share']:.0f}%", ha='center', fontsize=6.5, color=TXT2)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_ylabel('P(judged success | near-miss)')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.set_title('Near-miss misjudgments: perception floor vs imagination (numbers = imagination share)', color=TXT, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(f'{R3}/fig_e3_decomp.png', facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    m1(); m3(); e3()
    print('FIGS_OK')
