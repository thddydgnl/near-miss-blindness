"""Metrics, pre-registered criteria (PLAN.md section 6), diagnostics and the two figures. Runs on the pod.
usage: python analyze12.py [sets ...]   (default: main H10 trainsplit, missing sets are skipped)"""
import sys, os, json, csv
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP    # B E0 port
RES = BP.RES
POL = ['far_L', 'mixA_L', 'mixB_L', 'near_L', 'iid_L', 'far_H', 'near_H', 'iid_H']
BINS = ['success', 'near', 'mid', 'far']
NBOOT = 1000


def load(name):
    p = f'{RES}/traj_{name}.jsonl'
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p)]
    return recs


def flags(recs, J):
    tb, tb2 = J['b']['tau'], J['b2']['tau']
    g = lambda k: np.array([x[k] for x in recs], dtype=float)
    F = {}
    F['a_imag'], F['a_real'], F['a_null'] = g('pa_imag') > 0.5, g('pa_real') > 0.5, g('pa_null') > 0.5
    F['b_imag'], F['b_real'], F['b_null'] = g('dv_imag') < tb, g('dv_real') < tb, g('dv_null') < tb
    F['b2_imag'], F['b2_real'], F['b2_null'] = g('df_imag') < tb2, g('df_real') < tb2, g('df_null') < tb2
    F['ai_imag'], F['ai_real'] = g('pai_imag') > 0.5, g('pai_real') > 0.5
    return F


def wmean(x, w):
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else float('nan')


def core(pol, bins, succ, F, w, judges=('a', 'b', 'b2')):
    """All judged metrics for one weighting w (bootstrap = cluster multiplicities)."""
    out = {}
    inpol = np.isin(pol, POL)
    for j in judges:
        fi = F[f'{j}_imag'].astype(float)
        d = {}
        for b in BINS:
            mk = inpol & (bins == b)
            d[f'p_{b}'] = wmean(fi[mk], w[mk])
        d['p_expert'] = wmean(fi[pol == 'expert'], w[pol == 'expert'])
        pn, pf = d['p_near'], d['p_far']
        d['ratio_near_far'] = float(pn / pf) if pf > 0 else float('inf')
        s, sh, n = [], [], []
        for p in POL:
            mk = pol == p
            s.append(wmean(succ[mk].astype(float), w[mk]))
            sh.append(wmean(fi[mk], w[mk]))
            n.append(wmean((bins[mk] == 'near').astype(float), w[mk]))
        s, sh, n = np.array(s), np.array(sh), np.array(n)
        o = sh - s
        d['spearman_near_over'] = float(spearmanr(n, o).correlation)
        d['pearson_near_over'] = float(pearsonr(n, o)[0]) if np.std(o) > 0 else float('nan')
        d['kendall_tau'] = float(kendalltau(sh, s).correlation)
        best, bp = 0.0, None
        for i in range(len(POL)):
            for k in range(i + 1, len(POL)):
                if abs(s[i] - s[k]) <= 0.05 and abs(sh[i] - sh[k]) > best:
                    best, bp = abs(sh[i] - sh[k]), (POL[i], POL[k])
        d['max_imag_gap_matched'] = best
        d['max_gap_pair'] = bp
        d['mean_over'] = float(o.mean())
        d['per_policy'] = dict(true=s.tolist(), imag=sh.tolist(), over=o.tolist(), near_share=n.tolist())
        out[j] = d
    return out


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def analyze(name, recs, J):
    pol = np.array([x['policy'] for x in recs])
    bins = np.array([x['bin'] for x in recs])
    succ = np.array([x['success'] for x in recs])
    r = np.array([x['r'] for x in recs])
    traj = np.array([x['traj'] for x in recs])
    F = flags(recs, J)
    w1 = np.ones(len(recs))
    point = core(pol, bins, succ, F, w1)
    # cluster bootstrap over trajectories
    ut = np.unique(traj)
    tix = np.searchsorted(ut, traj)
    rng = np.random.RandomState(12345)
    keys = ['p_success', 'p_near', 'p_far', 'ratio_near_far', 'spearman_near_over', 'kendall_tau',
            'max_imag_gap_matched', 'mean_over']
    boots = {j: {k: [] for k in keys} for j in ('a', 'b', 'b2')}
    for _ in range(NBOOT):
        mult = np.bincount(rng.randint(0, len(ut), len(ut)), minlength=len(ut)).astype(float)
        bw = mult[tix]
        c = core(pol, bins, succ, F, bw)
        for j in boots:
            for k in keys:
                boots[j][k].append(c[j][k])
    ci = {j: {k: [float(np.nanpercentile(np.array(v, float)[np.isfinite(v)], 2.5)) if np.isfinite(v).any() else None,
                  float(np.nanpercentile(np.array(v, float)[np.isfinite(v)], 97.5)) if np.isfinite(v).any() else None]
              for k, v in boots[j].items()} for j in boots}
    # real-frame (perception) and null / a_imag (diagnostic) views
    inpol = np.isin(pol, POL)
    views = {}
    for v in ['a_real', 'b_real', 'b2_real', 'a_null', 'b_null', 'b2_null', 'ai_imag', 'ai_real']:
        views[v] = {f'p_{b}': float(F[v][inpol & (bins == b)].mean()) if (inpol & (bins == b)).any() else None
                    for b in BINS}
        views[v]['p_expert'] = float(F[v][pol == 'expert'].mean())
        views[v]['acc_vs_truth'] = float((F[v][inpol] == succ[inpol]).mean())
        views[v]['per_policy_success'] = {p: float(F[v][pol == p].mean()) for p in POL}
    # per-policy table
    table = []
    for p in POL + ['expert']:
        mk = pol == p
        row = dict(policy=p, n=int(mk.sum()), true_success=float(succ[mk].mean()),
                   near_share=float((bins[mk] == 'near').mean()), mid_share=float((bins[mk] == 'mid').mean()),
                   far_share=float((bins[mk] == 'far').mean()), r_median=float(np.median(r[mk])))
        for j in ('a', 'b', 'b2'):
            row[f'imag_{j}'] = float(F[f'{j}_imag'][mk].mean())
            row[f'over_{j}'] = row[f'imag_{j}'] - row['true_success']
            row[f'realjudge_{j}'] = float(F[f'{j}_real'][mk].mean())
            row[f'null_{j}'] = float(F[f'{j}_null'][mk].mean())
        row['imag_a_imagjudge'] = float(F['ai_imag'][mk].mean())
        row['real_acc_a'] = float((F['a_real'][mk] == succ[mk]).mean())
        row['real_acc_b'] = float((F['b_real'][mk] == succ[mk]).mean())
        table.append(row)
    # D1 attenuation
    g = lambda k: np.array([x[k] for x in recs], dtype=float)
    dvr, dvi, dve, dvir = g('dv_real'), g('dv_imag'), g('dv_imag_vs_expimag'), g('dv_imag_vs_real')
    D1 = {}
    for b in BINS:
        mk = inpol & (bins == b)
        if mk.any():
            D1[b] = dict(n=int(mk.sum()), med_dv_real=float(np.median(dvr[mk])), med_dv_imag=float(np.median(dvi[mk])),
                         med_dv_imag_vs_expimag=float(np.median(dve[mk])),
                         med_ratio_imagdev_over_realdev=float(np.median(dve[mk] / np.maximum(dvr[mk], 1e-6))),
                         med_dv_imag_vs_real=float(np.median(dvir[mk])))
    me = pol == 'expert'
    D1['expert'] = dict(n=int(me.sum()), med_dv_imag=float(np.median(dvi[me])), med_dv_real=float(np.median(dvr[me])))
    # D3 AUROC success vs near / far
    pa_r, pa_i, pai_i = g('pa_real'), g('pa_imag'), g('pai_imag')
    sm, nm, fm = inpol & (bins == 'success'), inpol & (bins == 'near'), inpol & (bins == 'far')
    D3 = {}
    for nmz, sc in (('a_real', pa_r), ('a_imag', pa_i), ('ai_imag', pai_i), ('b_real', -dvr), ('b_imag', -dvi)):
        D3[nmz] = dict(auc_success_vs_near=auc(sc[sm], sc[nm]), auc_success_vs_far=auc(sc[sm], sc[fm]),
                       auc_near_vs_far=auc(sc[nm], sc[fm]))
    # D4 near-miss composition
    rp, ra = g('r_pos'), g('r_ang')
    ag, bl = g('agent'), g('block')
    comp = np.where(ra > rp, 'angle', np.where(ag ** 2 > bl ** 2, 'agent_pos', 'block_pos'))
    D4 = {}
    for c in ['agent_pos', 'block_pos', 'angle']:
        mk = nm & (comp == c)
        D4[c] = dict(n=int(mk.sum()), **{f'p_{j}_imag': (float(F[f'{j}_imag'][mk].mean()) if mk.any() else None)
                                         for j in ('a', 'b')},
                     **{f'p_{j}_real': (float(F[f'{j}_real'][mk].mean()) if mk.any() else None) for j in ('a', 'b')})
    # curve P(judged success | r) for figure 2
    edges = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9]
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = inpol & (r >= lo) & (r < hi)
        if mk.sum() < 5:
            continue
        row = dict(lo=lo, hi=hi, n=int(mk.sum()), r_med=float(np.median(r[mk])))
        for v in ['a_imag', 'a_real', 'b_imag', 'b_real', 'b2_imag', 'ai_imag', 'a_null']:
            row[v] = float(F[v][mk].mean())
        curve.append(row)
    return dict(set=name, n_traj=len(recs), n_episodes=int(len(np.unique([x['ep'] for x in recs]))),
                n_clusters=int(len(ut)), point=point, ci=ci, views=views, table=table, D1=D1, D3=D3, D4=D4, curve=curve)


def criteria(A):
    pt = A['point']
    s1 = {j: bool(pt[j]['p_near'] >= 0.3 and (pt[j]['p_far'] == 0 or pt[j]['p_near'] >= 3 * pt[j]['p_far']))
          for j in ('a', 'b')}
    use = [j for j in ('a', 'b') if s1[j]] or ['a']
    s2 = {j: bool(pt[j]['spearman_near_over'] >= 0.7) for j in ('a', 'b')}
    s3 = {j: bool(pt[j]['max_imag_gap_matched'] >= 0.20 or pt[j]['kendall_tau'] <= 0.8) for j in ('a', 'b')}
    S1 = any(s1.values())
    S2 = any(s2[j] for j in use) if S1 else False
    S3 = any(s3[j] for j in use) if S1 else False
    verdict = '있음' if (S1 and S2) else ('조건부' if S1 else '없음')
    notes = {}
    for j in ('a', 'b'):
        pr = A['views'][f'{j}_real']['p_near']
        notes[j] = dict(perception_note=bool(abs(pt[j]['p_near'] - pr) < 0.10), p_near_real=pr,
                        pessimism_note=bool(pt[j]['p_success'] < 0.5))
    return dict(S1_by_judge=s1, S2_by_judge=s2, S3_by_judge=s3, judges_used_for_S2_S3=use, S1=S1, S2=S2, S3=S3,
                verdict=verdict, notes=notes)


def figures(A, J):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    TXT, TXT2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    BLUE, ORANGE = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': TXT2, 'axes.labelcolor': TXT, 'xtick.color': TXT2,
                         'ytick.color': TXT2, 'axes.titlesize': 10, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
    # fig 1: dumbbell per policy, sorted by near-miss share
    tab = [t for t in A['table'] if t['policy'] != 'expert']
    tab = sorted(tab, key=lambda t: t['near_share'])
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ys = np.arange(len(tab))
    for y, t in zip(ys, tab):
        ax.plot([t['true_success'], t['imag_a']], [y, y + 0.17], color=BLUE, lw=1.2, alpha=0.5, zorder=1)
        ax.plot([t['true_success'], t['imag_b']], [y, y - 0.17], color=ORANGE, lw=1.2, alpha=0.5, zorder=1)
        ax.scatter(t['true_success'], y, s=46, color=TXT, zorder=3, edgecolor=SURF, linewidth=1.5)
        ax.scatter(t['imag_a'], y + 0.17, s=46, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5)
        ax.scatter(t['imag_b'], y - 0.17, s=46, color=ORANGE, marker='D', zorder=4, edgecolor=SURF, linewidth=1.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t['policy']}  (near {100 * t['near_share']:.0f}%)" for t in tab], color=TXT)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel('success rate')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=TXT, label='true (simulator)'),
         Line2D([], [], marker='o', ls='', color=BLUE, label='imagined, (a) classifier'),
         Line2D([], [], marker='D', ls='', color=ORANGE, label='imagined, (b) goal distance')]
    ax.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8)
    ax.set_title('DINO-WM PushT: true vs imagined success per policy (sorted by near-miss share)', pad=22,
                 color=TXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{RES}/fig1_policy_true_vs_imagined.png', facecolor=SURF)
    plt.close(fig)
    # fig 2: P(judged success) vs r
    cv = A['curve']
    x = np.array([c['r_med'] for c in cv])
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.axvspan(1, 2, color='#f0efec', zorder=0)
    ax.axvline(1, color=TXT2, lw=0.8, ls=':')
    ax.text(1.03, 0.97, 'near-miss band', transform=ax.get_xaxis_transform(), fontsize=7.5, color=TXT2, va='top')
    for key, col, ls, mk, lab in (('a_imag', BLUE, '-', 'o', '(a) classifier, imagined'),
                                  ('a_real', BLUE, '--', 'o', '(a) classifier, real frame'),
                                  ('b_imag', ORANGE, '-', 'D', '(b) goal distance, imagined'),
                                  ('b_real', ORANGE, '--', 'D', '(b) goal distance, real frame')):
        ax.plot(x, [c[key] for c in cv], color=col, ls=ls, lw=2, marker=mk, ms=5,
                markerfacecolor=col if ls == '-' else SURF, markeredgecolor=col, label=lab)
    ax.set_xscale('log')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [0.1, 0.2, 0.5, 1, 2, 3, 5, 10]
    ax.xaxis.set_major_locator(FixedLocator(tk))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in tk]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('final error / success threshold  (r; success iff r < 1)')
    ax.set_ylabel('P(judged success)')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.set_title('P(success judged) vs final error, all 8 policies pooled (main set)', color=TXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{RES}/fig2_p_success_vs_error.png', facecolor=SURF)
    plt.close(fig)


def main():
    sets = sys.argv[1:] or ['main', 'H10', 'trainsplit']
    J = json.load(open(f'{RES}/judges.json'))
    allres = {}
    for s in sets:
        recs = load(s)
        if recs is None:
            print('skip', s)
            continue
        A = analyze(s, recs, J)
        A['criteria'] = criteria(A)
        allres[s] = A
        json.dump(A, open(f'{RES}/analysis_{s}.json', 'w'), indent=1)
        with open(f'{RES}/summary_policies_{s}.csv', 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(A['table'][0].keys()))
            wr.writeheader()
            for row in A['table']:
                wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        pt = A['point']
        print(f"=== {s}: {A['n_traj']} traj, {A['n_episodes']} episodes, {A['n_clusters']} clusters")
        for j in ('a', 'b', 'b2'):
            print(f"  [{j}] P(imag succ | success/near/mid/far/expert) = {pt[j]['p_success']:.3f} / {pt[j]['p_near']:.3f} / "
                  f"{pt[j]['p_mid']:.3f} / {pt[j]['p_far']:.3f} / {pt[j]['p_expert']:.3f}; ratio {pt[j]['ratio_near_far']:.2f}; "
                  f"Spearman(near, over) {pt[j]['spearman_near_over']:.3f}; Kendall {pt[j]['kendall_tau']:.3f}; "
                  f"max matched gap {pt[j]['max_imag_gap_matched']:.3f} {pt[j]['max_gap_pair']}; mean over {pt[j]['mean_over']:+.3f}")
            print(f"      CI p_near {A['ci'][j]['p_near']} p_far {A['ci'][j]['p_far']} spearman {A['ci'][j]['spearman_near_over']}"
                  f" kendall {A['ci'][j]['kendall_tau']}")
        for v in ['a_real', 'b_real', 'a_null', 'b_null', 'ai_imag']:
            vv = A['views'][v]
            print(f"  view {v}: " + ' '.join(f"{k}={vv[k]:.3f}" for k in ['p_success', 'p_near', 'p_mid', 'p_far', 'p_expert',
                                                                       'acc_vs_truth'] if vv.get(k) is not None))
        for t in A['table']:
            print(f"  {t['policy']:7s} true {t['true_success']:.3f} near {t['near_share']:.3f} far {t['far_share']:.3f} | "
                  f"imag a {t['imag_a']:.3f} b {t['imag_b']:.3f} b2 {t['imag_b2']:.3f} | real-judge a {t['realjudge_a']:.3f} "
                  f"b {t['realjudge_b']:.3f} | null a {t['null_a']:.3f} | a_imagjudge {t['imag_a_imagjudge']:.3f}")
        print('  D1', json.dumps(A['D1']))
        print('  D3', json.dumps(A['D3']))
        print('  D4', json.dumps(A['D4']))
        print('  CRITERIA', json.dumps(A['criteria'], ensure_ascii=False))
        if s == 'main':
            figures(A, J)
    json.dump({s: dict(criteria=A['criteria'], point={j: {k: v for k, v in A['point'][j].items() if k != 'per_policy'}
                                                      for j in A['point']}, ci=A['ci']) for s, A in allres.items()},
              open(f'{RES}/criteria.json', 'w'), indent=1, ensure_ascii=False)
    print('ANALYZE_DONE')


if __name__ == '__main__':
    main()
