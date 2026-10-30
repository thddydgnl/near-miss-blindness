"""R2-11 judging (PLAN.md section 6): criteria S1-S4, verdict, summary tables and the three figures.
Usage: python judge.py --res /workspace/results
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np

VARS = ['A', 'B', 'R', 'C', 'W', 'S']
VNAME = {'A': 'agent pos', 'B': 'box pos', 'R': 'box angle', 'C': 'contact', 'W': 'wall prox', 'S': 'box speed'}
CRUSH = {'mlp': ['w256', 'mlp64', 'mlp16', 'mlp8'], 'res': ['w256', 'res64', 'res16', 'res8']}
WIDTH = {'w256': 256, 'mlp64': 64, 'mlp16': 16, 'mlp8': 8, 'res64': 64, 'res16': 16, 'res8': 8}
NAMES = ['w256', 'mlp64', 'mlp16', 'mlp8', 'res64', 'res16', 'res8']
# reference categorical palette (dataviz skill, light mode, fixed slot order)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='/workspace/results')
    args = ap.parse_args()
    R = args.res
    shared = json.load(open(f'{R}/analysis/shared.json'))
    recs = {}
    for f in glob.glob(f'{R}/analysis/*_s*_*.json'):
        r = json.load(open(f))
        recs[(r['name'], r['seed'], r['kind'])] = r
    seeds = sorted({k[1] for k in recs})
    evals = []
    for f in glob.glob(f'{R}/evals_s*.jsonl'):
        evals += [json.loads(l) for l in open(f)]
    cem = json.load(open(f'{R}/cem.json')) if os.path.exists(f'{R}/cem.json') else None

    def get(name, s, kind='trained'):
        return recs.get((name, s, kind))

    def sm(fn, name, kind='trained'):
        return mean([fn(get(name, s, kind)) for s in seeds if get(name, s, kind) is not None])

    orc = shared['oracle']['r2_delta']
    resolvable = [v for v in VARS if orc[v] >= 0.2]
    out = dict(seeds=seeds, resolvable=resolvable, oracle_r2_delta=orc, shares=shared['shares'],
               copy_raw_r2=shared['copy']['raw_r2'], oracle_raw_r2=shared['oracle']['raw_r2'],
               oracle_CI=shared['oracle']['confusion']['CI'], copy_CI=shared['copy']['confusion']['CI'])
    # ---------------------------------------------------------------- per-config summary
    rows = []
    last_step = max(e['step'] for e in evals) if evals else None
    for name in NAMES:
        row = dict(name=name, crush='both' if name == 'w256' else name[:3], width=WIDTH[name])
        ev = [e for e in evals if e['cond'] == name]
        row['n_params'] = ev[0]['n_params'] if ev else None
        row['id_rel_final'] = mean([e['id_rel'] for e in ev if e['step'] == last_step])
        row['id_rel_half'] = mean([e['id_rel'] for e in ev if e['step'] == (last_step // 2 if last_step else -1)])
        for v in VARS:
            row[f'r2d_{v}'] = sm(lambda r: r['probe']['r2_delta'][v], name)
            row[f'raw_{v}'] = sm(lambda r: r['probe']['raw_r2'][v], name)
        row['r2d_mean_resolvable'] = mean([row[f'r2d_{v}'] for v in resolvable])
        row['CI'] = sm(lambda r: r['confusion']['CI'], name)
        row['CI_rel'] = sm(lambda r: r['confusion']['CI_rel'], name)
        for v in ['A', 'R', 'W']:
            row[f'conf_{v}'] = sm(lambda r: r['confusion'][v]['unrel'], name)
            row[f'confrel_{v}'] = sm(lambda r: r['confusion'][v]['rel'], name)
        row['CI_v1'] = sm(lambda r: r['confusion_v1']['CI'], name)
        row['CI_rel_v1'] = sm(lambda r: r['confusion_v1']['CI_rel'], name)
        for v in ['A', 'R', 'C', 'W', 'S']:
            row[f'conf_v1_{v}'] = sm(lambda r: r['confusion_v1'][v]['unrel'], name)
        row['Lw'] = sm(lambda r: r['neurons']['pooled'].get('Lw'), name)
        row['Lw_rot'] = sm(lambda r: mean([x.get('Lw') for x in r['neurons']['rot']]), name)
        row['Lw_untrained'] = sm(lambda r: r['neurons']['pooled'].get('Lw'), name, 'untrained')
        row['Lw_v1'] = sm(lambda r: r['neurons']['pooled_v1'].get('Lw'), name)
        row['Lw_rot_v1'] = sm(lambda r: mean([x.get('Lw') for x in r['neurons']['rot_v1']]), name)
        row['pack_v1'] = sm(lambda r: r['neurons']['pooled_v1'].get('pack_rate'), name)
        row['pack_rot_v1'] = sm(lambda r: mean([x.get('pack_rate') for x in r['neurons']['rot_v1']]), name)
        row['pack'] = sm(lambda r: r['neurons']['pooled'].get('pack_rate'), name)
        row['pack_rot'] = sm(lambda r: mean([x.get('pack_rate') for x in r['neurons']['rot']]), name)
        row['pack_untrained'] = sm(lambda r: r['neurons']['pooled'].get('pack_rate'), name, 'untrained')
        row['SLI_multi'] = sm(lambda r: r['neurons']['pooled'].get('SLI_multi'), name)
        row['SLI_multi_rot'] = sm(lambda r: mean([x.get('SLI_multi') for x in r['neurons']['rot']]), name)
        row['frac_multi'] = sm(lambda r: r['neurons']['pooled'].get('frac_multi'), name)
        row['frac_multi_rot'] = sm(lambda r: mean([x.get('frac_multi') for x in r['neurons']['rot']]), name)
        row['n_incl'] = sm(lambda r: r['neurons']['pooled'].get('n_incl'), name)
        row['n_units'] = sm(lambda r: r['neurons']['pooled'].get('n_units'), name)
        row['mean_R2full'] = sm(lambda r: r['neurons']['pooled'].get('mean_R2full'), name)
        for k in ['R2_1', 'R2_5', 'R2_all', 'rot_R2_1', 'rot_R2_5', 'frac_merged', 'coh_D', 'rand_D']:
            row[f'merged_{k}'] = sm(lambda r: (r.get('merged') or {}).get('summary', {}).get(k), name)
            row[f'merged_untr_{k}'] = sm(lambda r: (r.get('merged') or {}).get('summary', {}).get(k), name, 'untrained')
        if cem:
            runs = [x for x in cem['runs'] if x['cond'] == name]
            for k in ['succ_precise', 'succ_coarse', 'agent_err', 'box_err', 'ang_err']:
                row[f'cem_{k}'] = mean([x[k] for x in runs])
        rows.append(row)
    byname = {r['name']: r for r in rows}
    out['summary'] = rows
    # ---------------------------------------------------------------- S1
    crit = {}
    for c, chain in CRUSH.items():
        wide, nar = chain[0], chain[-1]
        drops = {v: byname[wide][f'r2d_{v}'] - byname[nar][f'r2d_{v}'] for v in VARS}
        mdrop = mean([drops[v] for v in resolvable])
        s1a = mdrop is not None and mdrop >= 0.1
        dci = byname[nar]['CI'] - byname[wide]['CI']
        B = min(len(get(wide, s)['confusion_boot']) for s in seeds)
        diffs = np.mean([np.array(get(nar, s)['confusion_boot'][:B]) - np.array(get(wide, s)['confusion_boot'][:B])
                         for s in seeds], 0)
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        sig_up = (lo > 0) and (dci >= 0.02)
        s1b = not sig_up
        # secondary: v1 definition (all samples, matrix as registered at 08:22Z, own-value control only)
        dci1 = byname[nar]['CI_v1'] - byname[wide]['CI_v1']
        d1 = np.mean([np.array(get(nar, s)['confusion_v1_boot'][:B]) - np.array(get(wide, s)['confusion_v1_boot'][:B])
                      for s in seeds], 0)
        lo1, hi1 = float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))
        crit[c] = dict(S1=dict(drops=drops, mean_drop_resolvable=mdrop, any_drop_ge_0p1=any(drops[v] >= 0.1 for v in resolvable),
                               S1a=s1a, dCI=dci, dCI_ci95=[lo, hi], conf_sig_increase=sig_up, S1b=s1b, pass_=s1a and s1b,
                               CI_wide=byname[wide]['CI'], CI_narrow=byname[nar]['CI'],
                               CI_rel_wide=byname[wide]['CI_rel'], CI_rel_narrow=byname[nar]['CI_rel'],
                               v1=dict(dCI=dci1, ci95=[lo1, hi1], sig_increase=(lo1 > 0) and (dci1 >= 0.02),
                                       CI_wide=byname[wide]['CI_v1'], CI_narrow=byname[nar]['CI_v1'])))
        # ------------------------------------------------------------ S2
        Lw_obs = byname[nar]['Lw']
        null = [x.get('Lw') for s in seeds for x in get(nar, s)['neurons']['rot'] if x.get('Lw') is not None]
        nm, nsd = float(np.mean(null)), float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
        s2a = Lw_obs is not None and Lw_obs > nm + 2 * nsd

        def lexc(name):
            return mean([get(name, s)['neurons']['pooled'].get('Lw') - mean([x.get('Lw') for x in get(name, s)['neurons']['rot']])
                         for s in seeds])
        Lw_wide, Lw_nar = lexc(wide), lexc(nar)
        dL = (Lw_wide - Lw_nar) / Lw_wide if Lw_wide else None
        Rw, Rn = byname[wide]['r2d_mean_resolvable'], byname[nar]['r2d_mean_resolvable']
        dR = (Rw - Rn) / Rw if Rw else None
        s2b = dL is not None and (dL <= 0 or (dR is not None and dL < dR))
        null1 = [x.get('Lw') for s in seeds for x in get(nar, s)['neurons']['rot_v1'] if x.get('Lw') is not None]
        v1 = dict(Lw_narrow=byname[nar]['Lw_v1'], rot_mean=float(np.mean(null1)), rot_sd=float(np.std(null1, ddof=1)),
                  Lw_wide=byname[wide]['Lw_v1'], rot_wide=byname[wide]['Lw_rot_v1'])
        v1['S2a'] = v1['Lw_narrow'] > v1['rot_mean'] + 2 * v1['rot_sd']
        crit[c]['S2'] = dict(v1=v1, Lw_narrow=Lw_obs, rot_null_mean=nm, rot_null_sd=nsd, rot_null_thr=nm + 2 * nsd, S2a=s2a,
                             Lexc_wide=Lw_wide, Lexc_narrow=Lw_nar, dL_rel=dL, dR_rel=dR, S2b=s2b, pass_=s2a and s2b,
                             Lw_untrained_narrow=byname[nar]['Lw_untrained'], Lw_wide=byname[wide]['Lw'],
                             pack_narrow=byname[nar]['pack'], pack_rot_narrow=byname[nar]['pack_rot'])
        # ------------------------------------------------------------ S3
        rl, sh, used = [], [], []
        for v in resolvable:
            w = byname[wide][f'r2d_{v}']
            if w is None or w <= 0:
                continue
            rl.append(1 - byname[nar][f'r2d_{v}'] / w)
            sh.append(-shared['shares'][v])
            used.append(v)
        rho = spearman(np.array(rl), np.array(sh)) if len(rl) >= 3 else None
        crit[c]['S3'] = dict(vars=used, rel_loss=dict(zip(used, rl)), share={v: shared['shares'][v] for v in used},
                             rho=rho, pass_=rho is not None and rho >= 0.6)
        # ------------------------------------------------------------ S4
        if cem:
            base = {x['cond']: x for x in cem['runs'] if x['seed'] == -1}
            bc = max(base['baseline_zero']['succ_coarse'], base['baseline_random']['succ_coarse'])
            P = {n: mean([x['succ_precise'] for x in cem['runs'] if x['cond'] == n]) for n in (wide, nar)}
            C = {n: mean([x['succ_coarse'] for x in cem['runs'] if x['cond'] == n]) for n in (wide, nar)}
            disc = C[wide] is not None and C[wide] >= 2 * bc and C[wide] >= bc + 0.30
            dp, dc = P[wide] - P[nar], C[wide] - C[nar]
            s4 = disc and dp >= 2 * dc and dp >= 0.10
            # uniform-degradation null: scale the wide model's per-task errors to reproduce the narrow coarse success
            nulls = []
            for s in seeds:
                wr = [x for x in cem['runs'] if x['cond'] == wide and x['seed'] == s][0]
                nr = [x for x in cem['runs'] if x['cond'] == nar and x['seed'] == s][0]
                ea, eb, er = (np.array(wr['errs'][k]) for k in ('agent', 'box', 'ang'))
                ks = np.arange(0.5, 20.0, 0.01)
                cs = np.array([np.mean((k * ea <= 8) & (k * eb <= 8)) for k in ks])
                kbest = ks[np.argmin(np.abs(cs - nr['succ_coarse']))]
                pp = float(np.mean((kbest * ea <= 2) & (kbest * eb <= 2) & (kbest * er <= 5)))
                nulls.append(dict(k=float(kbest), pred_precise=pp, null_drop=wr['succ_precise'] - pp))
            null_drop = mean([x['null_drop'] for x in nulls])
            crit[c]['S4'] = dict(discriminative=disc, baseline_coarse_max=bc, precise=P, coarse=C, drop_precise=dp,
                                 drop_coarse=dc, pass_=s4, uniform_null=nulls, null_drop_precise=null_drop,
                                 threshold_effect_flag=(dp <= null_drop) if null_drop is not None else None,
                                 baselines={k: dict(precise=v['succ_precise'], coarse=v['succ_coarse']) for k, v in base.items()})
    # ---------------------------------------------------------------- verdict
    S1 = all(crit[c]['S1']['pass_'] for c in CRUSH)
    S2 = all(crit[c]['S2']['pass_'] for c in CRUSH)
    full_one = [c for c in CRUSH if crit[c]['S1']['S1a'] and crit[c]['S1']['S1b'] and crit[c]['S2']['S2a'] and crit[c]['S2']['S2b']]
    weak_all = all(crit[c]['S1']['S1b'] and crit[c]['S2']['pass_'] for c in CRUSH) and not all(crit[c]['S1']['S1a'] for c in CRUSH)
    packing = [c for c in CRUSH if crit[c]['S1']['S1a'] and crit[c]['S1']['conf_sig_increase'] and not crit[c]['S2']['S2b']]
    if S1 and S2:
        verdict = '있음'
    elif packing:
        verdict = '있음(다른 방향, packing)'
    elif full_one or weak_all:
        verdict = '조건부 있음'
    else:
        verdict = '없음'
    out['criteria'] = crit
    out['verdict'] = dict(S1=S1, S2=S2, S3=all(crit[c]['S3']['pass_'] for c in CRUSH),
                          S3_any=[c for c in CRUSH if crit[c]['S3']['pass_']],
                          S4=all(crit[c].get('S4', {}).get('pass_', False) for c in CRUSH),
                          S4_any=[c for c in CRUSH if crit[c].get('S4', {}).get('pass_', False)],
                          full_one=full_one, weak_all=weak_all, packing=packing, verdict=verdict)
    json.dump(out, open(f'{R}/criteria.json', 'w'), indent=1)
    keys = list(rows[0].keys())
    with open(f'{R}/summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    with open(f'{R}/curves.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cond', 'seed', 'step', 'id_rel', 'id_mse', 'train_loss'])
        for e in sorted(evals, key=lambda e: (e['cond'], e['seed'], e['step'])):
            w.writerow([e['cond'], e['seed'], e['step'], round(e['id_rel'], 5), round(e['id_mse'], 6),
                        None if e['train_loss'] is None else round(e['train_loss'], 6)])
    print(json.dumps(out['verdict'], ensure_ascii=False, indent=1))
    for c in CRUSH:
        print(c, json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('uniform_null',)} for k, v in crit[c].items()},
                            ensure_ascii=False, default=str)[:3000])
    figures(R, rows, byname, crit, shared, seeds, recs)


def figures(R, rows, byname, crit, shared, seeds, recs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False})
    widths = [8, 16, 64, 256]
    titles = {'mlp': 'MLP-hidden crush (d_model 256)', 'res': 'Residual-width crush (d_model = d)'}

    def chain_vals(c, key):
        names = CRUSH[c][::-1]           # 8, 16, 64, 256
        return [byname[n][key] for n in names]
    # fig 1: resolution vs width per variable
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, c in zip(axs, CRUSH):
        for k, v in enumerate(VARS):
            ys = chain_vals(c, f'r2d_{v}')
            ls = '-' if v in shared and False else '-'
            ax.plot(widths, ys, ls, color=PAL[k], lw=2, marker='o', ms=5, label=f'{v} {VNAME[v]}')
            ax.annotate(v, (256, ys[-1]), xytext=(4, 0), textcoords='offset points', va='center', color=INK2, fontsize=8)
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlabel('width d (neurons per MLP layer)')
        ax.set_title(titles[c], color=INK, fontsize=9.5)
        ax.axhline(0, color=INK2, lw=0.8)
    axs[0].set_ylabel('innovation R²  (1 − MSE/MSE_copy)')
    axs[1].legend(loc='center left', bbox_to_anchor=(1.06, 0.5), fontsize=8)
    fig.suptitle('Resolution of each physical variable in the predicted next frame', color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f'{R}/fig1_resolution_vs_width.png', dpi=150)
    plt.close(fig)
    # fig 2: confusion vs width
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, c in zip(axs, CRUSH):
        names = CRUSH[c][::-1]
        ci = [byname[n]['CI'] for n in names]
        lo, hi = [], []
        for n in names:
            bs = np.mean([np.array(recs[(n, s, 'trained')]['confusion_boot']) for s in seeds], 0)
            lo.append(np.percentile(bs, 2.5))
            hi.append(np.percentile(bs, 97.5))
        ax.fill_between(widths, lo, hi, color=PAL[0], alpha=0.15, lw=0)
        ax.plot(widths, ci, '-', color=PAL[0], lw=2, marker='o', ms=5, label='unrelated variables (CI)')
        ax.plot(widths, [byname[n]['CI_rel'] for n in names], '--', color=PAL[1], lw=2, marker='s', ms=5,
                label='related variables (CI_rel)')
        ax.axhline(shared['oracle']['confusion']['CI'], color=INK2, lw=1, ls=':', label='oracle (true next frame)')
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlabel('width d')
        ax.set_title(titles[c], color=INK, fontsize=9.5)
    axs[0].set_ylabel('partial R² of probe error on other\nvariables (A,R,W; non-interacting)')
    axs[1].legend(loc='upper right', fontsize=8)
    fig.suptitle('Confusion: does a variable\'s probe error depend on other variables?', color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f'{R}/fig2_confusion_vs_width.png', dpi=150)
    plt.close(fig)
    # fig 3: locality vs width with nulls
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, c in zip(axs, CRUSH):
        names = CRUSH[c][::-1]
        tr = [byname[n]['Lw'] for n in names]
        rm, rs = [], []
        for n in names:
            null = [x.get('Lw') for s in seeds for x in recs[(n, s, 'trained')]['neurons']['rot'] if x.get('Lw') is not None]
            rm.append(np.mean(null))
            rs.append(np.std(null, ddof=1) if len(null) > 1 else 0)
        rm, rs = np.array(rm), np.array(rs)
        ax.fill_between(widths, rm - 2 * rs, rm + 2 * rs, color=PAL[1], alpha=0.18, lw=0)
        ax.plot(widths, rm, '--', color=PAL[1], lw=2, marker='s', ms=5, label='random-rotation null (±2 SD)')
        ax.plot(widths, [byname[n]['Lw_untrained'] for n in names], ':', color=PAL[2], lw=2, marker='^', ms=5,
                label='untrained model')
        ax.plot(widths, tr, '-', color=PAL[0], lw=2, marker='o', ms=5, label='trained neurons')
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlabel('width d')
        ax.set_title(titles[c], color=INK, fontsize=9.5)
    axs[0].set_ylabel('semantic locality L_w of\nneuron GT sets (1 = one variable)')
    axs[1].legend(loc='center right', fontsize=8)
    fig.suptitle('Neuron-level locality of ground-truth variable sets', color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f'{R}/fig3_locality_vs_width.png', dpi=150)
    plt.close(fig)
    print('figures written')


if __name__ == '__main__':
    main()
