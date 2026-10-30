"""B E0 (runs on the Mac, reads results only): R2-12 pilot numbers vs the server reproduction.
usage: python3 compare_m1_repro.py <pilot results dir> <server results dir (E0_server/m1)> <out json>"""
import sys, json, os

P, S, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def lj(d, f):
    p = os.path.join(d, f)
    return json.load(open(p)) if os.path.exists(p) else None


rows = []


def add(name, a, b):
    d = None if (a is None or b is None) else (b - a)
    rows.append(dict(metric=name, pilot=a, server=b, diff=d))


jp, js = lj(P, 'judges.json'), lj(S, 'judges.json')
for j in ('a', 'a_imag', 'b', 'b2'):
    add(f'judge {j} held-out acc', jp[j]['heldout_acc'], js[j]['heldout_acc'])
add('judge b tau', jp['b']['tau'], js['b']['tau'])
for setn in ('main', 'H10', 'trainsplit'):
    ap, asv = lj(P, f'analysis_{setn}.json'), lj(S, f'analysis_{setn}.json')
    if ap is None or asv is None:
        continue
    for j in ('a', 'b', 'b2'):
        for k in ('p_success', 'p_near', 'p_mid', 'p_far', 'p_expert', 'spearman_near_over', 'kendall_tau',
                  'max_imag_gap_matched', 'mean_over'):
            add(f'{setn} [{j}] {k}', ap['point'][j][k], asv['point'][j][k])
    for v in ('a_real', 'b_real'):
        add(f'{setn} view {v} p_near', ap['views'][v]['p_near'], asv['views'][v]['p_near'])
        add(f'{setn} view {v} p_success', ap['views'][v]['p_success'], asv['views'][v]['p_success'])
    for z in ('a_real', 'a_imag', 'ai_imag', 'b_real', 'b_imag'):
        add(f'{setn} AUROC succ-vs-near {z}', ap['D3'][z]['auc_success_vs_near'], asv['D3'][z]['auc_success_vs_near'])
        add(f'{setn} AUROC succ-vs-far {z}', ap['D3'][z]['auc_success_vs_far'], asv['D3'][z]['auc_success_vs_far'])
    for t_p, t_s in zip(ap['table'], asv['table']):
        assert t_p['policy'] == t_s['policy']
        add(f"{setn} policy {t_p['policy']} true_success", t_p['true_success'], t_s['true_success'])
        add(f"{setn} policy {t_p['policy']} near_share", t_p['near_share'], t_s['near_share'])
        for j in ('a', 'b'):
            add(f"{setn} policy {t_p['policy']} imag_{j}", t_p[f'imag_{j}'], t_s[f'imag_{j}'])
    add(f'{setn} verdict(pilot=1 if same)', 1.0, 1.0 if ap['criteria']['verdict'] == asv['criteria']['verdict'] else 0.0)
pp, ps = lj(P, 'posthoc.json'), lj(S, 'posthoc.json')
if pp and ps:
    for setn in pp:
        if setn not in ps:
            continue
        for j in ('a', 'b'):
            add(f'{setn} P1[{j}] near_shift', pp[setn][f'P1_{j}']['near_shift']['point'], ps[setn][f'P1_{j}']['near_shift']['point'])
            add(f'{setn} P1[{j}] succ_shift', pp[setn][f'P1_{j}']['succ_shift']['point'], ps[setn][f'P1_{j}']['succ_shift']['point'])
            add(f'{setn} P3[{j}] spearman(near, imagination)', pp[setn][f'P3_{j}']['spearman_near_imagination'],
                ps[setn][f'P3_{j}']['spearman_near_imagination'])
        add(f'{setn} P4 ai spearman', pp[setn]['P4_ai']['point']['spearman'], ps[setn]['P4_ai']['point']['spearman'])
exact = [r for r in rows if r['diff'] is not None and abs(r['diff']) < 1e-9]
absd = [abs(r['diff']) for r in rows if r['diff'] is not None]
summ = dict(n_metrics=len(rows), n_exact=len(exact), max_abs_diff=max(absd) if absd else None,
            n_absdiff_gt_0p01=sum(1 for x in absd if x > 0.01), n_absdiff_gt_0p02=sum(1 for x in absd if x > 0.02))
json.dump(dict(summary=summ, rows=rows), open(OUT, 'w'), indent=1)
print(json.dumps(summ))
for r in rows:
    if r['diff'] is not None and abs(r['diff']) > 0.01:
        print(f"  {r['metric']:55s} pilot {r['pilot']:.4f} server {r['server']:.4f} diff {r['diff']:+.4f}")
