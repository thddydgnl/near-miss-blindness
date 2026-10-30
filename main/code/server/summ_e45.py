"""B E4/E5 (Mac, reads results only): compact tables from e4_analysis.json / e5_analysis.json for the report.
usage: python3 summ_e45.py <results/E4_server> <results/E5_server>"""
import json, sys, os, math


def f(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    return f'{x:.{d}f}'


def fc(c, d=3):
    return f'[{f(c[0], d)}, {f(c[1], d)}]' if c else ''


def main():
    r4, r5 = sys.argv[1], sys.argv[2]
    p = f'{r4}/e4_analysis.json'
    if os.path.exists(p):
        A = json.load(open(p))
        print('## E4a M1')
        for tag, c in A['4a_M1'].items():
            print(f"\n{tag}: n near {c['n_near']}, success {c['n_success']}, floor NM {f(c['perception_floor_NM'])}, real SM {f(c['perception_SM'])}, orig-vs-E1 mismatches {c.get('orig_vs_e1_mismatches')}")
            for v in ('orig', 'S97', 'S50'):
                print(f"  {v}: NM {f(c[v]['NM'])} {fc(c[v]['ci_NM'])}  SM {f(c[v]['SM'])} {fc(c[v]['ci_SM'])}")
            for k in ('S97_minus_S50', 'S97_minus_orig', 'S50_minus_orig'):
                d = c[k]
                print(f"  {k}: dNM {f(d['NM']['diff'], 4)} {fc(d['NM']['ci'], 4)} (p<=0 {f(d['NM']['p_le0'], 4)}), dSM {f(d['SM']['diff'], 4)} {fc(d['SM']['ci'], 4)} (p>=0 {f(d['SM']['p_ge0'], 4)})")
            print(f"  H4a holds: {c['H4a_holds']}")
        c = A['4a_M3']
        print('\n## E4a M3')
        for v in ('p0.03', 'p0.1', 'p0.3'):
            print(f"  {v}: NM {f(c[v]['NM'])} {fc(c[v]['ci_NM'])}  SM {f(c[v]['SM'], 4)} {fc(c[v]['ci_SM'], 4)}")
        for k in ('p0.03_minus_p0.3', 'p0.03_minus_p0.1', 'p0.1_minus_p0.3'):
            d = c[k]
            print(f"  {k}: dNM {f(d['NM']['diff'], 4)} {fc(d['NM']['ci'], 4)}, dSM {f(d['SM']['diff'], 4)} {fc(d['SM']['ci'], 4)}")
        print(f"  H4a holds: {c['H4a_holds']}")
        print('\n## E4b')
        for k, v in A['4b_M1'].items():
            if k.startswith('val'):
                print(f"  M1 {k}: real {f(v['real'])} ceiling {f(v['imag_ceiling'])} drop {f(v['drop'])} {fc(v['ci_drop'])}")
        print('  ', json.dumps(A['4b_M1']['H4b_M1']))
        for k, v in A['4b_M3'].items():
            if k.startswith('p'):
                print(f"  M3 {k}: real {f(v['real'])} ceiling {f(v['imag_ceiling'])} drop {f(v['drop'])} {fc(v['ci_drop'])}")
        print('  ', json.dumps(A['4b_M3']['H4b_M3']))
        print('\n## sweep')
        for k, v in A['4b_sweep'].items():
            print(f"  {k}: AUROC real/imag/ceil {f(v['auc_sn_real'])}/{f(v['auc_sn_imag'])}/{f(v['auc_sn_ceiling'])} NM {f(v['NM'])} SM {f(v['SM'], 4)} r50 {f(v['r50_imag'])}")
        print('\n## 4c', json.dumps(A.get('4c_M1')), json.dumps(A.get('4c_M3')))
        print('\nVERDICT E4', json.dumps(A['verdict']))
    p = f'{r5}/e5_analysis.json'
    if os.path.exists(p):
        Bj = json.load(open(p))
        print('\n## E5 H5a')
        for k, v in Bj['H5a'].items():
            if k in ('verdict', 'posthoc_r29_band'):
                continue
            print(f"\n{k}:")
            for q in ('ref', 'rw', 'near', 'cp', 'full', 'bw'):
                if q in v:
                    print(f"  {q}: NM {f(v[q]['NM'])} {fc(v[q]['ci_NM'])}  SM {f(v[q]['SM'])} {fc(v[q]['ci_SM'])}")
            for q in ('near_minus_rw', 'near_minus_full', 'near_minus_ref', 'rw_minus_ref', 'full_minus_ref', 'cp_minus_ref', 'bw_minus_ref'):
                if q in v:
                    d = v[q]
                    print(f"  {q}: dNM {f(d['NM']['diff'], 4)} {fc(d['NM']['ci'], 4)}, dSM {f(d['SM']['diff'], 4)} {fc(d['SM']['ci'], 4)}")
            print(f"  criteria {json.dumps(v['criteria'])} -> {v['holds']}")
        print('  verdict', json.dumps(Bj['H5a']['verdict']))
        for k, v in Bj['H5a'].get('posthoc_r29_band', {}).items():
            print(f"  post hoc {k}: " + ' '.join(f"{q} NM {f(v[q]['NM'])} {fc(v[q]['ci_NM'])}" for q in ('ref', 'rw', 'near', 'full', 'cp', 'bw'))
                  + f" | near-rw {f(v['near_minus_rw']['NM']['diff'], 4)} {fc(v['near_minus_rw']['NM']['ci'], 4)} near-full {f(v['near_minus_full']['NM']['diff'], 4)} {fc(v['near_minus_full']['NM']['ci'], 4)}")
        if 'H5b' in Bj:
            h = Bj['H5b']
            print('\n## H5b dev', json.dumps(h['dev']))
            e = h['eval']
            print('eval q*', e['q_star'], 'mean|o|', json.dumps(e['mean_abs_o']), 'diff', f(e['diff_mean_abs_o'], 4), fc(e['ci_diff_mean_abs_o'], 4),
                  'p', e['p_boot_ge0'], 'FGRE', json.dumps(e['fgre']), 'abst', json.dumps(e['abstention']), 'check', e['check_p_det_vs_E2'], '->', e['H5b_holds'])
            for k in ('deterministic', 'mc_discount', 'mc_majority'):
                pt = e['results'][k]['point']
                print(f"  {k}: Spearman(n,o_imag) {f(pt['spearman_n_oimag'])} Pearson {f(pt['pearson_sh_s'])} MMRV {f(pt['mmrv'])} FGRE {f(pt['fgre'])} mean o {f(pt['mean_o'], 4)} mean|o| {f(pt['mean_abs_o'], 4)} tau_sim {f(pt['tau_similar'])} pairs {f(pt['h2b_n_pairs'], 0)}")
        print('\nVERDICT E5', json.dumps(Bj['verdict']))


if __name__ == '__main__':
    main()
