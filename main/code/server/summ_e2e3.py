"""B E2/E3 (Mac, reads results only): Markdown tables from the E2 / E3 analysis JSONs for the report.
usage: python3 summ_e2e3.py <results/E2_server> <results/E3_server>"""
import json, sys, os, math


def f(x, d=3):
    if x is None:
        return '—'
    if isinstance(x, float) and not math.isfinite(x):
        return '—'
    return f'{x:.{d}f}'


def fc(c, d=3):
    return f'[{f(c[0], d)}, {f(c[1], d)}]' if c else ''


def policy_block(A, title):
    p, c = A['point'], A['ci']
    print(f'\n#### {title}\n')
    print(f"- Spearman(n, o_imag) {f(p['spearman_n_oimag'])} {fc(c['spearman_n_oimag'])}, perm p {A['perm_spearman_n_oimag']['p_one_sided']:.4g} "
          f"({A['perm_spearman_n_oimag']['kind']}), p_boot(<=0) {A['p_boot_spearman_n_oimag_le0']:.4g}")
    print(f"- Spearman(n, o) {f(p['spearman_n_o'])} {fc(c['spearman_n_o'])}, Spearman(n, o_judge) {f(p['spearman_n_ojudge'])} {fc(c['spearman_n_ojudge'])}")
    print(f"- H2b: pairs {f(p['h2b_n_pairs'], 0)} {fc(c['h2b_n_pairs'], 0)}, max imagined gap among similar pairs {f(p['h2b_max_gap_similar'])}, "
          f"tau_sim {f(p['tau_similar'])} {fc(c['tau_similar'])} (similar pairs {f(p.get('n_similar_pairs', float('nan')), 0)}), "
          f"tau_sim real-judge {f(p['tau_similar_realjudge'])}; densest window tau_b {f(A['tau_densest_window']['tau_b'])} "
          f"({len(A['tau_densest_window']['policies'])} policies, ties {A['tau_densest_window']['n_tied_windows']})")
    print(f"- H2c: Pearson(sh, s) {f(p['pearson_sh_s'])} {fc(c['pearson_sh_s'])}, MMRV {f(p['mmrv'])} {fc(c['mmrv'])}; "
          f"FGRE {f(p['fgre'])} {fc(c['fgre'])}; real-judge: Pearson {f(p['pearson_shreal_s'])}, MMRV {f(p['mmrv_realjudge'])}, FGRE {f(p['fgre_realjudge'])}")
    print(f"- mean o {f(p['mean_o'])} {fc(c['mean_o'])}, mean |o| {f(p['mean_abs_o'])}, mean o_imag {f(p['mean_oimag'])} {fc(c['mean_oimag'])}, "
          f"mean o_judge {f(p['mean_ojudge'])} {fc(c['mean_ojudge'])}")
    if A['h2b_pairs']:
        print('- qualifying pairs: ' + '; '.join(f'{a} vs {b}: s {sa:.2f}/{sb:.2f}, sh {ha:.2f}/{hb:.2f}' for a, b, sa, sb, ha, hb in A['h2b_pairs'][:12])
              + (f' ... ({len(A["h2b_pairs"])} total)' if len(A['h2b_pairs']) > 12 else ''))
    print('\n| policy | s | n | sh_real | sh | o | o_judge | o_imag [95%] |')
    print('|---|---|---|---|---|---|---|---|')
    for t in A['table']:
        print(f"| {t['policy']} | {f(t['s'], 2)} | {f(t['n'], 2)} | {f(t['sh_real'], 2)} | {f(t['sh'], 2)} | {f(t['o'], 2)} | {f(t['o_judge'], 2)} | "
              f"{f(t['o_imag'], 2)} {fc(t['ci_o_imag'], 2)} |")


def main():
    r2, r3 = sys.argv[1], sys.argv[2]
    p = f'{r2}/m1/e2_m1_analysis.json'
    if os.path.exists(p):
        A = json.load(open(p))
        for H in (5, 3, 7, 10):
            a = A['dp'][f'H{H}']
            policy_block(a, f'M1 DP H{H}')
            print('- pooled windows:', json.dumps(a['pooled']))
        print('\nVERDICT M1', json.dumps(A['dp']['verdict_M1'], ensure_ascii=False)[:1500])
        if 'ctrl' in A:
            for j in ('a', 'b'):
                policy_block(A['ctrl'][j], f'M1 control (R2-12 main) judge {j}')
    p = f'{r2}/m3/e2_m3_analysis.json'
    if os.path.exists(p):
        M = json.load(open(p))
        for pp in ('0.03', '0.1', '0.3'):
            for bn in ('r29', 'design'):
                policy_block(M[pp][bn], f'M3 p={pp} band {bn}')
            print('- per-seed sh:', json.dumps(M[pp]['per_seed_sh']))
        print('\nVERDICT M3', json.dumps(M['verdict_M3'], ensure_ascii=False)[:1500])
    p = f'{r3}/e3_m1_analysis.json'
    if os.path.exists(p):
        E = json.load(open(p))
        print('\n#### E3 rules\n')
        for k, v in E['rules'].items():
            print(k, json.dumps({kk: vv for kk, vv in v.items()}, ensure_ascii=False))
        print('\n| set | judge | n near | P(S^=1 | near) real [95%] | imag [95%] | share [95%] | P(S^=1 | success) real / imag | AUROC s-vs-near real / imag |')
        print('|---|---|---|---|---|---|---|---|')
        rows = [(f'E1 {t}', j, c) for t, cell in E['e1_cells'].items() for j, c in cell.items()]
        rows += [(f'DP {H}', j, c) for H, cell in E['dp'].items() for j, c in cell.items() if 'p_near_real' in c]
        rows += [(f'M3 p={p_}', 'clf', c) for p_, c in E['m3'].items()]
        for s, j, c in rows:
            print(f"| {s} | {j} | {c['n_near']} | {f(c['p_near_real'])} {fc(c['ci_p_near_real'])} | {f(c['p_near_imag'])} {fc(c['ci_p_near_imag'])} | "
                  f"{f(c['share'])} {fc(c['ci_share'])} | {f(c['p_success_real'])} / {f(c['p_success_imag'])} | "
                  f"{f(c.get('auc_sn_real'))} / {f(c.get('auc_sn_imag'))} |")
        print('\nVERDICT H3', json.dumps(E['verdict_H3'], ensure_ascii=False))
        for H, cell in E['dp'].items():
            if 'd_policy' in cell:
                policy_block(cell['d_policy'], f'M1 DP {H}, judge (d) (secondary, not in the H2 verdict)')


if __name__ == '__main__':
    main()
