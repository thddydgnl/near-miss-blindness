"""B E1 (Mac, reads results only): Markdown tables from results/E1_server/m1/e1_m1_analysis.json for the report.
usage: python3 summ_e1_m1.py <analysis json>"""
import json, sys, math

A = json.load(open(sys.argv[1]))
C = A['cells']


def f(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—' if x is None or isinstance(x, float) and math.isnan(x) else ('∞' if x > 0 else '−∞')
    return f'{x:.{d}f}'


def fci(c, k, d=3):
    v = c['ci'].get(k)
    return f'[{f(v[0], d)}, {f(v[1], d)}]' if v else ''


for st in ('val', 'trainsplit'):
    for j in ('a', 'b'):
        print(f'\n### {st}, judge {j}\n')
        print('| H | 정의 | 지각 바닥 P(S~=1 \\| near) | P(S^=1 \\| 성공) | **P(S^=1 \\| near)** | P(S^=1 \\| far) | near/far | 실패 기울기 [95%] | AUROC 성공 대 near 실제 / 상상 / 상한 | ΔAUROC near [95%] | ΔAUROC far [95%] | r50 상상 | Δr10 [95%] | n 성공/near/mid/far |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for d in ('r', 'r_obj'):
            for H in (3, 5, 7, 10):
                c = C.get(f'{st}/H{H}/{d}', {}).get(j)
                if not c:
                    continue
                ceil = f(c.get('auc_sn_ceiling')) if j == 'a' else '—'
                n = c.get('n', {})
                print(f"| {H} | {d} | {f(c['p_near_real'])} | {f(c['p_success_imag'])} | **{f(c['p_near_imag'])}** {fci(c, 'p_near_imag')} | "
                      f"{f(c['p_far_imag'])} | {f(c['ratio_near_far_imag'], 1)} | {f(c['slope_fail_imag'], 2)} {fci(c, 'slope_fail_imag', 2)} | "
                      f"{f(c['auc_sn_real'])} / {f(c['auc_sn_imag'])} / {ceil} | {f(c['delta_auc_sn'])} {fci(c, 'delta_auc_sn')} | "
                      f"{f(c['delta_auc_sf'])} {fci(c, 'delta_auc_sf')} | {f(c['r50_imag'], 2)} | {f(c['delta_r10'], 2)} {fci(c, 'delta_r10', 2)} | "
                      f"{n.get('success')}/{n.get('near')}/{n.get('mid')}/{n.get('far')} |")
        for d in ('r', 'r_obj'):
            hs = C.get(f'{st}/{d}/horizon_slope_{j}')
            if hs:
                print(f"\n- horizon slope ({d}, judge {j}): gap(imag − real) by H {[round(x, 3) for x in hs['gap']]}, slope {f(hs['slope'], 4)} "
                      f"[{f(hs['ci'][0], 4)}, {f(hs['ci'][1], 4)}], p_boot(≥0) {hs.get('p_boot_ge0')}")
print('\n### primary cell details (val/H5/r, judge a)')
c = C['val/H5/r']['a']
print('expert P(S^=1) imag/real', f(c['p_expert_imag']), f(c['p_expert_real']))
print('mid imag/real', f(c['p_mid_imag']), f(c['p_mid_real']))
print('near composition', c.get('near_composition'))
print('near by family', c.get('near_by_family'))
print('members', {k: (([round(x, 3) for x in v]) if isinstance(v, list) else round(v, 4)) for k, v in c.get('members', {}).items()})
print('glmm imag', c.get('glmm_imag'))
print('glmm real', c.get('glmm_real'))
print('r10 imag/real', f(c['r10_imag'], 3), f(c['r10_real'], 3), 'r50 real', f(c['r50_real'], 3))
print('\n### curve (val/H5/r, a)')
for b in c['curve']:
    print(b)
print('\n### pooled grids (sensitivity)')
for k, v in A.get('pooled_grids', {}).items():
    print(k, {kk: (f(vv) if not isinstance(vv, dict) else vv) for kk, vv in v.items()})
print('\n### verdict', json.dumps(A.get('verdict_M1'), ensure_ascii=False))
