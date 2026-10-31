"""B E7 (Mac, reads results only): Markdown tables for reports/E7_runpod.md from results/E7_runpod/pm/e7_analysis.json
(+ judges_e7.json, convcheck_e7.json, calib_e7.json, eval_*_meta.json in the same folder).
usage: python3 summ_e7.py <results/E7_runpod/pm>"""
import json, sys, math, os, glob

D = sys.argv[1]
A = json.load(open(f'{D}/e7_analysis.json'))
C = A['cells']
HS = (3, 5, 7, 10)


def f(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else ('∞' if x > 0 else '−∞')
    return f'{x:.{d}f}'


def fci(c, k, d=3):
    v = c.get('ci', {}).get(k)
    return f'[{f(v[0], d)}, {f(v[1], d)}]' if v else ''


print('## 판정 칸 (val, H = 5, r, 판정기 (a) 앙상블)\n')
P = C['val/H5/r']['a']
print('| 지표 | 실제 최종 프레임 | 상상 최종 프레임 |\n|---|---|---|')
print(f"| 지각 바닥 P(성공 판정 \\| near) | **{f(P['p_near_real'])}** | — |")
print(f"| P(성공 판정 \\| 전문가 r = 0) | {f(P['p_expert_real'])} | {f(P['p_expert_imag'])} |")
print(f"| P(성공 판정 \\| 성공 r < 1) | {f(P['p_success_real'])} | {f(P['p_success_imag'])} {fci(P, 'p_success_imag')} |")
print(f"| P(성공 판정 \\| near 1 ≤ r < 2) | {f(P['p_near_real'])} | **{f(P['p_near_imag'])}** {fci(P, 'p_near_imag')} |")
print(f"| P(성공 판정 \\| mid) | {f(P['p_mid_real'])} | {f(P['p_mid_imag'])} |")
print(f"| P(성공 판정 \\| far r ≥ 3) | {f(P['p_far_real'])} | **{f(P['p_far_imag'])}** {fci(P, 'p_far_imag')} |")
print(f"| near/far(상상) | | {f(P['ratio_near_far_imag'], 1)} |")
print(f"| 실패만(r ≥ 1)의 log r 로지스틱 기울기 | {f(P['slope_fail_real'], 2)} | **{f(P['slope_fail_imag'], 2)}** {fci(P, 'slope_fail_imag', 2)} |")
print(f"| AUROC 성공 대 near | {f(P['auc_sn_real'])} | **{f(P['auc_sn_imag'])}**(probe {f(P.get('auc_sn_ceiling'))}) |")
print(f"| AUROC 성공 대 far | {f(P['auc_sf_real'])} | {f(P['auc_sf_imag'])}(probe {f(P.get('auc_sf_ceiling'))}) |")
print(f"| ΔAUROC(실제 − 상상) 성공 대 near / far | | **{f(P['delta_auc_sn'])}** {fci(P, 'delta_auc_sn')} / **{f(P['delta_auc_sf'])}** {fci(P, 'delta_auc_sf')} |")
print(f"| 양방향 이동: 성공 쪽 P(상상 성공 \\| 성공) − 실제, near 쪽 P(상상 성공 \\| near) − 실제 | | {f(P['succ_shift'])} {fci(P, 'succ_shift')} / {f(P['near_shift'])} {fci(P, 'near_shift')} |")
print(f"| r50(고정 효과 로지스틱) | {f(P['r50_real'], 2)} | {f(P['r50_imag'], 2)} |")
print(f"| r10, Δr10 | {f(P['r10_real'], 2)} | {f(P['r10_imag'], 2)}, Δr10 = {f(P['delta_r10'], 2)} {fci(P, 'delta_r10', 2)} |")
g = P.get('glmm_real', {}); gi = P.get('glmm_imag', {})
print(f"| GLMM(시작 상태 {gi.get('n_groups')}개, 무작위 절편 SD {f(g.get('re_sd'), 2)}~{f(gi.get('re_sd'), 2)}): 기울기, r50, r10 | {f(g.get('b1'), 2)}, {f(g.get('r50'), 2)}, {f(g.get('r10'), 2)} | {f(gi.get('b1'), 2)}, {f(gi.get('r50'), 2)}, {f(gi.get('r10'), 2)} |")
n = P['n']; print(f"\n- n: 성공 {n['success']}, near {n['near']}, mid {n['mid']}, far {n['far']}; 전문가 {P['n_expert']}")
print('- r 구간별 곡선(상상 / 실제): ' + ', '.join(f"[{b['lo']}, {b['hi'] if b['hi'] < 1e8 else '∞'}) {b['p_imag']:.3f} / {b['p_real']:.3f}" for b in P['curve']))
mem = P.get('members', {})
if mem:
    print(f"- 앙상블 구성원 5개의 상상 P(성공 판정 | near) {min(mem['p_near_imag_members']):.3f}~{max(mem['p_near_imag_members']):.3f}(SD {mem['p_near_imag_sd']:.3f}), AUROC {min(mem['auc_sn_imag_members']):.3f}~{max(mem['auc_sn_imag_members']):.3f}")
print(f"- near 구성(패턴): {P.get('near_by_family')}")

for st in ('val', 'trainsplit'):
    for j in ('a', 'b'):
        if not any(f'{st}/H{H}/r' in C for H in HS):
            continue
        print(f'\n## 모든 칸: {st}, 판정기 {j}\n')
        print('| H | 지각 바닥 | P(상상 성공 \\| 성공) | **P(상상 성공 \\| near)** | P(상상 성공 \\| far) | near/far | 실패 기울기 [95%] | AUROC 성공 대 near 실제 / 상상 / probe | ΔAUROC near [95%] | ΔAUROC far [95%] | r50 상상 | Δr10 [95%] | n 성공/near/mid/far |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for H in HS:
            c = C.get(f'{st}/H{H}/r', {}).get(j)
            if not c:
                continue
            ceil = f(c.get('auc_sn_ceiling')) if j == 'a' else '—'
            nn = c.get('n', {})
            print(f"| {H} | {f(c['p_near_real'])} | {f(c['p_success_imag'])} | **{f(c['p_near_imag'])}** {fci(c, 'p_near_imag')} | {f(c['p_far_imag'])} | {f(c['ratio_near_far_imag'], 1)} | "
                  f"{f(c['slope_fail_imag'], 2)} {fci(c, 'slope_fail_imag', 2)} | {f(c['auc_sn_real'])} / {f(c['auc_sn_imag'])} / {ceil} | {f(c['delta_auc_sn'])} {fci(c, 'delta_auc_sn')} | "
                  f"{f(c['delta_auc_sf'])} {fci(c, 'delta_auc_sf')} | {f(c['r50_imag'], 2)} | {f(c['delta_r10'], 2)} {fci(c, 'delta_r10', 2)} | {nn.get('success')}/{nn.get('near')}/{nn.get('mid')}/{nn.get('far')} |")
        hs = C.get(f'{st}/r/horizon_slope_{j}')
        if hs:
            print(f"\n- horizon 기울기({st}, 판정기 {j}): gap(상상 − 실제) H별 {[round(x, 3) for x in hs['gap']]}, 기울기 {f(hs['slope'], 4)} {('[' + f(hs['ci'][0], 4) + ', ' + f(hs['ci'][1], 4) + ']') if hs.get('ci') else ''}, bootstrap에서 기울기 ≥ 0인 비율 {hs.get('p_boot_ge0')}")

V = A.get('verdict_E7')
if V:
    print('\n## 판정(verdict_E7)\n')
    print(json.dumps({k: V[k] for k in ('cell', 'H1b', 'H4b')}, indent=1, ensure_ascii=False))
    print('\n서술:', json.dumps(V['descriptive'], indent=1, ensure_ascii=False))
    print('\n규약 점검:', json.dumps(V.get('convention_check'), ensure_ascii=False))
print('\n## 평가 칸 메타\n')
for p in sorted(glob.glob(f'{D}/eval_*_meta.json')):
    e = json.load(open(p))
    print(f"- {os.path.basename(p)}: 표적 {e['n_targets']}, 버림 {e['n_dropped']}, episode {e['n_episodes']}, 도달 불가 시도 {e['n_unreachable_attempts']}, 시도당 도달률 {e['reach_rate_per_attempt']:.3f}, S_MAX {e['s_max']}, {e['time_s']:.0f}초")
print(f"\n분석 시간 {A.get('time_s', 0):.0f}초, bootstrap {A.get('B')}")
