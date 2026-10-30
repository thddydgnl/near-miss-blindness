"""B E2 M3 analysis (DESIGN_main v1 section 0.3, v2 section 5, v2.2 section 7). Runs in the CPU analysis env (b-ana).
Per p (seeds 0-2 pooled: imagined judgment per scene = mean over the 3 models; primary p = 0.03): policies pi1-pi6 on the
1,024 R2-9 evaluation scenes; s = P(r < 1), n = P(r in band) with the primary band [1, 1.2) and the design band [1, 2);
sh = frozen R2-9 classifier on the imagined final frame, sh_real = the same classifier on the real f4;
o_imag = sh - sh_real. Scene bootstrap B = 10,000 (scenes resampled jointly for policies and seeds). H2a p-value: exact
one-sided permutation test over the 720 policy permutations.
usage: python m3_e2_analyze.py [B]"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import e2stats as S2
from e2_m1_analyze import analyze, boot_weights, bands

RES = os.path.expanduser(os.environ.get('R29_OUT', '~/pmx/B/results/E2/m3'))
PS = ('0.03', '0.1', '0.3')
BANDS = {'r29': (1.0, 1.2), 'design': (1.0, 2.0)}


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    t0 = time.time()
    out = dict(B=B, bands=BANDS)
    W = None
    for p in PS:
        Z = [np.load(f'{RES}/e2_p{p}_std_1x_s{s}.npz') for s in (0, 1, 2)]
        names = [str(x) for x in Z[0]['policies']]
        r, succ, real_logit = Z[0]['r'], Z[0]['succ'], Z[0]['real_logit']
        for z in Z[1:]:
            assert np.array_equal(z['r'], r) and np.array_equal(z['real_logit'], real_logit)
        ji = np.mean([(z['imag_logit'] > 0).astype(float) for z in Z], 0)          # (6, 1024): seed-pooled
        jr = (real_logit > 0).astype(float)
        if W is None:
            W = boot_weights(r.shape[1], B, 20261024)
        Sm = (r < 1).astype(float)
        res = dict(policies=names, per_seed_sh={f's{s}': (Z[s]['imag_logit'] > 0).mean(1).tolist() for s in (0, 1, 2)})
        for bn, (lo, hi) in BANDS.items():
            Nm = bands(r, lo, hi)
            res[bn] = analyze(Sm, Nm, ji, jr, W, names, extra=dict(n_r29=bands(r, 1.0, 1.2).mean(1), n_design=bands(r, 1.0, 2.0).mean(1),
                                                                 far_ge_1p5=(r >= 1.5).mean(1)))
            pt = res[bn]['point']
            print(f"M3 p={p} band {bn}: Spearman(n, o_imag) {pt['spearman_n_oimag']:.3f} {res[bn]['ci']['spearman_n_oimag']} perm p "
                  f"{res[bn]['perm_spearman_n_oimag']['p_one_sided']:.4f} | pairs {pt['h2b_n_pairs']:.0f} tau_sim {pt['tau_similar']} | "
                  f"Pearson {pt['pearson_sh_s']:.3f} MMRV {pt['mmrv']:.3f} FGRE {pt['fgre']:.3f}", flush=True)
        out[p] = res
    P = out['0.03']['r29']['point']
    A = out['0.03']['r29']
    v = dict(cell='p = 0.03, seeds 0-2 pooled, frozen R2-9 classifier, n band [1, 1.2) (v2 section 5)',
             H2a=dict(spearman=P['spearman_n_oimag'], ci=A['ci']['spearman_n_oimag'], ge_0p6=bool(P['spearman_n_oimag'] >= 0.6),
                      p_perm_one_sided=A['perm_spearman_n_oimag']['p_one_sided'], p_boot_le0=A['p_boot_spearman_n_oimag_le0']),
             H2b=dict(n_pairs=P['h2b_n_pairs'], pairs=A['h2b_pairs'], tau_similar=P['tau_similar'],
                      exists_pair=bool(P['h2b_n_pairs'] >= 1), tau_le_0p5=bool(P['tau_similar'] is not None and P['tau_similar'] <= 0.5),
                      tau_densest_window=A['tau_densest_window']),
             H2c=dict(pearson=P['pearson_sh_s'], mmrv=P['mmrv'], pearson_ge_0p8=bool(P['pearson_sh_s'] >= 0.8),
                      mmrv_le_0p05=bool(P['mmrv'] <= 0.05), fgre=P['fgre']),
             design_band=dict(spearman=out['0.03']['design']['point']['spearman_n_oimag'],
                              ci=out['0.03']['design']['ci']['spearman_n_oimag'],
                              p_perm_one_sided=out['0.03']['design']['perm_spearman_n_oimag']['p_one_sided']))
    v['H2a']['holds'] = v['H2a']['ge_0p6']
    v['H2b']['holds'] = bool(v['H2b']['exists_pair'] and v['H2b']['tau_le_0p5'])
    v['H2c']['holds'] = bool(v['H2c']['pearson_ge_0p8'] and v['H2c']['mmrv_le_0p05'])
    out['verdict_M3'] = v
    print('VERDICT M3', json.dumps({k: (vv.get('holds') if isinstance(vv, dict) else vv) for k, vv in v.items()}), flush=True)
    out['time_s'] = time.time() - t0
    with open(f'{RES}/e2_m3_analysis.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('E2_M3_ANALYZE_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
