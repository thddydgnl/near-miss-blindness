"""Post-hoc analyses (사후 분석; decided after seeing the main results, not used for the verdict).
P1 imagination-attributable shift of P(judged success) at near-misses and successes (imagined - real frame), CI
P2 paired flips real-frame judgment -> imagined judgment
P3 per-policy decomposition of the over-estimation: perception (real-frame judge - truth) + imagination (imag - real judge)
P4 imagination-calibrated evaluator: the D3 judge trained on imagined features, used as a policy evaluator
P5 AUROC success-vs-near with CIs
usage: python posthoc12.py"""
import sys, json
import numpy as np
from scipy.stats import spearmanr, kendalltau, mannwhitneyu
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from analyze12 import load, flags, POL, RES

NB = 1000


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None


def wm(x, w):
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else float('nan')


def policy_metrics(pol, succ, bins, flag, w):
    s = np.array([wm(succ[pol == p].astype(float), w[pol == p]) for p in POL])
    sh = np.array([wm(flag[pol == p].astype(float), w[pol == p]) for p in POL])
    n = np.array([wm((bins[pol == p] == 'near').astype(float), w[pol == p]) for p in POL])
    o = sh - s
    best = 0.0
    for i in range(len(POL)):
        for k in range(i + 1, len(POL)):
            if abs(s[i] - s[k]) <= 0.05:
                best = max(best, abs(sh[i] - sh[k]))
    return dict(spearman=float(spearmanr(n, o).correlation), kendall=float(kendalltau(sh, s).correlation),
                max_gap=best, over=o.tolist(), mean_over=float(o.mean()))


def run(name, J):
    recs = load(name)
    if recs is None:
        return None
    F = flags(recs, J)
    g = lambda k: np.array([x[k] for x in recs])
    pol, bins, succ, traj = g('policy'), g('bin'), g('success').astype(bool), g('traj')
    inpol = np.isin(pol, POL)
    nm, sm, fm = inpol & (bins == 'near'), inpol & (bins == 'success'), inpol & (bins == 'far')
    ut = np.unique(traj)
    tix = np.searchsorted(ut, traj)
    rng = np.random.RandomState(777)
    W = [np.ones(len(recs))] + [np.bincount(rng.randint(0, len(ut), len(ut)), minlength=len(ut)).astype(float)[tix]
                                for _ in range(NB)]
    out = dict(set=name)
    # P1 shifts
    for j in ('a', 'b', 'b2'):
        fi, fr = F[f'{j}_imag'].astype(float), F[f'{j}_real'].astype(float)
        vals = {k: [] for k in ['near_imag', 'near_real', 'near_shift', 'succ_imag', 'succ_real', 'succ_shift']}
        for w in W:
            a1, a2 = wm(fi[nm], w[nm]), wm(fr[nm], w[nm])
            b1, b2 = wm(fi[sm], w[sm]), wm(fr[sm], w[sm])
            for k, v in zip(vals, [a1, a2, a1 - a2, b1, b2, b1 - b2]):
                vals[k].append(v)
        out[f'P1_{j}'] = {k: dict(point=v[0], ci=ci(v[1:])) for k, v in vals.items()}
    # P2 flips (paired, same trajectory)
    for j in ('a', 'b'):
        fi, fr = F[f'{j}_imag'], F[f'{j}_real']
        out[f'P2_{j}'] = dict(
            near_realfail_to_imagsucc=float((~fr[nm] & fi[nm]).mean()), near_realsucc_to_imagfail=float((fr[nm] & ~fi[nm]).mean()),
            succ_realsucc_to_imagfail=float((fr[sm] & ~fi[sm]).mean()), succ_realfail_to_imagsucc=float((~fr[sm] & fi[sm]).mean()),
            far_realfail_to_imagsucc=float((~fr[fm] & fi[fm]).mean()))
    # P3 per-policy decomposition
    for j in ('a', 'b'):
        fi, fr = F[f'{j}_imag'].astype(float), F[f'{j}_real'].astype(float)
        rows, n, perc, imag = [], [], [], []
        for p in POL:
            mk = pol == p
            t, rj, im = succ[mk].mean(), fr[mk].mean(), fi[mk].mean()
            rows.append(dict(policy=p, true=float(t), realjudge=float(rj), imag=float(im), perception=float(rj - t),
                             imagination=float(im - rj), total=float(im - t), near=float((bins[mk] == 'near').mean())))
            n.append(rows[-1]['near']); perc.append(rj - t); imag.append(im - rj)
        out[f'P3_{j}'] = dict(rows=rows, spearman_near_perception=float(spearmanr(n, perc).correlation),
                              spearman_near_imagination=float(spearmanr(n, imag).correlation))
    # P4 imagination-calibrated evaluator (judge trained on imagined features)
    vals = [policy_metrics(pol, succ, bins, F['ai_imag'], w) for w in W]
    out['P4_ai'] = dict(point=vals[0], ci={k: ci([v[k] for v in vals[1:]]) for k in ['spearman', 'kendall', 'max_gap', 'mean_over']},
                        p_near=float(F['ai_imag'][nm].mean()), p_far=float(F['ai_imag'][fm].mean()),
                        p_success=float(F['ai_imag'][sm].mean()))
    # P5 AUROC success vs near
    sc = dict(a_real=g('pa_real').astype(float), a_imag=g('pa_imag').astype(float), ai_imag=g('pai_imag').astype(float),
              b_real=-g('dv_real').astype(float), b_imag=-g('dv_imag').astype(float))
    res5 = {}
    rng2 = np.random.RandomState(99)
    for k, s in sc.items():
        bs = []
        for _ in range(200):
            mult = np.bincount(rng2.randint(0, len(ut), len(ut)), minlength=len(ut))
            idx = np.concatenate([np.where(tix == t)[0].repeat(c) for t, c in enumerate(mult) if c > 0])
            sm2, nm2 = idx[sm[idx]], idx[nm[idx]]
            bs.append(auc(s[sm2], s[nm2]))
        res5[k] = dict(point=auc(s[sm], s[nm]), ci=ci(bs))
    out['P5_auc_success_vs_near'] = res5
    return out


def main():
    J = json.load(open(f'{RES}/judges.json'))
    allr = {}
    for name in ('main', 'H10', 'trainsplit'):
        o = run(name, J)
        if o is None:
            continue
        allr[name] = o
        print(f'=== {name}')
        for j in ('a', 'b', 'b2'):
            P = o[f'P1_{j}']
            print(f"  P1[{j}] near: imag {P['near_imag']['point']:.3f} real {P['near_real']['point']:.3f} shift "
                  f"{P['near_shift']['point']:+.3f} CI {np.round(P['near_shift']['ci'], 3).tolist()} | success: imag "
                  f"{P['succ_imag']['point']:.3f} real {P['succ_real']['point']:.3f} shift {P['succ_shift']['point']:+.3f} "
                  f"CI {np.round(P['succ_shift']['ci'], 3).tolist()}")
        for j in ('a', 'b'):
            print(f'  P2[{j}]', {k: round(v, 3) for k, v in o[f'P2_{j}'].items()})
            P3 = o[f'P3_{j}']
            print(f"  P3[{j}] spearman(near, perception) {P3['spearman_near_perception']:.3f} spearman(near, imagination) "
                  f"{P3['spearman_near_imagination']:.3f}")
            for r in P3['rows']:
                print(f"     {r['policy']:7s} near {r['near']:.3f} true {r['true']:.3f} realjudge {r['realjudge']:.3f} imag "
                      f"{r['imag']:.3f} | perception {r['perception']:+.3f} imagination {r['imagination']:+.3f}")
        P4 = o['P4_ai']
        print(f"  P4 ai: p_success {P4['p_success']:.3f} p_near {P4['p_near']:.3f} p_far {P4['p_far']:.3f}; spearman "
              f"{P4['point']['spearman']:.3f} CI {P4['ci']['spearman']}; kendall {P4['point']['kendall']:.3f}; max gap "
              f"{P4['point']['max_gap']:.3f} CI {P4['ci']['max_gap']}; mean over {P4['point']['mean_over']:+.3f}; over "
              f"{np.round(P4['point']['over'], 3).tolist()}")
        print('  P5', {k: (round(v['point'], 3), np.round(v['ci'], 3).tolist()) for k, v in o['P5_auc_success_vs_near'].items()})
    json.dump(allr, open(f'{RES}/posthoc.json', 'w'), indent=1)
    print('POSTHOC_DONE')


if __name__ == '__main__':
    main()
