"""B E3 M1 judge (d) rule (DESIGN_main v2.1 section 6, v2.2 section 7), computed from the VLM scores of the rule sets
before any evaluation set is judged. Per rule set: AUROC(success vs failure) of P(Yes) on the held-out real (decoded)
frames; criterion AUROC >= 0.80 -> (d) is a judge for that set, otherwise auxiliary (scores reported only). The Yes
threshold maximises the balanced accuracy of (score > threshold) on the training rows (same rule as M2's judge (d)).
Rule sets: dset_r (dset goals, label r < 1), dset_robj (dset goals, label r_obj < 1), std (standard goal, r_obj < 1).
Diagnostics: AUROC success vs near / far, the held-out rows without the expert sequences (dset), P(judged success | band).
Runs in the CPU analysis env. usage: python e3_d_rules.py"""
import os, sys, json
import numpy as np

FEAT = os.path.expanduser('~/pmx/B/feat/E3/m1')
VLM = os.path.expanduser('~/pmx/B/results/E3/m1/vlm')
OUT = os.path.expanduser('~/pmx/B/results/E3/m1')


def auroc(sp, sn):
    sp, sn = np.asarray(sp, float), np.asarray(sn, float)
    if len(sp) == 0 or len(sn) == 0:
        return None
    return float(((sp[:, None] > sn[None]).sum() + 0.5 * (sp[:, None] == sn[None]).sum()) / (len(sp) * len(sn)))


def rule(sc, r, split, extra_mask=None):
    y = r < 1
    tr, he = split == 'train', split == 'held'
    best = max(((0.5 * ((sc[tr & y] > th).mean() + (sc[tr & ~y] <= th).mean()), th) for th in np.unique(sc[tr])))
    th = float(best[1])
    near, far = (r >= 1) & (r < 2), r >= 3
    j = sc > th
    out = dict(n_held=int(he.sum()), n_train=int(tr.sum()), held_success_frac=float(y[he].mean()),
               auroc_held=auroc(sc[he & y], sc[he & ~y]), auroc_held_success_vs_near=auroc(sc[he & y], sc[he & near]),
               auroc_held_success_vs_far=auroc(sc[he & y], sc[he & far]), auroc_train=auroc(sc[tr & y], sc[tr & ~y]),
               threshold=th, train_balanced_acc=float(best[0]),
               held_balanced_acc=float(0.5 * ((sc[he & y] > th).mean() + (sc[he & ~y] <= th).mean())),
               held_acc=float((j[he] == y[he]).mean()),
               held_p_judged_success={b: (float(j[he & m].mean()) if (he & m).any() else None)
                                      for b, m in (('success', y), ('near', near), ('mid', (r >= 2) & (r < 3)), ('far', far))},
               held_mean_score={b: (float(sc[he & m].mean()) if (he & m).any() else None)
                                for b, m in (('success', y), ('near', near), ('far', far))})
    if extra_mask is not None:
        he2 = he & extra_mask
        out['auroc_held_without_expert'] = auroc(sc[he2 & y], sc[he2 & ~y])
        out['n_held_without_expert'] = int(he2.sum())
    out['passes_0p80'] = bool(out['auroc_held'] is not None and out['auroc_held'] >= 0.80)
    if extra_mask is not None:
        out['passes_0p80_without_expert'] = bool(out['auroc_held_without_expert'] is not None and out['auroc_held_without_expert'] >= 0.80)
    return out


def main():
    res = {}
    Dd = np.load(f'{FEAT}/rules_dset.npz')
    split, fam = Dd['split'], Dd['fam']
    for v, lab in (('dset_r', 'r'), ('dset_robj', 'r_obj')):
        f = f'{VLM}/rules_{v}_final.npy'
        if os.path.exists(f):
            res[v] = rule(np.load(f), Dd[lab], split, extra_mask=(fam != 'expert'))
    Ds = np.load(f'{FEAT}/rules_std.npz')
    f = f'{VLM}/rules_std_final.npy'
    if os.path.exists(f):
        res['std'] = rule(np.load(f), Ds['r_obj'], Ds['split'])
    res['criterion'] = 'held AUROC(success vs failure) >= 0.80 on real decoded frames (DESIGN_main v2.1 section 6)'
    with open(f'{OUT}/d_rules.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, json.dumps({kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}))
    print('E3_D_RULES_DONE')


if __name__ == '__main__':
    main()
