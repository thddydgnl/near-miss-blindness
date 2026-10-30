"""B E0 diagnostic: realization variance of the oracle discriminator (a) (R2-12 PairJudge, same training set and rule).
The server reproduction gave the same held-out accuracy as the pilot but a different operating point on the evaluation
sets (imagined features identical within GPU noise). Here (a) is retrained with several seeds (seed 0 twice, to test
same-GPU determinism) and re-applied to the stored evaluation features (no new imagination).
usage: python judge_seeds_e0.py [seeds=0,0,1,2,3]"""
import sys, os, json, time, glob
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau, mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
from analyze12 import POL


def auc(pos, neg):
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def load_eval(tag):
    recs = [json.loads(l) for l in open(f'{BP.RES}/traj_{tag}.jsonl')]
    fs = sorted(glob.glob(f'{BP.FEAT}/eval_{tag}_*.pt'), key=lambda p: int(p.rsplit('_', 1)[1][:-3]))
    zr, zi, zg = [], [], []
    for f in fs:
        d = torch.load(f, weights_only=False)
        n_ep = d['zg'].shape[0]
        nS = d['zr'].shape[0] // n_ep
        zr.append(d['zr']); zi.append(d['zi']); zg.append(d['zg'].repeat_interleave(nS, 0))
    return recs, torch.cat(zr), torch.cat(zi), torch.cat(zg)


def metrics(recs, pr, pi):
    pol = np.array([x['policy'] for x in recs]); b = np.array([x['bin'] for x in recs])
    succ = np.array([x['success'] for x in recs])
    inpol = np.isin(pol, POL)
    sm, nm, fm = inpol & (b == 'success'), inpol & (b == 'near'), inpol & (b == 'far')
    out = dict(p_near_real=float((pr[nm] > 0.5).mean()), p_near_imag=float((pi[nm] > 0.5).mean()),
               p_succ_real=float((pr[sm] > 0.5).mean()), p_succ_imag=float((pi[sm] > 0.5).mean()),
               p_far_imag=float((pi[fm] > 0.5).mean()),
               auc_sn_real=auc(pr[sm], pr[nm]), auc_sn_imag=auc(pi[sm], pi[nm]), auc_sf_imag=auc(pi[sm], pi[fm]))
    s = np.array([succ[pol == p].mean() for p in POL]); sh = np.array([(pi[pol == p] > 0.5).mean() for p in POL])
    shr = np.array([(pr[pol == p] > 0.5).mean() for p in POL])
    n = np.array([(b[pol == p] == 'near').mean() for p in POL])
    out.update(spearman_near_over=float(spearmanr(n, sh - s).correlation), kendall=float(kendalltau(sh, s).correlation),
               mean_over=float((sh - s).mean()), spearman_near_imagination=float(spearmanr(n, sh - shr).correlation))
    return out


def main():
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,0,1,2,3').split(',')]
    t0 = time.time()
    D = torch.load(f'{BP.FEAT}/clf.pt', weights_only=False)
    ep = D['ep'].numpy(); isv = D['ep_is_val'].numpy()[ep]
    r = D['r'].numpy(); y = (r < 1).astype(np.float32); lr = np.log(np.maximum(r, 1e-3))
    zg_all = D['z_goal'][D['ep']]
    tr, va = np.where(~isv)[0], np.where(isv)[0]
    sets = {t: load_eval(t) for t in ('main', 'H10', 'trainsplit')}
    print(f'loaded ({time.time() - t0:.0f}s)', flush=True)
    res = []
    for k, sd in enumerate(seeds):
        net, hist = R.train_judge(D['z_real'][tr], zg_all[tr], y[tr], lr[tr], D['z_real'][va], zg_all[va], y[va],
                                  epochs=30, seed=sd, log=lambda *a, **kw: None)
        pv = R.judge_prob(net, D['z_real'][va], zg_all[va])
        row = dict(run=k, seed=sd, heldout_acc=float(((pv > 0.5) == (y[va] > 0.5)).mean()),
                   heldout_bacc=R.balanced_acc(pv > 0.5, y[va]), best_epoch=int(np.argmax([h['val_bacc'] for h in hist])))
        for t, (recs, zr, zi, zg) in sets.items():
            row[t] = metrics(recs, R.judge_prob(net, zr, zg), R.judge_prob(net, zi, zg))
        res.append(row)
        print(json.dumps(row), flush=True)
    summ = {}
    for t in sets:
        for key in res[0][t]:
            v = np.array([x[t][key] for x in res])
            summ[f'{t}:{key}'] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()))
    R.jdump(dict(runs=res, summary=summ, time_s=time.time() - t0), f'{BP.RES}/judge_seeds_e0.json')
    print('JUDGE_SEEDS_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
