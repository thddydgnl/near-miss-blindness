"""Train / calibrate the success judges on the judge-training set (train split), then freeze them.
(a)      PairJudge on REAL final-frame DINOv2 patch features + goal features (true labels). Primary.
(b)      visual goal distance d_vis = mean (z - z_goal)^2, threshold maximizing balanced accuracy on real frames. Primary.
(b2)     DINO-WM planning objective (visual + proprio-emb MSE, alpha=1), threshold on real frames. Secondary.
(a_imag) PairJudge trained on IMAGINED final features with true labels. Diagnostic only (information content).
Selection rule (PLAN 4.3): epoch with best held-out balanced accuracy; if held-out real accuracy of (a) < 0.90, also
train a larger variant (d=256, 4 layers) and keep the better on held-out balanced accuracy.
usage: python judges12.py"""
import sys, time, json
import numpy as np
import torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import r12lib as R


def per_bin(pred, r):
    out = {}
    for b in ['success', 'near', 'mid', 'far']:
        mk = np.array([R.r_bin(x) == b for x in r])
        out[b] = dict(n=int(mk.sum()), p_judged_success=float(np.mean(pred[mk])) if mk.any() else None)
    return out


def auc(score_pos, score_neg):
    """P(score_pos > score_neg) (Mann-Whitney)."""
    from scipy.stats import mannwhitneyu
    if len(score_pos) == 0 or len(score_neg) == 0:
        return None
    u = mannwhitneyu(score_pos, score_neg, alternative='two-sided').statistic
    return float(u / (len(score_pos) * len(score_neg)))


def main():
    t0 = time.time()
    D = torch.load(f'{R.FEAT}/clf.pt')
    ep = D['ep'].numpy()
    isv = D['ep_is_val'].numpy()[ep]
    r = D['r'].numpy()
    y = (r < 1).astype(np.float32)
    lr = np.log(np.maximum(r, 1e-3))
    zg_all = D['z_goal'][D['ep']]
    pg_all = D['p_goal'][D['ep']]
    tr, va = np.where(~isv)[0], np.where(isv)[0]
    print(f'judge set: {len(r)} seqs, train {len(tr)}, held-out {len(va)}, success rate {y.mean():.3f}', flush=True)
    res = dict(n_train=int(len(tr)), n_val=int(len(va)), success_rate=float(y.mean()))
    nets = {}
    for name, zkey in (('a', 'z_real'), ('a_imag', 'z_imag')):
        Z = D[zkey]
        net, hist = R.train_judge(Z[tr], zg_all[tr], y[tr], lr[tr], Z[va], zg_all[va], y[va], epochs=30, log=print)
        pv = R.judge_prob(net, Z[va], zg_all[va])
        acc = float(((pv > 0.5) == (y[va] > 0.5)).mean())
        variant = 'd128_l2'
        if name == 'a' and acc < 0.90:
            print('held-out acc < 0.90 -> larger variant', flush=True)
            net2, hist2 = R.train_judge(Z[tr], zg_all[tr], y[tr], lr[tr], Z[va], zg_all[va], y[va], epochs=30, d=256,
                                        layers=4, log=print)
            pv2 = R.judge_prob(net2, Z[va], zg_all[va])
            if R.balanced_acc(pv2 > 0.5, y[va]) > R.balanced_acc(pv > 0.5, y[va]):
                net, hist, pv, variant = net2, hist2, pv2, 'd256_l4'
                acc = float(((pv > 0.5) == (y[va] > 0.5)).mean())
        nets[name] = (net, variant)
        res[name] = dict(variant=variant, heldout_acc=acc, heldout_bacc=R.balanced_acc(pv > 0.5, y[va]),
                         heldout_per_bin=per_bin(pv > 0.5, r[va]),
                         heldout_auc_success_vs_near=auc(pv[r[va] < 1], pv[(r[va] >= 1) & (r[va] < 2)]),
                         history=hist)
        print(name, variant, 'held-out acc %.4f bacc %.4f' % (acc, res[name]['heldout_bacc']), flush=True)
    # (a) applied to imagined features of the held-out episodes (judge-set view only; eval sets are in main12)
    pvi = R.judge_prob(nets['a'][0], D['z_imag'][va], zg_all[va])
    res['a']['heldout_on_imagined_per_bin'] = per_bin(pvi > 0.5, r[va])
    # thresholds on real frames (train part), evaluated on held-out
    dv = R.d_vis(D['z_real'], zg_all).numpy()
    df = R.d_full(D['z_real'], zg_all, D['p_real'], pg_all).numpy()
    dvi = R.d_vis(D['z_imag'], zg_all).numpy()
    dfi = R.d_full(D['z_imag'], zg_all, D['p_imag'], pg_all).numpy()
    for name, d, di in (('b', dv, dvi), ('b2', df, dfi)):
        tau, ba_tr = R.best_threshold(d[tr], y[tr])
        res[name] = dict(tau=tau, train_bacc=ba_tr, heldout_acc=float(((d[va] < tau) == (y[va] > 0.5)).mean()),
                         heldout_bacc=R.balanced_acc(d[va] < tau, y[va]), heldout_per_bin=per_bin(d[va] < tau, r[va]),
                         heldout_on_imagined_per_bin=per_bin(di[va] < tau, r[va]),
                         heldout_auc_success_vs_near=auc(-d[va][r[va] < 1], -d[va][(r[va] >= 1) & (r[va] < 2)]))
        print(name, 'tau %.4f held-out acc %.4f bacc %.4f' % (tau, res[name]['heldout_acc'], res[name]['heldout_bacc']),
              flush=True)
    torch.save(dict(a=nets['a'][0].state_dict(), a_variant=nets['a'][1], a_imag=nets['a_imag'][0].state_dict(),
                    a_imag_variant=nets['a_imag'][1], tau_b=res['b']['tau'], tau_b2=res['b2']['tau']),
               f'{R.FEAT}/judges.pt')
    res['time_s'] = time.time() - t0
    R.jdump(res, f'{R.RES}/judges.json')
    print('JUDGES_DONE', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
