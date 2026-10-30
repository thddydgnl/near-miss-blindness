"""B E1 M1 step 2b: judges, fixed before any evaluation set is imagined (DESIGN_main v1 section 0.2).
 a_full : PairJudge on REAL final-frame features + goal, labels r < 1        (K = 5 seeds; ensemble = mean prob >= 0.5)
 a_obj  : same, labels r_obj < 1                                              (K = 5)
 ai_full, ai_obj: same structure on IMAGINED final features (all four horizons pooled, R2-12 D3) = information ceiling
 (b)    : frozen thresholds tau_full = 0.2125, tau_obj = 0.345 (DESIGN_main); thresholds re-derived on this set are
          reported only.
Held-out = 20% of the judge-set episodes. Use rule: ensemble held-out accuracy on real frames >= 0.95.
usage: python e1_judges.py"""
import os, sys, time, json
import numpy as np
import torch
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R

SEEDS = (0, 1, 2, 3, 4)
TAU = dict(full=0.2125, obj=0.345)
BINS = ('success', 'near', 'mid', 'far')


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def summarize(pv, y, r, H):
    rb = np.array([R.r_bin(x) for x in r])
    pred = pv > 0.5
    out = dict(acc=float((pred == (y > 0.5)).mean()), bacc=R.balanced_acc(pred, y),
               per_bin={b: (float(pred[rb == b].mean()) if (rb == b).any() else None) for b in BINS},
               auc_sn=auc(pv[rb == 'success'], pv[rb == 'near']), auc_sf=auc(pv[rb == 'success'], pv[rb == 'far']))
    out['acc_by_H'] = {int(h): float((pred[H == h] == (y[H == h] > 0.5)).mean()) for h in np.unique(H)}
    return out


def main():
    t0 = time.time()
    D = torch.load(f'{BP.FEAT}/judgeset_e1.pt', weights_only=False)
    ep = D['ep'].numpy()
    isv = D['ep_is_val'].numpy()[ep]
    tr, va = np.where(~isv)[0], np.where(isv)[0]
    H = D['H'].numpy()
    zg = D['z_goal'][D['ep']]
    res = dict(n_train=int(len(tr)), n_val=int(len(va)), seeds=list(SEEDS))
    sds = {}
    for name, zkey, rkey in (('a_full', 'z_real', 'r'), ('a_obj', 'z_real', 'r_obj'),
                             ('ai_full', 'z_imag', 'r'), ('ai_obj', 'z_imag', 'r_obj')):
        r = D[rkey].numpy()
        y = (r < 1).astype(np.float32)
        lr = np.log(np.maximum(r, 1e-3))
        Z = D[zkey]
        members, probs = [], []
        for sd in SEEDS:
            net, hist = R.train_judge(Z[tr], zg[tr], y[tr], lr[tr], Z[va], zg[va], y[va], epochs=30, seed=sd,
                                      log=lambda *a, **k: None)
            pv = R.judge_prob(net, Z[va], zg[va])
            members.append(dict(seed=sd, best_epoch=int(np.argmax([h['val_bacc'] for h in hist])), **summarize(pv, y[va], r[va], H[va])))
            probs.append(pv)
            sds.setdefault(name, []).append({k: v.detach().cpu() for k, v in net.state_dict().items()})
            print(f"{name} seed {sd}: held-out acc {members[-1]['acc']:.4f} bacc {members[-1]['bacc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        P = np.stack(probs)
        ens = summarize(P.mean(0), y[va], r[va], H[va])
        ens['member_prob_sd_mean'] = float(P.std(0, ddof=1).mean())
        ens['passes_0p95'] = bool(ens['acc'] >= 0.95)
        res[name] = dict(members=members, ensemble=ens, label=rkey, features=zkey)
        print(f"{name} ENSEMBLE held-out acc {ens['acc']:.4f} bacc {ens['bacc']:.4f} pass>=0.95 {ens['passes_0p95']}", flush=True)
    # (b): frozen thresholds; re-derived thresholds for reference only
    dv = torch.cat([R.d_vis(D['z_real'][i:i + 2000], zg[i:i + 2000]) for i in range(0, len(zg), 2000)]).numpy()
    for key, rkey in (('full', 'r'), ('obj', 'r_obj')):
        r = D[rkey].numpy()
        y = r < 1
        tau_re, ba_re = R.best_threshold(dv[tr], y[tr])
        pred = dv[va] < TAU[key]
        rb = np.array([R.r_bin(x) for x in r[va]])
        res[f'b_{key}'] = dict(tau_frozen=TAU[key], tau_rederived_reference=tau_re, train_bacc_rederived=ba_re,
                               heldout=dict(acc=float((pred == y[va]).mean()), bacc=R.balanced_acc(pred, y[va]),
                                            per_bin={b: (float(pred[rb == b].mean()) if (rb == b).any() else None) for b in BINS},
                                            auc_sn=auc(-dv[va][rb == 'success'], -dv[va][rb == 'near']),
                                            auc_sf=auc(-dv[va][rb == 'success'], -dv[va][rb == 'far'])))
        print(f"b_{key}: frozen tau {TAU[key]} held-out acc {res[f'b_{key}']['heldout']['acc']:.4f}; re-derived {tau_re:.4f}", flush=True)
    torch.save(dict(sds=sds, tau=TAU, seeds=SEEDS), f'{BP.FEAT}/judges_e1.pt')
    res['time_s'] = time.time() - t0
    R.jdump(res, f'{BP.RES}/judges_e1.json')
    print('JUDGES_E1_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
