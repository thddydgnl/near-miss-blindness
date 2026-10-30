"""B E0 diagnostic: judges for the object-pose-only PushT definition (r_obj) on the R2-12 judge-training set (same
episodes / features as judges12.py, labels r_obj < 1 instead of r < 1). Checks whether the oracle discriminator reaches
the design's held-out accuracy >= 0.95 under r_obj, the perception floor of the goal-distance judge (b_obj), and
applies both to the stored R2-12 evaluation features (descriptive only; E0 is not a hypothesis test).
usage: python judges_obj_e0.py"""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
from judge_seeds_e0 import load_eval, auc
from analyze12 import POL

BINS = ['success', 'near', 'mid', 'far']


def per_bin(flag, rb):
    return {b: (float(flag[rb == b].mean()) if (rb == b).any() else None) for b in BINS}


def main():
    t0 = time.time()
    D = torch.load(f'{BP.FEAT}/clf.pt', weights_only=False)
    ep = D['ep'].numpy(); isv = D['ep_is_val'].numpy()[ep]
    ro = D['r_obj'].numpy(); y = (ro < 1).astype(np.float32); lr = np.log(np.maximum(ro, 1e-3))
    zg_all = D['z_goal'][D['ep']]
    tr, va = np.where(~isv)[0], np.where(isv)[0]
    out = dict(n_train=int(len(tr)), n_val=int(len(va)), success_rate_obj=float(y.mean()),
               bins_obj={b: float(np.mean([R.r_bin(x) == b for x in ro])) for b in BINS})
    net, hist = R.train_judge(D['z_real'][tr], zg_all[tr], y[tr], lr[tr], D['z_real'][va], zg_all[va], y[va], epochs=30,
                              seed=0, log=lambda *a, **k: None)
    pv = R.judge_prob(net, D['z_real'][va], zg_all[va])
    rb_va = np.array([R.r_bin(x) for x in ro[va]])
    out['a_obj'] = dict(heldout_acc=float(((pv > 0.5) == (y[va] > 0.5)).mean()), heldout_bacc=R.balanced_acc(pv > 0.5, y[va]),
                        heldout_per_bin=per_bin(pv > 0.5, rb_va), passes_0p95=bool(((pv > 0.5) == (y[va] > 0.5)).mean() >= 0.95))
    dv = R.d_vis(D['z_real'], zg_all).numpy()
    tau, ba = R.best_threshold(dv[tr], y[tr])
    out['b_obj'] = dict(tau=tau, train_bacc=ba, heldout_acc=float(((dv[va] < tau) == (y[va] > 0.5)).mean()),
                        heldout_per_bin=per_bin(dv[va] < tau, rb_va))
    print(json.dumps(out), flush=True)
    for t in ('main', 'H10', 'trainsplit'):
        recs, zr, zi, zg = load_eval(t)
        pol = np.array([x['policy'] for x in recs]); inpol = np.isin(pol, POL)
        rb = np.array([x['bin_obj'] for x in recs])
        pr, pi = R.judge_prob(net, zr, zg), R.judge_prob(net, zi, zg)
        dr, di = R.d_vis(zr, zg).numpy(), R.d_vis(zi, zg).numpy()
        sm, nm, fm = inpol & (rb == 'success'), inpol & (rb == 'near'), inpol & (rb == 'far')
        out[t] = dict(n_bin_obj={b: int((inpol & (rb == b)).sum()) for b in BINS},
                      a_obj_real=per_bin(pr[inpol] > 0.5, rb[inpol]), a_obj_imag=per_bin(pi[inpol] > 0.5, rb[inpol]),
                      b_obj_real=per_bin(dr[inpol] < tau, rb[inpol]), b_obj_imag=per_bin(di[inpol] < tau, rb[inpol]),
                      auc_sn_a_real=auc(pr[sm], pr[nm]), auc_sn_a_imag=auc(pi[sm], pi[nm]),
                      auc_sf_a_real=auc(pr[sm], pr[fm]), auc_sf_a_imag=auc(pi[sm], pi[fm]),
                      auc_sn_b_real=auc(-dr[sm], -dr[nm]), auc_sn_b_imag=auc(-di[sm], -di[nm]))
        print(t, json.dumps(out[t]), flush=True)
    out['time_s'] = time.time() - t0
    R.jdump(out, f'{BP.RES}/judges_obj_e0.json')
    print('JUDGES_OBJ_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
