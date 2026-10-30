"""R2-11 POST-HOC (사후) positive control for the confusion index: how large a planted packing-like interference
would the pre-registered v2 confusion metric detect?

For w256 and the d=8 models (seed 0), the level-target probe errors are recomputed exactly as in analyze.py; then a
synthetic dependence on an unrelated variable is planted into the agent-position error,
    e_A' = e_A + c * sd(e_A) * h(theta_{t+1}),   h = standardised sin(4 theta) (box angle, unrelated to A),
and into the box-angle error, e_R' = e_R + c * sd(e_R) * h(pa_{t+1}) (standardised agent x).
The CI (v2) and its episode-bootstrap 95% interval are recomputed for c in {0, 0.1, 0.2, 0.3, 0.5}.
Usage: python posthoc_power.py --data /workspace/data --ckpt /workspace/ckpt --out /workspace/results
"""
import argparse
import json
import math
import os
import time

import torch

import analyze as A
from model import make_model

DEV = 'cuda'
NPER = 38


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--n_boot', type=int, default=100)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    st = torch.load(f'{args.data}/stats.pt')
    mu, sd = st['mu'].to(DEV), st['sd'].to(DEV)
    d = torch.load(f'{args.data}/ana.pt')
    n_ep = d['Z'].shape[0]
    Z = torch.empty(d['Z'].shape, dtype=torch.float16, device=DEV)
    for i in range(0, n_ep, 50):
        Z[i:i + 50] = ((d['Z'][i:i + 50].to(DEV).float() - mu) / sd).half()
    Aact = d['a'].to(DEV).float()
    ep = torch.arange(n_ep, device=DEV).repeat_interleave(NPER)
    tt = torch.arange(2, 2 + NPER, device=DEV).repeat(n_ep)
    tr, te = ep < 240, ep >= 240
    fit = (ep[tr] < 180).nonzero().flatten()
    val = (ep[tr] >= 180).nonzero().flatten()
    Yd, basis, fam, own2, free = A.gt_tables(d, ep.cpu(), tt.cpu())
    folds = (ep[te] % 2).to(DEV)
    names_y = [(v, Yd[v].shape[1]) for v in A.VARS]
    Ycat = torch.cat([Yd[v] for v in A.VARS], 1)
    Ytr, Yte = Ycat[tr], Ycat[te]
    basis_te = {v: A.std_cols(basis[v][te]) for v in A.VARS}
    own2_te = {v: A.std_cols(own2[v][te]) for v in own2}
    free_te = free[te]
    th1 = d['th'][ep.cpu(), tt.cpu() + 1].double().to(DEV)[te]
    hx = A.std_cols(torch.sin(4 * th1)[:, None])
    pax = A.std_cols(Yd['A'][te][:, :1])
    te_eps = ep[te].unique()
    ep_te = ep[te]
    pos = {int(e): (ep_te == e).nonzero().flatten() for e in te_eps.tolist()}
    gb = torch.Generator(device='cpu')
    gb.manual_seed(777)
    boots = []
    for b in range(args.n_boot):
        pick = te_eps[torch.randint(te_eps.numel(), (te_eps.numel(),), generator=gb).to(DEV)]
        boots.append(torch.cat([pos[int(e)] for e in pick.tolist()]))
    zt = Z[ep, tt].reshape(ep.numel(), -1)
    zt_c = zt - zt[tr].float().mean(0).half()
    Kt_tr, Kt_te = A.gram(zt_c[tr], zt_c[tr]), A.gram(zt_c[te], zt_c[tr])
    del zt_c
    out = dict(note='POST-HOC positive control: planted dependence of e_A on box angle and of e_R on agent x', runs={})
    for name in ('w256', 'mlp8', 'res8'):
        model = make_model(name).to(DEV)
        model.load_state_dict(torch.load(f'{args.ckpt}/{name}_s0_final.pt', map_location=DEV))
        model.eval()
        zh = torch.empty_like(zt)
        with torch.no_grad():
            for i in range(0, ep.numel(), 500):
                e, t_ = ep[i:i + 500], tt[i:i + 500]
                fr = t_[:, None] + torch.arange(-2, 1, device=DEV)[None]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pz = model(Z[e[:, None], fr].float(), Aact[e, t_])
                zh[i:i + 500] = pz.float().reshape(pz.shape[0], -1).half()
        zh_c = zh - zh[tr].float().mean(0).half()
        Kh_tr, Kh_te = A.gram(zh_c[tr], zh_c[tr]), A.gram(zh_c[te], zh_c[tr])
        del zh, zh_c
        pred, _ = A.kridge(Kh_tr + Kt_tr, Kh_te + Kt_te, Ytr, fit, val)
        del Kh_tr, Kh_te
        err, c0 = {}, 0
        for v, k in names_y:
            err[v] = pred[:, c0:c0 + k] - Yte[:, c0:c0 + k]
            c0 += k
        res = {}
        for c in (0.0, 0.1, 0.2, 0.3, 0.5):
            e2 = dict(err)
            e2['A'] = err['A'] + c * err['A'].std(0, keepdim=True) * hx
            e2['R'] = err['R'] + c * err['R'].std(0, keepdim=True) * pax
            ci = A.confusion2(e2, basis_te, own2_te, free_te, folds)
            bs = sorted(A.confusion2(e2, basis_te, own2_te, free_te, folds, idx=bi)['CI'] for bi in boots)
            res[str(c)] = dict(CI=ci['CI'], A=ci['A']['unrel'], R=ci['R']['unrel'], W=ci['W']['unrel'],
                               lo=bs[int(0.025 * len(bs))], hi=bs[int(0.975 * len(bs)) - 1])
            print(name, c, {k: round(v, 4) for k, v in res[str(c)].items()}, f'{time.time() - t0:.0f}s', flush=True)
        out['runs'][name] = res
        json.dump(out, open(f'{args.out}/analysis/posthoc_power.json', 'w'), indent=1)
        del model
        torch.cuda.empty_cache()
    print('done', flush=True)


if __name__ == '__main__':
    main()
