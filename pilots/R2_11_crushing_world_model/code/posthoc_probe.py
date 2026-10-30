"""R2-11 POST-HOC (사후) differential probe, written after seeing that the pre-registered level-target probe has a low
oracle ceiling for the box variables (oracle innovation R^2: B 0.12, W 0.10, S 0.03).

Reason: with level targets v(t+1), the error of reading the absolute box position from a frame (~2 px) dominates the
small per-step box change, so even the true next frame barely beats copy. Here the probe reads the CHANGE directly:
targets dv = v(t+1) - v(t) (A: agent displacement, B: box displacement, R: wrapped rotation (deg), C: contact change,
W: wall-gap change, S: box speed), same linear kernel ridge and the same inputs:
  model  [z_hat_{t+1}, z_t]   copy [z_t]   oracle [z_{t+1}, z_t]
Resolution = test R^2 = 1 - MSE / Var(dv). Normalised resolution = (R2_model - R2_copy) / (R2_oracle - R2_copy).
Usage: python posthoc_probe.py --data /workspace/data --ckpt /workspace/ckpt --out /workspace/results
"""
import argparse
import json
import math
import os
import time

import torch

import analyze as A
import env as E
from model import CONFIGS, make_model

DEV = 'cuda'
NPER = 38


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--seeds', default='0,1')
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
    split = 240
    ep = torch.arange(n_ep, device=DEV).repeat_interleave(NPER)
    tt = torch.arange(2, 2 + NPER, device=DEV).repeat(n_ep)
    tr, te = ep < split, ep >= split
    fit = (ep[tr] < 180).nonzero().flatten()
    val = (ep[tr] >= 180).nonzero().flatten()
    epc, tc = ep.cpu(), tt.cpu()
    g = lambda k, dt=0: d[k][epc, tc + dt].double().to(DEV)
    pa0, pa1, pb0, pb1, th0, th1 = g('pa'), g('pa', 1), g('pb'), g('pb', 1), g('th'), g('th', 1)
    dth = th1 - th0
    dth = (dth - (math.pi / 2) * torch.round(dth / (math.pi / 2))) * 180 / math.pi
    con0 = (E.box_gap(pa0, pb0, th0) <= 0.5).double()
    con1 = (E.box_gap(pa1, pb1, th1) <= 0.5).double()
    wg0 = E.wall_gap_box(pb0, th0).clamp(0, 20)
    wg1 = E.wall_gap_box(pb1, th1).clamp(0, 20)
    T = {'A': pa1 - pa0, 'B': pb1 - pb0, 'R': dth[:, None], 'C': (con1 - con0)[:, None], 'W': (wg1 - wg0)[:, None],
         'S': (pb1 - pb0).norm(dim=-1, keepdim=True)}
    names_y = [(v, T[v].shape[1]) for v in A.VARS]
    Y = torch.cat([T[v] for v in A.VARS], 1)
    Ytr, Yte = Y[tr], Y[te]

    def per_var_mse(pred):
        o, c = {}, 0
        for v, k in names_y:
            o[v] = float(((pred[:, c:c + k] - Yte[:, c:c + k]) ** 2).sum(1).mean())
            c += k
        return o
    var_y, c = {}, 0
    for v, k in names_y:
        yy = Yte[:, c:c + k]
        var_y[v] = float(((yy - yy.mean(0)) ** 2).sum(1).mean())
        c += k
    zt = Z[ep, tt].reshape(ep.numel(), -1)
    z1 = Z[ep, tt + 1].reshape(ep.numel(), -1)
    zt_c = zt - zt[tr].float().mean(0).half()
    z1_c = z1 - z1[tr].float().mean(0).half()
    Kt_tr, Kt_te = A.gram(zt_c[tr], zt_c[tr]), A.gram(zt_c[te], zt_c[tr])
    K1_tr, K1_te = A.gram(z1_c[tr], z1_c[tr]), A.gram(z1_c[te], z1_c[tr])
    del zt_c, z1_c
    out = dict(note='POST-HOC differential probe (targets = per-step change)', var_y=var_y,
               frac_nonzero={v: float((T[v][te].abs().sum(1) > 1e-6).double().mean()) for v in A.VARS})
    r2 = lambda m: {v: 1 - m[v] / var_y[v] for v in A.VARS}
    p, _ = A.kridge(Kt_tr, Kt_te, Ytr, fit, val)
    out['copy'] = r2(per_var_mse(p))
    p, _ = A.kridge(Kt_tr + K1_tr, Kt_te + K1_te, Ytr, fit, val)
    out['oracle'] = r2(per_var_mse(p))
    print('copy', {k: round(v, 3) for k, v in out['copy'].items()}, flush=True)
    print('oracle', {k: round(v, 3) for k, v in out['oracle'].items()}, f'{time.time() - t0:.0f}s', flush=True)
    out['models'] = {}
    for s in [int(x) for x in args.seeds.split(',')]:
        for name in CONFIGS:
            fn = f'{args.ckpt}/{name}_s{s}_final.pt'
            if not os.path.exists(fn):
                continue
            model = make_model(name).to(DEV)
            model.load_state_dict(torch.load(fn, map_location=DEV))
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
            p, _ = A.kridge(Kh_tr + Kt_tr, Kh_te + Kt_te, Ytr, fit, val)
            del Kh_tr, Kh_te
            r = r2(per_var_mse(p))
            out['models'][f'{name}_s{s}'] = dict(r2=r, norm={v: (r[v] - out['copy'][v]) / max(1e-9, out['oracle'][v] - out['copy'][v])
                                                           for v in A.VARS})
            print(name, s, {k: round(v, 3) for k, v in r.items()}, f'{time.time() - t0:.0f}s', flush=True)
            json.dump(out, open(f'{args.out}/analysis/posthoc_probe.json', 'w'), indent=1)
            del model
            torch.cuda.empty_cache()
    print('done', flush=True)


if __name__ == '__main__':
    main()
