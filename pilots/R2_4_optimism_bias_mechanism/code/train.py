"""Train an ensemble of MLP world models in one batched run (one GPU).
Each model: (p, mode, seed, T). mode 'std' = uniform sampling from its finite dataset of N transitions,
mode 'rw' = failures upsampled to 50% of every batch (loss-reweighting baseline).
Manual Adam with a per-model cosine schedule; a model is frozen after its own T steps."""
import argparse
import csv
import json
import math
import os
import time

import torch

from common import (D_IN, D_OUT, PKEYS, binned_lift_prob, forward, init_ensemble, norm_stats, nx, ny,
                    outcome_metrics, raw_y, test_sets, to_py, train_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--ps', default='0.3,0.1,0.03,0.01')
    ap.add_argument('--modes', default='std,rw')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--Ts', default='20000')
    ap.add_argument('--N', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(os.path.join(args.out, args.name), exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)

    ps = [float(x) for x in args.ps.split(',')]
    modes = args.modes.split(',')
    seeds = [int(x) for x in args.seeds.split(',')]
    Ts = [int(x) for x in args.Ts.split(',')]
    cfgs = [dict(p=p, mode=mo, seed=s, T=T, N=args.N) for p in ps for mo in modes for s in seeds for T in Ts]
    M, N, B = len(cfgs), args.N, args.batch
    Tmax = max(Ts)
    print(f'[{args.name}] M={M} N={N} B={B} Tmax={Tmax}', flush=True)

    nm = norm_stats(dev)
    TS = test_sets(dev)
    Xbal = nx(TS['bal']['X'], nm)
    Xnear = nx(TS['near']['X'], nm)

    # ---- data
    cache = {}
    Xn = torch.empty(M, N, D_IN, device=dev)
    Yn = torch.empty(M, N, D_OUT, device=dev)
    fail_pad = torch.zeros(M, N, dtype=torch.long, device=dev)
    succ_pad = torch.zeros(M, N, dtype=torch.long, device=dev)
    n_fail = torch.zeros(M, dtype=torch.long, device=dev)
    n_succ = torch.zeros(M, dtype=torch.long, device=dev)
    data_info = []
    for m, c in enumerate(cfgs):
        key = (c['p'], c['seed'])
        if key not in cache:
            cache[key] = train_data(c['p'], c['seed'], N, dev)
        d = cache[key]
        Xn[m] = nx(d['X'], nm)
        Yn[m] = ny(d['Y'], nm)
        f = (d['outcome'] != 0).nonzero().squeeze(1)
        s = (d['outcome'] == 0).nonzero().squeeze(1)
        fail_pad[m, :len(f)] = f
        succ_pad[m, :len(s)] = s
        n_fail[m], n_succ[m] = len(f), len(s)
        data_info.append(dict(n_fail=len(f), n_miss=int((d['outcome'] == 1).sum()), n_slip=int((d['outcome'] == 2).sum())))
    del cache
    rw = torch.tensor([c['mode'] == 'rw' for c in cfgs], device=dev)
    Tm = torch.tensor([c['T'] for c in cfgs], device=dev, dtype=torch.float32)
    print('data:', data_info[:8], flush=True)

    # ---- params / adam
    P = init_ensemble([c['seed'] for c in cfgs], dev)
    P0 = {k: v.clone().cpu() for k, v in P.items()}
    for k in PKEYS:
        P[k].requires_grad_(True)
    mom = {k: torch.zeros_like(P[k]) for k in PKEYS}
    vel = {k: torch.zeros_like(P[k]) for k in PKEYS}
    b1, b2, eps = 0.9, 0.999, 1e-8

    rows, finals = [], {}
    loss_acc = torch.zeros(M, device=dev)
    n_acc = 0
    t0 = time.time()
    half = B // 2
    for step in range(Tmax):
        # per-model lr: warmup + cosine to 1% of base over the model's own T; 0 afterwards
        s_f = torch.full((M,), float(step), device=dev)
        prog = torch.clamp(s_f / Tm, max=1.0)
        lr_m = args.lr * min(1.0, (step + 1) / args.warmup) * (0.01 + 0.99 * 0.5 * (1 + torch.cos(math.pi * prog)))
        lr_m = torch.where(s_f < Tm, lr_m, torch.zeros_like(lr_m))

        u = torch.rand(M, B, device=dev)
        idx_std = torch.clamp((u * N).long(), max=N - 1)
        i_f = torch.minimum((u[:, :half] * n_fail[:, None]).long(), (n_fail - 1).clamp_min(0)[:, None])
        i_s = torch.minimum((u[:, half:] * n_succ[:, None]).long(), (n_succ - 1)[:, None])
        idx_rw = torch.cat([torch.gather(fail_pad, 1, i_f), torch.gather(succ_pad, 1, i_s)], 1)
        idx = torch.where(rw[:, None], idx_rw, idx_std)
        xb = torch.gather(Xn, 1, idx[..., None].expand(M, B, D_IN))
        yb = torch.gather(Yn, 1, idx[..., None].expand(M, B, D_OUT))
        y = forward(P, xb)
        lossm = ((y - yb) ** 2).mean(dim=(1, 2))
        grads = torch.autograd.grad(lossm.sum(), [P[k] for k in PKEYS])
        with torch.no_grad():
            bc1, bc2 = 1 - b1 ** (step + 1), 1 - b2 ** (step + 1)
            for k, g in zip(PKEYS, grads):
                mom[k].mul_(b1).add_(g, alpha=1 - b1)
                vel[k].mul_(b2).addcmul_(g, g, value=1 - b2)
                lr_b = lr_m.view(-1, *([1] * (g.dim() - 1)))
                P[k].sub_(lr_b * (mom[k] / bc1) / ((vel[k] / bc2).sqrt() + eps))
            loss_acc += lossm.detach()
            n_acc += 1
        if (step + 1) % args.eval_every == 0 or step + 1 == Tmax:
            with torch.no_grad():
                Pd = {k: v.detach() for k, v in P.items()}
                yr = raw_y(forward(Pd, Xbal.unsqueeze(0).expand(M, -1, -1)), nm)
                mb = outcome_metrics(yr, TS['bal'])
                yrn = raw_y(forward(Pd, Xnear.unsqueeze(0).expand(M, -1, -1)), nm)
                mn = outcome_metrics(yrn, TS['near'])
                tl = (loss_acc / n_acc).tolist()
            loss_acc.zero_()
            n_acc = 0
            for m, c in enumerate(cfgs):
                row = dict(step=step + 1, model=m, p=c['p'], mode=c['mode'], seed=c['seed'], T=c['T'], N=N,
                           train_loss=tl[m])
                row.update({f'bal_{k}': float(v[m]) for k, v in mb.items()})
                row.update({f'near_{k}': float(v[m]) for k, v in mn.items()})
                rows.append(row)
                if step + 1 == c['T']:
                    finals[m] = dict(row)
            print(f'step {step + 1} ({time.time() - t0:.0f}s) loss[0]={tl[0]:.2e} '
                  f'opt={[round(float(x), 3) for x in mb["opt"][:8]]}', flush=True)

    # final detailed metrics per model (curve of P(lifted) vs alignment ratio)
    with torch.no_grad():
        Pd = {k: v.detach() for k, v in P.items()}
        yr = raw_y(forward(Pd, Xbal.unsqueeze(0).expand(M, -1, -1)), nm)
        yrn = raw_y(forward(Pd, Xnear.unsqueeze(0).expand(M, -1, -1)), nm)
        edges_bal = [i / 10 for i in range(0, 21)]
        edges_near = [0.8 + i * 0.025 for i in range(17)]
        cb = binned_lift_prob(yr, TS['bal'], edges_bal)
        cn = binned_lift_prob(yrn, TS['near'], edges_near)
    for m in range(M):
        finals[m]['curve_bal'] = to_py(cb[m])
        finals[m]['curve_near'] = to_py(cn[m])
        finals[m].update(data_info[m])
    with open(os.path.join(args.out, args.name, 'train_curves.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, args.name, 'final_metrics.json'), 'w') as f:
        json.dump(dict(args=vars(args), edges_bal=edges_bal, edges_near=edges_near,
                       finals=[finals[m] for m in range(M)]), f, indent=1)
    torch.save(dict(P={k: v.detach().cpu() for k, v in P.items()}, P0=P0, cfgs=cfgs, args=vars(args),
                    norm={k: v.cpu() for k, v in nm.items()}), os.path.join(args.ckpt, f'{args.name}.pt'))
    print(f'[{args.name}] done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
