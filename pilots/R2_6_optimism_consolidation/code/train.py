"""Train an ensemble of MLP world models in one batched run (R2-6).
version 'state': 12-dim state input (R2-4).  version 'visual': frozen-DINOv2 PCA patch features of the rendered
initial scene (8192) + action (2); object-relative targets.
mode 'std' = uniform sampling; 'rw' = failures upsampled to 50% of each batch (uniform reweighting);
'hn' = near-miss hard-example resampling: 50% successes, 25% failures (uniform), 25% near-miss failures (miss, r<1.2).
Manual Adam with a per-model cosine schedule over its own T. Optional parameter snapshots for time series."""
import argparse
import csv
import json
import math
import os
import time

import torch

from common import (D_IN, D_OUT, EDGES_BAL, EDGES_NEAR, PKEYS, binned_lift_prob, forward, init_ensemble, norm_stats,
                    nx, ny, outcome_metrics, raw_y, rel_y, test_sets, to_py, train_data)


def load_visual(seeds, N, dev):
    from visual import FEAT_DIR
    banks, offs, ranks, o = [], {}, {}, 0
    for s in seeds:
        b = torch.load(os.path.join(FEAT_DIR, f'bank_s{s}_N{N}.pt'))
        banks.append(b['feat'])
        offs[s] = o
        ranks[s] = b['slip_rank'].to(dev)
        o += b['feat'].shape[0]
    bank = torch.cat(banks).to(dev)
    nm = {k: v.to(dev) for k, v in torch.load(os.path.join(FEAT_DIR, 'norm_visual.pt')).items()}
    VT = torch.load(os.path.join(FEAT_DIR, 'test_visual.pt'))
    TS = {k: {kk: vv.to(dev) for kk, vv in VT[k].items()} for k in ('bal', 'near')}
    return bank, offs, ranks, nm, TS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--version', default='state', choices=['state', 'visual'])
    ap.add_argument('--ps', default='0.3,0.03,0.01')
    ap.add_argument('--modes', default='std,rw,hn')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--T', type=int, default=10000)
    ap.add_argument('--N', type=int, default=100000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--snap_every', type=int, default=0)
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    od = os.path.join(args.out, args.name)
    os.makedirs(od, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    ps = [float(x) for x in args.ps.split(',')]
    modes = args.modes.split(',')
    seeds = [int(x) for x in args.seeds.split(',')]
    cfgs = [dict(p=p, mode=mo, seed=s, T=args.T, N=args.N, version=args.version) for p in ps for mo in modes for s in seeds]
    M, N, B, T = len(cfgs), args.N, args.batch, args.T
    vis = args.version == 'visual'
    print(f'[{args.name}] {args.version} M={M} N={N} B={B} T={T}', flush=True)

    if vis:
        bank, offs, ranks, nm, TS = load_visual(sorted(set(seeds)), N, dev)
        d_in = bank.shape[1] + 2
        xm_f, xs_f = nm['xm'][:-2], nm['xs'][:-2]
        img_map = torch.empty(M, N, dtype=torch.long, device=dev)
        An = torch.empty(M, N, 2, device=dev)
    else:
        nm = norm_stats(dev)
        TS = test_sets(dev)
        d_in = D_IN
        Xn = torch.empty(M, N, D_IN, device=dev)
    Xbal, Xnear = nx(TS['bal']['X'], nm), nx(TS['near']['X'], nm)
    Yn = torch.empty(M, N, D_OUT, device=dev)
    pads = {k: torch.zeros(M, N, dtype=torch.long, device=dev) for k in ('f', 's', 'nm')}
    cnt = {k: torch.zeros(M, dtype=torch.long, device=dev) for k in ('f', 's', 'nm')}
    cache, data_info = {}, []
    for m, c in enumerate(cfgs):
        key = (c['p'], c['seed'])
        if key not in cache:
            cache[key] = train_data(c['p'], c['seed'], N, dev)
        d = cache[key]
        if vis:
            col5 = d['X'][:, 11] > 0.5
            idx = torch.arange(N, device=dev)
            img_map[m] = offs[c['seed']] + torch.where(col5, N + ranks[c['seed']].clamp_min(0), idx)
            An[m] = (d['X'][:, 4:6] - nm['xm'][-2:]) / nm['xs'][-2:]
            Yn[m] = ny(rel_y(d['Y'], d['X']), nm)
        else:
            Xn[m] = nx(d['X'], nm)
            Yn[m] = ny(d['Y'], nm)
        sets = {'f': d['outcome'] != 0, 's': d['outcome'] == 0, 'nm': (d['outcome'] == 1) & (d['r'] < 1.2)}
        for k, msk in sets.items():
            ii = msk.nonzero().squeeze(1)
            pads[k][m, :len(ii)] = ii
            cnt[k][m] = len(ii)
        data_info.append(dict(n_fail=int(cnt['f'][m]), n_miss=int((d['outcome'] == 1).sum()),
                              n_slip=int((d['outcome'] == 2).sum()), n_nearmiss=int(cnt['nm'][m])))
    del cache
    mode_id = torch.tensor([{'std': 0, 'rw': 1, 'hn': 2}[c['mode']] for c in cfgs], device=dev)
    print('data:', data_info[:6], flush=True)

    P = init_ensemble([c['seed'] for c in cfgs], dev, d_in)
    P0 = {k: v.clone().cpu() for k, v in P.items()}
    for k in PKEYS:
        P[k].requires_grad_(True)
    mom = {k: torch.zeros_like(P[k]) for k in PKEYS}
    vel = {k: torch.zeros_like(P[k]) for k in PKEYS}
    b1, b2, eps = 0.9, 0.999, 1e-8

    def pick(k, u):
        i = torch.minimum((u * cnt[k][:, None]).long(), (cnt[k] - 1).clamp_min(0)[:, None])
        return torch.gather(pads[k], 1, i)

    rows, snaps = [], []
    loss_acc, n_acc = torch.zeros(M, device=dev), 0
    t0 = time.time()
    h2, h4 = B // 2, B // 4
    for step in range(T):
        prog = step / T
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog)))
        u = torch.rand(M, B, device=dev)
        idx_std = torch.clamp((u * N).long(), max=N - 1)
        idx_rw = torch.cat([pick('f', u[:, :h2]), pick('s', u[:, h2:])], 1)
        idx_hn = torch.cat([pick('nm', u[:, :h4]), pick('f', u[:, h4:h2]), pick('s', u[:, h2:])], 1)
        idx = torch.where((mode_id == 0)[:, None], idx_std, torch.where((mode_id == 1)[:, None], idx_rw, idx_hn))
        if vis:
            im = torch.gather(img_map, 1, idx)
            xb = torch.cat([(bank[im].float() - xm_f) / xs_f, torch.gather(An, 1, idx[..., None].expand(M, B, 2))], -1)
        else:
            xb = torch.gather(Xn, 1, idx[..., None].expand(M, B, D_IN))
        yb = torch.gather(Yn, 1, idx[..., None].expand(M, B, D_OUT))
        lossm = ((forward(P, xb) - yb) ** 2).mean(dim=(1, 2))
        grads = torch.autograd.grad(lossm.sum(), [P[k] for k in PKEYS])
        with torch.no_grad():
            bc1, bc2 = 1 - b1 ** (step + 1), 1 - b2 ** (step + 1)
            for k, g in zip(PKEYS, grads):
                mom[k].mul_(b1).add_(g, alpha=1 - b1)
                vel[k].mul_(b2).addcmul_(g, g, value=1 - b2)
                P[k].sub_(lr * (mom[k] / bc1) / ((vel[k] / bc2).sqrt() + eps))
            loss_acc += lossm.detach()
            n_acc += 1
        if args.snap_every and (step + 1) % args.snap_every == 0:
            snaps.append(dict(step=step + 1, P={k: v.detach().cpu().clone() for k, v in P.items()}))
        if (step + 1) % args.eval_every == 0 or step + 1 == T:
            with torch.no_grad():
                Pd = {k: v.detach() for k, v in P.items()}
                mb = outcome_metrics(raw_y(forward(Pd, Xbal), nm), TS['bal'])
                mn = outcome_metrics(raw_y(forward(Pd, Xnear), nm), TS['near'])
                tl = (loss_acc / n_acc).tolist()
            loss_acc.zero_()
            n_acc = 0
            for m, c in enumerate(cfgs):
                row = dict(step=step + 1, model=m, p=c['p'], mode=c['mode'], seed=c['seed'], T=T, N=N, train_loss=tl[m])
                row.update({f'bal_{k}': float(v[m]) for k, v in mb.items()})
                row.update({f'near_{k}': float(v[m]) for k, v in mn.items()})
                rows.append(row)
            print(f'step {step + 1} ({time.time() - t0:.0f}s) loss[0]={tl[0]:.2e} opt={[round(float(x), 3) for x in mb["opt"][:9]]}', flush=True)

    with torch.no_grad():
        Pd = {k: v.detach() for k, v in P.items()}
        cb = binned_lift_prob(raw_y(forward(Pd, Xbal), nm), TS['bal'], EDGES_BAL)
        cn = binned_lift_prob(raw_y(forward(Pd, Xnear), nm), TS['near'], EDGES_NEAR)
    finals = []
    for m in range(M):
        f = dict([r for r in rows if r['model'] == m and r['step'] == T][0])
        f.update(data_info[m])
        f['curve_bal'], f['curve_near'] = to_py(cb[m]), to_py(cn[m])
        finals.append(f)
    with open(os.path.join(od, 'train_curves.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(args=vars(args), edges_bal=EDGES_BAL, edges_near=EDGES_NEAR, finals=finals),
              open(os.path.join(od, 'final_metrics.json'), 'w'), indent=1)
    torch.save(dict(P={k: v.detach().cpu() for k, v in P.items()}, P0=P0, cfgs=cfgs, args=vars(args),
                    norm={k: v.cpu() for k, v in nm.items()}), os.path.join(args.ckpt, f'{args.name}.pt'))
    if snaps:
        torch.save(snaps, os.path.join(args.ckpt, f'{args.name}_snaps.pt'))
    print(f'[{args.name}] done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
