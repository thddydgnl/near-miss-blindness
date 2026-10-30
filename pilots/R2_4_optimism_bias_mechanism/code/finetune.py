"""Off-ramp check (S3b): fine-tune trained 'std' models with 20 new miss-failure examples + 80 replay samples
from the model's own training set. Measures per-layer weight change share, activation change share,
gradient share, and transplant recovery (swap fine-tuned layers into the base model).
Control: same procedure with 20 new SUCCESS examples instead of failures."""
import argparse
import json
import math
import os

import torch

from common import (DEPTH, forward, nx, ny, outcome_metrics, raw_y, realize, sample_scene, slice_model,
                    test_sets, to_py, train_data)

LAYERS = list(range(DEPTH + 1))   # 0..3 hidden linear maps, 4 = readout
LR_GRID = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]


def run_ft(Pb, X, Y, steps, lr, opt, layers):
    keys = [f'{t}{l}' for l in layers for t in ('W', 'b')]
    P = {k: v.clone() for k, v in Pb.items()}
    for k in keys:
        P[k].requires_grad_(True)
    mom = {k: torch.zeros_like(P[k]) for k in keys}
    vel = {k: torch.zeros_like(P[k]) for k in keys}
    losses, g0 = [], None
    for s in range(steps):
        loss = ((forward(P, X) - Y) ** 2).mean()
        g = torch.autograd.grad(loss, [P[k] for k in keys])
        if s == 0:
            g0 = {k: gi.detach().pow(2).sum() for k, gi in zip(keys, g)}
        losses.append(float(loss))
        with torch.no_grad():
            for k, gi in zip(keys, g):
                if opt == 'sgd':
                    P[k].sub_(lr * gi)
                else:
                    mom[k].mul_(0.9).add_(gi, alpha=0.1)
                    vel[k].mul_(0.999).addcmul_(gi, gi, value=0.001)
                    P[k].sub_(lr * (mom[k] / (1 - 0.9 ** (s + 1))) / ((vel[k] / (1 - 0.999 ** (s + 1))).sqrt() + 1e-8))
    with torch.no_grad():
        losses.append(float(((forward(P, X) - Y) ** 2).mean()))
    return {k: v.detach() for k, v in P.items()}, losses, g0


def stable(losses):
    return all(math.isfinite(x) for x in losses) and losses[-1] < losses[0] and max(losses) < 2 * losses[0]


def layer_shares(Pb, Pf, g0=None):
    sq, rel, gs = [], [], []
    for l in LAYERS:
        d = (Pf[f'W{l}'] - Pb[f'W{l}']).pow(2).sum() + (Pf[f'b{l}'] - Pb[f'b{l}']).pow(2).sum()
        nrm = Pb[f'W{l}'].pow(2).sum() + Pb[f'b{l}'].pow(2).sum()
        sq.append(float(d))
        rel.append(math.sqrt(float(d)) / math.sqrt(float(nrm)))
        if g0 is not None:
            gs.append(float(g0.get(f'W{l}', torch.tensor(0.)) + g0.get(f'b{l}', torch.tensor(0.))))
    out = dict(w_sq_share=[x / max(sum(sq), 1e-30) for x in sq], w_rel_share=[x / max(sum(rel), 1e-30) for x in rel],
               w_sq=sq, w_rel=rel)
    if g0 is not None:
        out['grad0_share'] = [x / max(sum(gs), 1e-30) for x in gs]
    return out


def act_shares(Pb, Pf, X):
    yb, ab = forward(Pb, X, return_acts=True)
    yf, af = forward(Pf, X, return_acts=True)
    d = [float((af[l][0] - ab[l][0]).pow(2).sum(-1).mean() / ab[l][0].pow(2).sum(-1).mean()) for l in range(DEPTH)]
    d.append(float((yf - yb).pow(2).sum(-1).mean() / yb.pow(2).sum(-1).mean()))
    return dict(act_rel=d, act_share=[x / max(sum(d), 1e-30) for x in d])


def transplant(Pb, Pf, layers):
    P = dict(Pb)
    for l in layers:
        P[f'W{l}'], P[f'b{l}'] = Pf[f'W{l}'], Pf[f'b{l}']
    return P


def metrics(P, nm, TS):
    return {k: float(v) for k, v in outcome_metrics(raw_y(forward(P, nx(TS['bal']['X'], nm)), nm), TS['bal']).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='main')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--n_fail', type=int, default=20)
    ap.add_argument('--n_replay', type=int, default=80)
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(os.path.join(args.ckpt, f'{args.name}.pt'), map_location=dev)
    nm = {k: v.to(dev) for k, v in ck['norm'].items()}
    P = {k: v.to(dev) for k, v in ck['P'].items()}
    cfgs = ck['cfgs']
    TS = test_sets(dev)
    Xmiss = nx(TS['bal']['X'][TS['bal']['outcome'] == 1], nm)
    results = []
    for m, c in enumerate(cfgs):
        if c['mode'] != 'std':
            continue
        Pb = slice_model(P, m)
        gen = torch.Generator(device=dev)
        gen.manual_seed(31337 + c['seed'])
        S = sample_scene(args.n_fail, gen, dev)
        fails = realize(S, torch.ones(args.n_fail, dtype=torch.long, device=dev))
        S2 = sample_scene(args.n_fail, gen, dev)
        succs = realize(S2, torch.zeros(args.n_fail, dtype=torch.long, device=dev))
        d = train_data(c['p'], c['seed'], c['N'], dev)
        ridx = torch.randperm(c['N'], generator=gen, device=dev)[:args.n_replay]
        Xr, Yr = nx(d['X'][ridx], nm), ny(d['Y'][ridx], nm)
        Xft = torch.cat([nx(fails['X'], nm), Xr])
        Yft = torch.cat([ny(fails['Y'], nm), Yr])
        Xct = torch.cat([nx(succs['X'], nm), Xr])
        Yct = torch.cat([ny(succs['Y'], nm), Yr])
        res = dict(model=m, cfg=c, base=metrics(Pb, nm, TS), n_replay_fail=int((d['outcome'][ridx] != 0).sum()))
        # full SGD: largest stable lr from the grid (optimisation-validity rule, fixed before results)
        lr_used, Pf, losses, g0 = None, None, None, None
        for lr in LR_GRID:
            Pf, losses, g0 = run_ft(Pb, Xft, Yft, args.steps, lr, 'sgd', LAYERS)
            if stable(losses):
                lr_used = lr
                break
        res['lr_sgd'] = lr_used
        variants = {}
        if lr_used is not None:
            variants['full_sgd'] = (Pf, losses, g0)
            variants['ctrl_sgd'] = run_ft(Pb, Xct, Yct, args.steps, lr_used, 'sgd', LAYERS)
            variants['last1_sgd'] = run_ft(Pb, Xft, Yft, args.steps, lr_used, 'sgd', [4])
            variants['last2_sgd'] = run_ft(Pb, Xft, Yft, args.steps, lr_used, 'sgd', [3, 4])
        variants['full_adam'] = run_ft(Pb, Xft, Yft, args.steps, 1e-4, 'adam', LAYERS)
        variants['ctrl_adam'] = run_ft(Pb, Xct, Yct, args.steps, 1e-4, 'adam', LAYERS)
        for vn, (Pv, lv, gv) in variants.items():
            with torch.no_grad():
                r = dict(loss_first=lv[0], loss_last=lv[-1], after=metrics(Pv, nm, TS))
                r.update(layer_shares(Pb, Pv, gv))
                r.update(act_shares(Pb, Pv, Xmiss))
                if vn in ('full_sgd', 'full_adam'):
                    tr = {}
                    gain = r['after']['acc_miss'] - res['base']['acc_miss']
                    for nmz, ls in (('L0', [0]), ('L1', [1]), ('L2', [2]), ('L3', [3]), ('L4', [4]), ('L3L4', [3, 4]),
                                    ('L0L1L2', [0, 1, 2])):
                        mt = metrics(transplant(Pb, Pv, ls), nm, TS)
                        tr[nmz] = dict(acc_miss=mt['acc_miss'], acc_s=mt['acc_s'],
                                       recovery=(mt['acc_miss'] - res['base']['acc_miss']) / gain if gain > 0.02 else float('nan'))
                    r['transplant'] = tr
                    r['gain_acc_miss'] = gain
            res[vn] = r
        results.append(res)
        fs = res.get('full_sgd', {})
        print(f"model {m} p={c['p']} seed={c['seed']} lr={lr_used} base_miss={res['base']['acc_miss']:.3f} "
              f"-> {fs.get('after', {}).get('acc_miss', float('nan')):.3f} w_sq_share={[round(x, 2) for x in fs.get('w_sq_share', [])]}",
              flush=True)
    od = os.path.join(args.out, args.name)
    os.makedirs(od, exist_ok=True)
    json.dump(to_py(results), open(os.path.join(od, 'finetune.json'), 'w'), indent=1)
    print('finetune done', flush=True)


if __name__ == '__main__':
    main()
