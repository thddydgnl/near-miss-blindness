"""R2-11 S4: CEM goal reaching in DINOv2 feature space with the final checkpoints (PLAN.md 4.6).

Tasks from held-out `id` episodes: start = sim state at frame t (history t-2..t), goal = frame t+H of the same episode
(reachable by the logged actions). CEM plans H actions minimising the feature MSE between the autoregressively
predicted frame t+H and the goal; the plan is executed open-loop in the simulator from the saved start state.
Success (pre-registered):
  precise: agent <= 2 px AND box <= 2 px AND |dtheta| <= 5 deg (mod 90)
  coarse : agent <= 8 px AND box <= 8 px
Usage: python cem.py --data /workspace/data --ckpt /workspace/ckpt --out /workspace/results
"""
import argparse
import json
import math
import os
import time

import torch

import env as E
from model import CONFIGS, make_model

H, POP, ELITE, ITERS = 5, 256, 32, 5
TOL_P = (2.0, 2.0, 5.0)
TOL_C = (8.0, 8.0)


def wrap_deg(x):
    q = math.pi / 2
    return (x - q * torch.round(x / q)).abs() * 180 / math.pi


def pick_tasks(d, n_push, n_free, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    pb, th = d['pb'], d['th']
    n_ep, T1 = pb.shape[:2]
    push, free = [], []
    for ep in torch.randperm(n_ep, generator=g).tolist():
        t = int(torch.randint(2, T1 - H, (1,), generator=g))
        mv = float((pb[ep, t + H] - pb[ep, t]).norm())
        rot = float(wrap_deg(th[ep, t + H] - th[ep, t]))
        if (mv >= 3.0 or rot >= 5.0) and len(push) < n_push:
            push.append((ep, t))
        elif mv < 0.5 and rot < 0.5 and len(free) < n_free:
            free.append((ep, t))
        if len(push) >= n_push and len(free) >= n_free:
            break
    return [(e, t, 'push') for e, t in push] + [(e, t, 'free') for e, t in free]


@torch.no_grad()
def plan(model, H0, zg, g):
    """H0: [n,3,64,384], zg: [n,64,384] -> planned actions [n,H,2]"""
    n = H0.shape[0]
    mu = torch.zeros(n, H, 2, device=H0.device)
    sd = torch.full((n, H, 2), 0.7, device=H0.device)
    for _ in range(ITERS):
        acts = (mu[:, None] + sd[:, None] * torch.randn(n, POP, H, 2, device=H0.device, generator=g)).clamp(-1, 1)
        acts[:, 0] = mu
        hist = H0[:, None].expand(n, POP, 3, 64, 384).reshape(n * POP, 3, 64, 384)
        A = acts.reshape(n * POP, H, 2)
        for k in range(H):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(hist, A[:, k]).float()
            hist = torch.cat([hist[:, 1:], z[:, None]], 1)
        cost = ((z.view(n, POP, 64, 384) - zg[:, None]) ** 2).mean((2, 3))
        idx = cost.topk(ELITE, dim=1, largest=False).indices
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(n, ELITE, H, 2))
        mu, sd = el.mean(1), el.std(1).clamp_min(0.05)
    return mu


def execute(d, tasks, acts, dev):
    ep = torch.tensor([t[0] for t in tasks])
    t0 = torch.tensor([t[1] for t in tasks])
    st = [d[k][ep, t0].to(dev) for k in ('pa', 'va', 'pb', 'vb', 'th', 'wb')]
    for k in range(H):
        st = list(E.control_step(*st, acts[:, k]))[:6]
    pa, _, pb, _, th, _ = st
    ea = (pa - d['pa'][ep, t0 + H].to(dev)).norm(dim=-1)
    eb = (pb - d['pb'][ep, t0 + H].to(dev)).norm(dim=-1)
    er = wrap_deg(th - d['th'][ep, t0 + H].to(dev))
    return ea, eb, er


def score(ea, eb, er, kinds):
    okp = (ea <= TOL_P[0]) & (eb <= TOL_P[1]) & (er <= TOL_P[2])
    okc = (ea <= TOL_C[0]) & (eb <= TOL_C[1])
    kp = torch.tensor([k == 'push' for k in kinds], device=ea.device)
    f = lambda m, sel=None: float(m.float().mean()) if sel is None else float(m[sel].float().mean())
    return dict(succ_precise=f(okp), succ_coarse=f(okc), succ_precise_push=f(okp, kp), succ_precise_free=f(okp, ~kp),
                succ_coarse_push=f(okc, kp), succ_coarse_free=f(okc, ~kp),
                agent_err=float(ea.mean()), box_err=float(eb.mean()), ang_err=float(er.mean()),
                agent_err_med=float(ea.median()), box_err_med=float(eb.median()),
                errs=dict(agent=ea.tolist(), box=eb.tolist(), ang=er.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--n_push', type=int, default=32)
    ap.add_argument('--n_free', type=int, default=32)
    ap.add_argument('--tag', default='final')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--names', default=','.join(CONFIGS))
    ap.add_argument('--chunk', type=int, default=16)
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    st = torch.load(f'{args.data}/stats.pt')
    mu, sd = st['mu'].to(dev), st['sd'].to(dev)
    d = torch.load(f'{args.data}/id.pt')
    tasks = pick_tasks(d, args.n_push, args.n_free, seed=123)
    kinds = [t[2] for t in tasks]
    ep = torch.tensor([t[0] for t in tasks])
    t0 = torch.tensor([t[1] for t in tasks])
    fr = t0[:, None] + torch.arange(-2, 1)[None]
    H0 = ((d['Z'][ep[:, None], fr].to(dev).float() - mu) / sd)
    ZG = ((d['Z'][ep, t0 + H].to(dev).float() - mu) / sd)
    res = {'tasks': len(tasks), 'n_push': kinds.count('push'), 'n_free': kinds.count('free'), 'H': H, 'pop': POP,
           'elite': ELITE, 'iters': ITERS, 'tol_precise': TOL_P, 'tol_coarse': TOL_C, 'kinds': kinds, 'runs': []}
    logged = d['a'][ep[:, None], t0[:, None] + torch.arange(H)[None]].to(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    for name, acts in (('logged', logged), ('zero', torch.zeros_like(logged)),
                       ('random', torch.rand(logged.shape, device=dev, generator=g) * 2 - 1)):
        r = dict(cond='baseline_' + name, seed=-1, **score(*execute(d, tasks, acts, dev), kinds))
        res['runs'].append(r)
        print({k: v for k, v in r.items() if k != 'errs'}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    for s in [int(x) for x in args.seeds.split(',')]:
        for name in args.names.split(','):
            fn = f'{args.ckpt}/{name}_s{s}_{args.tag}.pt'
            if not os.path.exists(fn):
                print('missing', fn, flush=True)
                continue
            model = make_model(name).to(dev)
            model.load_state_dict(torch.load(fn, map_location=dev))
            model.eval()
            g = torch.Generator(device=dev)
            g.manual_seed(1000 + s)
            acts = torch.cat([plan(model, H0[i:i + args.chunk], ZG[i:i + args.chunk], g)
                              for i in range(0, len(tasks), args.chunk)])
            r = dict(cond=name, seed=s, **score(*execute(d, tasks, acts, dev), kinds))
            res['runs'].append(r)
            print({k: v for k, v in r.items() if k != 'errs'}, f'{time.time() - t_start:.0f}s', flush=True)
            json.dump(res, open(f'{args.out}/cem.json', 'w'), indent=1)
            del model
            torch.cuda.empty_cache()
    json.dump(res, open(f'{args.out}/cem.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
