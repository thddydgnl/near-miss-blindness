"""R2-11 training driver: one process per seed trains all crushing configs in lockstep on the same minibatch sequence
(equal steps, equal optimiser and schedule). Evals (held-out ID loss) are appended to <out>/evals_s<seed>.jsonl.

Usage: python train.py --seed 0 --steps 12000 [--configs w256,mlp64,...]
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from model import CONFIGS, make_model, n_params

NPER = 38          # samples per episode (t = 2..39 of 41 frames)
WD = 0.05
CH = 500


def load_set(path, dev, mu, sd):
    d = torch.load(path, map_location='cpu')
    Z = d['Z']
    out = torch.empty(Z.shape, dtype=torch.float16, device=dev)
    for i in range(0, Z.shape[0], 100):
        out[i:i + 100] = ((Z[i:i + 100].to(dev).float() - mu) / sd).half()
    return out, d['a'].to(dev).float()


def gather(Z, A, idx):
    ep = idx // NPER
    t = idx % NPER + 2
    fr = t[:, None] + torch.arange(-2, 2, device=idx.device)[None]
    X = Z[ep[:, None], fr]
    return X[:, :3].float(), X[:, 3].float(), A[ep, t]


@torch.no_grad()
def id_loss(model, EV):
    was = model.training
    model.eval()
    Z, A, sel = EV
    se = torch.zeros((), dtype=torch.float64, device=Z.device)
    for i in range(0, sel.numel(), CH):
        H, Y, a = gather(Z, A, sel[i:i + CH])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(H, a)
        se += ((p.float() - Y) ** 2).double().sum()
    model.train(was)
    return float(se) / (sel.numel() * 64 * 384)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--configs', default=','.join(CONFIGS))
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--warm', type=int, default=300)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lr_final', type=float, default=0.1)
    ap.add_argument('--n_eval', type=int, default=3000)
    ap.add_argument('--evals', default='0,500,1000,2000,3000,4000,6000,8000,10000,12000')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    json.dump(vars(args), open(f'{args.out}/train_args_s{args.seed}.json', 'w'), indent=1)
    t0 = time.time()
    st = torch.load(f'{args.data}/stats.pt')
    mu, sd = st['mu'].to(dev), st['sd'].to(dev)
    Ztr, Atr = load_set(f'{args.data}/train.pt', dev, mu, sd)
    Zid, Aid = load_set(f'{args.data}/id.pt', dev, mu, sd)
    g = torch.Generator(device=dev)
    g.manual_seed(99)
    n_id = Zid.shape[0] * NPER
    EV = (Zid, Aid, torch.randperm(n_id, generator=g, device=dev)[:min(args.n_eval, n_id)])
    # copy baseline (prediction = current frame)
    se = 0.0
    for i in range(0, EV[2].numel(), CH):
        H, Y, _ = gather(Zid, Aid, EV[2][i:i + CH])
        se += float(((H[:, 2] - Y) ** 2).double().sum())
    copy_mse = se / (EV[2].numel() * 64 * 384)
    print(f'data loaded {time.time() - t0:.0f}s train {tuple(Ztr.shape)} copy_mse {copy_mse:.4f} '
          f'mem {torch.cuda.memory_allocated() / 2**30:.1f} GB', flush=True)
    s = args.seed
    n_tr = Ztr.shape[0] * NPER
    g1 = torch.Generator(device=dev)
    g1.manual_seed(10000 + s)
    idx = torch.randint(n_tr, (args.steps, args.bs), generator=g1, device=dev)
    evals = sorted({int(e) for e in args.evals.split(',') if int(e) <= args.steps} | {args.steps})
    log = open(f'{args.out}/evals_s{s}.jsonl', 'a')
    names = args.configs.split(',')
    runs = []
    for k, name in enumerate(names):
        torch.manual_seed(1000 * s + 17 * k + 1)
        m = make_model(name).to(dev)
        torch.save(m.state_dict(), f'{args.ckpt}/{name}_s{s}_init.pt')
        dec, nodec = [], []
        for n, p in m.named_parameters():
            (dec if (p.ndim >= 2 and n.split('.')[-1] not in ('pos', 'tim')) else nodec).append(p)
        opt = torch.optim.AdamW([dict(params=dec, weight_decay=WD), dict(params=nodec, weight_decay=0.0)],
                                lr=args.lr, betas=(0.9, 0.98), fused=True)
        runs.append(dict(name=name, model=m, opt=opt, loss_acc=torch.zeros((), device=dev), loss_n=0,
                         n_params=n_params(m)))
        print(name, CONFIGS[name], 'params', n_params(m), flush=True)

    def lr_at(k):
        if k < args.warm:
            return args.lr * (k + 1) / args.warm
        p = (k - args.warm) / max(1, args.steps - args.warm)
        return args.lr * (args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * p)))

    t1 = time.time()
    for k in range(args.steps + 1):
        if k in evals:
            msg = []
            for r in runs:
                rec = dict(cond=r['name'], seed=s, step=k, id_mse=id_loss(r['model'], EV), copy_mse=copy_mse,
                           train_loss=(float(r['loss_acc']) / r['loss_n']) if r['loss_n'] else None,
                           n_params=r['n_params'], wall=round(time.time() - t1, 1))
                rec['id_rel'] = rec['id_mse'] / copy_mse
                r['loss_acc'].zero_()
                r['loss_n'] = 0
                log.write(json.dumps(rec) + '\n')
                log.flush()
                msg.append(f"{r['name']}:{rec['id_rel']:.3f}")
            print(f'[s{s}] step {k} id_rel ' + ' '.join(msg) + f' {(time.time() - t1) / 60:.1f} min', flush=True)
        if k == args.steps:
            break
        H, Y, a = gather(Ztr, Atr, idx[k])
        lr = lr_at(k)
        for r in runs:
            for gr in r['opt'].param_groups:
                gr['lr'] = lr
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = r['model'](H, a)
            loss = F.mse_loss(pred.float(), Y)
            r['opt'].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(r['model'].parameters(), 1.0)
            r['opt'].step()
            r['loss_acc'] += loss.detach()
            r['loss_n'] += 1
        if (k + 1) % 500 == 0:
            torch.cuda.synchronize()
            print(f'[s{s}] it {k + 1} {(time.time() - t1) / (k + 1) * 1000:.1f} ms/it '
                  f'{(time.time() - t1) / 60:.1f} min', flush=True)
    for r in runs:
        torch.save(r['model'].state_dict(), f'{args.ckpt}/{r["name"]}_s{s}_final.pt')
    log.close()
    print(f'all done {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
