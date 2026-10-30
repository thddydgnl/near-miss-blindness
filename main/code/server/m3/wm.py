"""R2-9 latent video world model (DINO-WM style) and frozen success classifier.
Predictor: frozen-DINOv2 frame tokens (64 x 64, normalised) + action token -> small ViT (4 blocks, d=128, 4 heads,
MLP 512, pre-LN) -> next-frame tokens (residual). Imagination = autoregressive rollout over the 4 actions.
Classifier: MLP on real final-frame tokens -> P(object lifted); trained once, then frozen.
Usage:  python wm.py clf            (train the classifier)
        python wm.py train --data std --mode rw --seed 0 --T 6000 --name ..."""
import os as _os
# B guard (added 14:43Z): GPU 1 only, torch hub inside ~/pmx (see ../bpaths.py)
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from world import D_TOK, FEAT_DIR, N_TOK, N_TRAIN

CK = os.environ.get('R29_CK', os.path.expanduser('~/pmx/B/ckpt/m3'))            # B E0 port
OUT = os.environ.get('R29_OUT', os.path.expanduser('~/pmx/B/results/E0/m3'))


class Block(nn.Module):
    def __init__(self, d, heads, mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.fc1, self.fc2 = nn.Linear(d, mlp), nn.Linear(mlp, d)

    def forward(self, x, hook=None, l=0):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        u = F.gelu(self.fc1(self.ln2(x)))
        if hook is not None:
            u = hook(l, u)
        return x + self.fc2(u)


class Predictor(nn.Module):
    def __init__(self, d=128, L=4, heads=4, mlp=512):
        super().__init__()
        self.L, self.mlp = L, mlp
        self.inp, self.act = nn.Linear(D_TOK, d), nn.Linear(3, d)
        self.pos = nn.Parameter(torch.randn(1, N_TOK + 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(L)])
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, D_TOK)

    def forward(self, z, a, hook=None):
        x = torch.cat([self.act(a)[:, None], self.inp(z)], 1) + self.pos
        for l, b in enumerate(self.blocks):
            x = b(x, hook, l)
        return z + self.out(self.ln(x[:, 1:]))

    def rollout(self, z0, acts, hook_fn=None):
        z, zs = z0, []
        for t in range(acts.shape[1]):
            z = self(z, acts[:, t], hook_fn(t) if hook_fn else None)
            zs.append(z)
        return zs


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(N_TOK * D_TOK, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def load_norm(dev):
    nm = torch.load(os.path.join(FEAT_DIR, 'norm.pt'))
    return nm['mean'].to(dev), nm['std'].to(dev)


def train_classifier(dev):
    torch.manual_seed(0)
    E = torch.load(os.path.join(FEAT_DIR, 'eval.pt'))['E']
    mu, sd = load_norm(dev)
    X = (E['clf_f4'].to(dev).float() - mu) / sd
    y = E['clf_y'].to(dev).float()
    Xtr, ytr, Xva, yva = X[:20000], y[:20000], X[20000:], y[20000:]
    clf = Classifier().to(dev)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for s in range(3000):
        i = torch.randint(0, len(Xtr), (512,), device=dev)
        loss = F.binary_cross_entropy_with_logits(clf(Xtr[i]), ytr[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float(((clf(Xva) > 0).float() == yva).float().mean())
        accs = {}
        for name, P in E['pol'].items():
            accs[name] = float(((clf((P['f4'].to(dev).float() - mu) / sd) > 0) == P['succ'].to(dev)).float().mean())
    os.makedirs(CK, exist_ok=True)
    torch.save(clf.state_dict(), os.path.join(CK, 'classifier.pt'))
    res = dict(val_acc=acc, policy_real_frame_acc=accs)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'classifier.json'), 'w'), indent=1)
    print('classifier', res, flush=True)


def train(args, dev):
    t0 = time.time()
    B = torch.load(os.path.join(FEAT_DIR, f'bank_s{args.seed}_N{args.N}.pt'))
    Fb = B['F'].to(dev)
    tab = {k: v.to(dev) for k, v in B['tabs'][args.data].items()}
    mu, sd = load_norm(dev)
    seq = torch.cat([tab['f0'][:, None], tab['f']], 1)            # [n_ep, 5] frame indices
    act = tab['act']
    n_ep = seq.shape[0]
    groups = {k: (tab['group'] == k).nonzero().squeeze(1) for k in range(4)}
    fails = (~tab['succ']).nonzero().squeeze(1)
    succs = tab['succ'].nonzero().squeeze(1)
    torch.manual_seed(1000 + args.seed)
    model = Predictor().to(dev)
    init_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    gen = torch.Generator(device=dev)
    gen.manual_seed(77 + args.seed)
    Bs = args.batch

    def draw(pool, n):
        return pool[torch.randint(0, len(pool), (n,), generator=gen, device=dev)]

    def sample():
        if args.mode == 'std':
            ep = torch.randint(0, n_ep, (Bs,), generator=gen, device=dev)
            t = torch.randint(0, 4, (Bs,), generator=gen, device=dev)
        elif args.mode == 'rw':
            ep = torch.cat([draw(fails, Bs // 2), draw(succs, Bs - Bs // 2)])
            t = torch.randint(0, 4, (Bs,), generator=gen, device=dev)
        elif args.mode == 'cp':
            k = Bs // 4
            f = draw(groups[2], k)
            tf = torch.randint(0, 4, (k,), generator=gen, device=dev)
            rest = draw(groups[0], Bs - 2 * k)
            ep = torch.cat([f, tab['partner'][f], rest])
            t = torch.cat([tf, tf, torch.randint(0, 4, (Bs - 2 * k,), generator=gen, device=dev)])
        elif args.mode == 'nm':
            k = Bs // 4
            ep = torch.cat([draw(groups[2], k), draw(groups[3], k), draw(groups[0], Bs - 2 * k)])
            t = torch.randint(0, 4, (Bs,), generator=gen, device=dev)
        return ep, t

    log = []
    for step in range(args.T):
        lr = args.lr * min(1.0, (step + 1) / 300) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / args.T)))
        for g in opt.param_groups:
            g['lr'] = lr
        ep, t = sample()
        zi = (Fb[seq[ep, t]].float() - mu) / sd
        zo = (Fb[seq[ep, t + 1]].float() - mu) / sd
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(zi, act[ep, t])
        loss = ((pred.float() - zo) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            log.append(dict(step=step + 1, loss=float(loss), sec=time.time() - t0))
            print(f'{args.name} step {step + 1} loss {float(loss):.4f} ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(CK, exist_ok=True)
    torch.save(dict(state=model.state_dict(), init=init_state, cfg=vars(args), log=log), os.path.join(CK, f'{args.name}.pt'))
    print(f'{args.name} done ({time.time() - t0:.0f}s)', flush=True)


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    if sys.argv[1] == 'clf':
        train_classifier(dev)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--data', default='std')
    ap.add_argument('--mode', default='std')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--T', type=int, default=6000)
    ap.add_argument('--N', type=int, default=N_TRAIN)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    train(args, dev)


if __name__ == '__main__':
    main()
