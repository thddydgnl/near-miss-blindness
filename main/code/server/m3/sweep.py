"""B E4b M3 resolution sweep (descriptive; DESIGN_main v3.1 section 9), p = 0.03 ('std' data), seeds 0-2.
Variants (base = E1: 64 tokens (2x2-pooled DINOv2 16x16 patches) x PCA 64, predictor d 128 / 4 blocks / 4 heads / MLP 512):
  R1  256 tokens (no pooling) x PCA 64, base predictor           -> new frame banks, classifier, E1 grid
  R2  64 tokens x PCA 384 (full rotation, no reduction), base predictor
  R3  base representation, predictor d 64 / 2 blocks / 4 heads / MLP 256   (E1 banks)
  R4  base representation, predictor d 256 / 8 blocks / 8 heads / MLP 1024 (E1 banks)
For R1 / R2 everything is rebuilt with the same scene draws and recipes as world.py / world_e1.py / wm.py: PCA fit on
the 800 scenes of generator 999, training banks with the p = 0.03 base episodes only (generator 2000 + seed, f0 + 4
frames), token normalisation from the first 20,000 training frames (seed 0), the frozen success classifier (R2-9 recipe on the
22,000 classifier scenes of the R2-9 evaluation bank), and the E1 grid / ceiling scenes (generators 31337 / 31338).
Training: T 6,000, batch 256, AdamW lr 3e-4, init seed 1000 + seed, sampling generator 77 + seed (uniform 'std').
The bank stays in CPU memory for R1 / R2 (batches gathered per step).
usage: python sweep.py bank R1 | train R1 0 | eval R1"""
import os as _os
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import world as W
from wm import Block

VAR = dict(R1=dict(pool=1, D=64, d=128, L=4, heads=4, mlp=512), R2=dict(pool=2, D=384, d=128, L=4, heads=4, mlp=512),
           R3=dict(pool=2, D=64, d=64, L=2, heads=4, mlp=256), R4=dict(pool=2, D=64, d=256, L=8, heads=8, mlp=1024))
ROOT = os.path.expanduser('~/pmx/B/feat/E4/m3_sweep')
CKD = os.path.expanduser('~/pmx/B/ckpt/E4/m3_sweep')
OUT = os.path.expanduser('~/pmx/B/results/E4/m3')
E1F = os.path.expanduser('~/pmx/B/feat/E1/m3')
E1CK = os.path.expanduser('~/pmx/B/ckpt/E1/m3')


def ntok(v):
    return 64 if VAR[v]['pool'] == 2 else 256


def fdir(v):
    return E1F if v in ('R3', 'R4') else f'{ROOT}/{v}'


class Enc(W.Encoder):
    def __init__(self, dev, pool, D):
        super().__init__(dev)
        self.pool, self.D = pool, D

    @torch.no_grad()
    def pooled(self, img):
        with torch.autocast('cuda', dtype=torch.float16):
            t = self.model.forward_features((img - self.mean) / self.std)['x_norm_patchtokens'].float()
        t = t.reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
        if self.pool > 1:
            t = F.avg_pool2d(t, self.pool)
        return t.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def fit_pca(self, S, e, bs=256):
        T = []
        for t in range(5):
            for i in range(0, len(e), bs):
                T.append(self.pooled(W.render(W.sub(S, slice(i, i + bs)), e[i:i + bs], t)).reshape(-1, 384))
        T = torch.cat(T)
        self.mu = T.mean(0)
        C = (T - self.mu).T @ (T - self.mu) / len(T)
        ev, evec = torch.linalg.eigh(C)
        self.V = evec[:, -self.D:].flip(-1).contiguous()
        return float(ev[-self.D:].sum() / ev.sum())


class PredictorV(nn.Module):
    def __init__(self, n_tok, D, d, L, heads, mlp):
        super().__init__()
        self.L, self.mlp = L, mlp
        self.inp, self.act = nn.Linear(D, d), nn.Linear(3, d)
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(L)])
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, D)

    def forward(self, z, a):
        x = torch.cat([self.act(a)[:, None], self.inp(z)], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return z + self.out(self.ln(x[:, 1:]))

    def rollout(self, z0, acts):
        z, zs = z0, []
        for t in range(acts.shape[1]):
            z = self(z, acts[:, t])
            zs.append(z)
        return zs


class Clf(nn.Module):
    def __init__(self, n_tok, D):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(n_tok * D, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def train_clf(X, y, dev, n_tok, D, seed=0):
    torch.manual_seed(seed)
    clf = Clf(n_tok, D).to(dev)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    Xtr, ytr, Xva, yva = X[:20000], y[:20000], X[20000:], y[20000:]
    for _ in range(3000):
        i = torch.randint(0, len(Xtr), (512,), device=dev)
        loss = F.binary_cross_entropy_with_logits(clf(Xtr[i]), ytr[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        acc = float(((clf(Xva) > 0).float() == yva).float().mean())
    return clf, acc


def bank(v):
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    c = VAR[v]
    d = fdir(v)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    enc = Enc(dev, c['pool'], c['D'])
    gen = torch.Generator(device=dev)
    gen.manual_seed(999)
    Sp = W.sample_scene(800, gen, dev)
    expl = enc.fit_pca(Sp, Sp['sign'] * 2 * Sp['v'] * Sp['w'])
    print(f'{v}: PCA {c["D"]} of {ntok(v)} tokens, explained {expl:.4f} ({time.time() - t0:.0f}s)', flush=True)
    # classifier scenes: the R2-9 evaluation bank draws (generator 12345) -> replay the draw order of world.build_eval
    gen.manual_seed(12345)
    n = 1024
    S = W.sample_scene(n, gen, dev)
    for name in W.POLICIES:
        W.policy_r(name, n, gen, dev)
    S2 = W.sample_scene(256, gen, dev)
    S3 = W.sample_scene(512, gen, dev)
    S4 = W.sample_scene(64, gen, dev)
    S5 = W.sample_scene(22000, gen, dev)
    r5 = 2 * S5['v']
    clf_f4 = enc.frames(S5, S5['sign'] * r5 * S5['w'], 4)
    # E1 grid and ceiling scenes (world_e1.py)
    gen.manual_seed(31337)
    Sg = W.sample_scene(128, gen, dev)
    grid_f0 = enc.frames(Sg, torch.zeros(128, device=dev), 0)
    gact, gf4, use = [], [], []
    from world_e1 import GRID
    for r in GRID:
        e = Sg['sign'] * r * Sg['w']
        gact.append(W.actions(Sg, e))
        gf4.append(enc.frames(Sg, e, 4))
        use.append((Sg['w'] <= 0.13) if r >= 3.0 else torch.ones(128, dtype=torch.bool, device=dev))
    gen.manual_seed(31338)
    Sc = W.sample_scene(22000, gen, dev)
    rc = 3.2 * Sc['v']
    ec = Sc['sign'] * rc * Sc['w']
    ev = dict(clf_f4=clf_f4.cpu(), clf_y=(r5 < 1).cpu(), grid_r=torch.tensor(GRID), grid_f0=grid_f0.cpu(), grid_act=torch.stack(gact).cpu(),
              grid_f4=torch.stack(gf4).cpu(), grid_use=torch.stack(use).cpu(), ceil_f0=enc.frames(Sc, torch.zeros(22000, device=dev), 0).cpu(),
              ceil_act=W.actions(Sc, ec).cpu(), ceil_f4=enc.frames(Sc, ec, 4).cpu(), ceil_y=(rc < 1).cpu(), pca_mu=enc.mu.cpu(), pca_V=enc.V.cpu(),
              explained=expl)
    torch.save(ev, f'{d}/eval_sweep.pt')
    print(f'{v}: evaluation data ({time.time() - t0:.0f}s)', flush=True)
    for s in (0, 1, 2):
        gen.manual_seed(2000 + s)
        S = W.sample_scene(W.N_TRAIN, gen, dev)
        fail = S['u'] < W.P_LOW
        r_base = torch.where(fail, 1 + S['v'], S['v'])
        e_base = S['sign'] * r_base * S['w']
        fr = [enc.frames(S, e_base, t).cpu() for t in range(5)]                  # 5 x [N, ntok, D]
        Fb = torch.stack(fr, 1)                                                  # [N, 5, ntok, D]
        if s == 0:                                                           # token normalisation from 20,000 frames
            x = Fb[:4000].reshape(-1, c['D']).float()
            torch.save(dict(mean=x.mean(0), std=x.std(0).clamp_min(1e-3)), f'{d}/norm.pt')
            del x
        torch.save(dict(F=Fb, act=W.actions(S, e_base).cpu(), succ=(~fail).cpu(), r=r_base.cpu()), f'{d}/bank_s{s}.pt')
        print(f'{v}: bank seed {s} {tuple(Fb.shape)} ({time.time() - t0:.0f}s)', flush=True)
        del Fb, fr
    # frozen classifier on real f4 (R2-9 recipe)
    nm = torch.load(f'{d}/norm.pt')
    mu, sd = nm['mean'].to(dev), nm['std'].to(dev)
    X = (ev['clf_f4'].to(dev).float() - mu) / sd
    clf, acc = train_clf(X, ev['clf_y'].to(dev).float(), dev, ntok(v), c['D'])
    os.makedirs(CKD, exist_ok=True)
    torch.save(clf.state_dict(), f'{CKD}/{v}_classifier.pt')
    json.dump(dict(variant=v, pca_explained=expl, clf_val_acc=acc, time_s=time.time() - t0), open(f'{d}/bank_info.json', 'w'), indent=1)
    print(f'{v}: classifier val acc {acc:.4f}; SWEEP_BANK_DONE ({time.time() - t0:.0f}s)', flush=True)


def train(v, s):
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    c = VAR[v]
    name = f'sweep_{v}_s{s}'
    os.makedirs(CKD, exist_ok=True)
    if os.path.exists(f'{CKD}/{name}.pt'):
        print('exists', name)
        return
    t0 = time.time()
    d = fdir(v)
    if v in ('R3', 'R4'):
        B = torch.load(f'{d}/bank_s{s}_N{W.N_TRAIN}.pt', weights_only=False)
        tab = B['tabs']['std']
        seq = torch.cat([tab['f0'][:, None], tab['f']], 1)
        Fb = B['F'][seq]                                                          # [N, 5, 64, 64] (CPU)
        act, nmf = tab['act'], f'{d}/norm.pt'
    else:
        B = torch.load(f'{d}/bank_s{s}.pt', weights_only=False)
        Fb, act, nmf = B['F'], B['act'], f'{d}/norm.pt'
    nm = torch.load(nmf)
    mu, sd = nm['mean'].to(dev), nm['std'].to(dev)
    act = act.to(dev)
    N = Fb.shape[0]
    on_gpu = Fb.numel() * 2 < 2.5e9
    Fg = Fb.to(dev) if on_gpu else Fb
    torch.manual_seed(1000 + s)
    model = PredictorV(ntok(v), c['D'], c['d'], c['L'], c['heads'], c['mlp']).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    gen = torch.Generator(device=dev)
    gen.manual_seed(77 + s)
    T_, Bs = 6000, 256
    for step in range(T_):
        lr = 3e-4 * min(1.0, (step + 1) / 300) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / T_)))
        for g in opt.param_groups:
            g['lr'] = lr
        ep = torch.randint(0, N, (Bs,), generator=gen, device=dev)
        t = torch.randint(0, 4, (Bs,), generator=gen, device=dev)
        if on_gpu:
            zi, zo = Fg[ep, t], Fg[ep, t + 1]
        else:
            epc, tc = ep.cpu(), t.cpu()
            zi, zo = Fg[epc, tc].to(dev, non_blocking=True), Fg[epc, tc + 1].to(dev, non_blocking=True)
        zi, zo = (zi.float() - mu) / sd, (zo.float() - mu) / sd
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(zi, act[ep, t])
        loss = ((pred.float() - zo) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.save(dict(state=model.state_dict(), cfg=dict(variant=v, seed=s, **c), final_loss=float(loss), sec=time.time() - t0), f'{CKD}/{name}.pt')
    print(f'{name} done loss {float(loss):.4f} ({time.time() - t0:.0f}s)', flush=True)


def evaluate(v):
    """E1-grid evaluation per seed: real / imagined frozen-classifier logits and the ceiling classifier (trained on the
    model's imagined f4 of the 22,000 ceiling scenes, 20,000 / 2,000 split) applied to the grid's imagined f4."""
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    c = VAR[v]
    os.makedirs(OUT, exist_ok=True)
    if v in ('R3', 'R4'):
        G = torch.load(f'{E1F}/eval_e1.pt', weights_only=False)
        nm = torch.load(f'{E1F}/norm.pt')
        from wm import Classifier
        clf = Classifier().to(dev).eval()
        clf.load_state_dict(torch.load(f'{E1CK}/classifier.pt', map_location=dev))
    else:
        G = torch.load(f'{fdir(v)}/eval_sweep.pt', weights_only=False)
        nm = torch.load(f'{fdir(v)}/norm.pt')
        clf = Clf(ntok(v), c['D']).to(dev).eval()
        clf.load_state_dict(torch.load(f'{CKD}/{v}_classifier.pt', map_location=dev))
    mu, sd = nm['mean'].to(dev), nm['std'].to(dev)
    nz = lambda x: (x.to(dev).float() - mu) / sd
    nG = G['grid_act'].shape[0]
    with torch.no_grad():
        real_logit = torch.stack([clf(nz(G['grid_f4'][g])) for g in range(nG)]).cpu().numpy()
    z0 = nz(G['grid_f0'])
    for s in (0, 1, 2):
        o = f'{OUT}/sweep_{v}_s{s}.npz'
        if os.path.exists(o):
            continue
        t0 = time.time()
        model = PredictorV(ntok(v), c['D'], c['d'], c['L'], c['heads'], c['mlp']).to(dev).eval()
        model.load_state_dict(torch.load(f'{CKD}/sweep_{v}_s{s}.pt', map_location=dev, weights_only=False)['state'])
        with torch.no_grad():
            zi = [torch.cat([model.rollout(z0[i:i + 64], G['grid_act'][g][i:i + 64].to(dev))[-1] for i in range(0, z0.shape[0], 64)])
                  for g in range(nG)]
            imag_logit = torch.stack([clf(z) for z in zi]).cpu().numpy()
            zc = torch.cat([model.rollout(nz(G['ceil_f0'][i:i + 512]), G['ceil_act'][i:i + 512].to(dev))[-1] for i in range(0, 22000, 512)])
        y = G['ceil_y'].to(dev).float()
        torch.manual_seed(0)
        cc = Clf(ntok(v), c['D']).to(dev)
        opt = torch.optim.Adam(cc.parameters(), 1e-3)
        for _ in range(3000):
            i = torch.randint(0, 20000, (512,), device=dev)
            loss = F.binary_cross_entropy_with_logits(cc(zc[i]), y[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        cc.eval()
        with torch.no_grad():
            cacc = float(((cc(zc[20000:]) > 0).float() == y[20000:]).float().mean())
            ceil_logit = torch.stack([cc(z) for z in zi]).cpu().numpy()
        np.savez_compressed(o, grid_r=G['grid_r'].numpy(), use=G['grid_use'].numpy(), real_logit=real_logit, imag_logit=imag_logit,
                            ceil_logit=ceil_logit, ceil_val_acc=cacc)
        print(f'sweep {v} s{s}: ceiling val acc {cacc:.4f} ({time.time() - t0:.0f}s)', flush=True)
    print('SWEEP_EVAL_DONE', v, flush=True)


if __name__ == '__main__':
    cmd, v = sys.argv[1], sys.argv[2]
    if cmd == 'bank':
        bank(v)
    elif cmd == 'train':
        train(v, int(sys.argv[3]))
    else:
        evaluate(v)
