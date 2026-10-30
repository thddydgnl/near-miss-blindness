"""R2-9 world: the R2-4/R2-6 grasp scene as a 4-step rendered video (object-centred side camera, 224 px).
Episode (relative coordinates, object initially at x=0):
  f0  gripper open at (g, gz0)                              g = gx0 - ox ~ U[-0.25, 0.25]
  a1  move laterally by dx -> f1 gripper at (e, gz0)         e = g + dx = alignment error
  a2  descend to ZG and close -> f2                         captured iff |e| < w (object centred under gripper)
  a3  lift L/2 -> f3,  a4  lift L/2 -> f4                   success iff captured (object rises with gripper)
No slippery mode (R2-6 showed that mixing failure modes confounds the overall optimism).
Frames are encoded by frozen DINOv2 ViT-S/14 (16x16 patch tokens) -> 2x2 average pooling -> 8x8 = 64 tokens ->
PCA 64 dims per token. Run as a script to build the per-seed frame banks and the evaluation banks."""
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

IMG = 224
XR = (-0.45, 0.45)
ZR = (-0.08, 0.80)
PALETTE = [[0.85, 0.25, 0.25], [0.25, 0.70, 0.30], [0.25, 0.40, 0.85], [0.90, 0.75, 0.20], [0.60, 0.30, 0.75]]
ZG = 0.05
OPEN_IN, FW = 0.15, 0.02
H = 4
N_TOK, D_TOK = 64, 64
FEAT_DIR = '/workspace/feat9'
N_TRAIN = 30000
P_LOW, P_HIGH = 0.03, 0.3
NEAR = (1.0, 1.2)
ACT_SCALE = torch.tensor([0.3, 0.3, 1.0])

# evaluation policies: distribution of the alignment ratio r = |e|/w (sign random)
POLICIES = {
    'pi1_precise': ('gauss', 0.35),
    'pi2_good': ('gauss', 0.6),
    'pi3_medium': ('gauss', 0.9),
    'pi4_sloppy': ('gauss', 1.4),
    'pi5_far_failer': ('bimodal', (1.5, 2.0)),
    'pi6_near_failer': ('bimodal', (1.0, 1.2)),
}


def sample_scene(n, gen, dev):
    U = lambda a, b: a + (b - a) * torch.rand(n, generator=gen, device=dev)
    S = dict(w=U(0.05, 0.15), gz0=U(0.4, 0.6), L=U(0.1, 0.5), g=U(-0.25, 0.25), v=U(0, 1), v2=U(0, 1), v3=U(0, 1),
             u=U(0, 1))
    S['sign'] = 1.0 - 2.0 * (torch.rand(n, generator=gen, device=dev) < 0.5).float()
    S['c'] = torch.randint(0, 5, (n,), generator=gen, device=dev)
    return S


def sub(S, idx):
    return {k: v[idx] for k, v in S.items()}


def policy_r(name, n, gen, dev):
    kind, par = POLICIES[name]
    if kind == 'gauss':
        return (torch.randn(n, generator=gen, device=dev) * par).abs()
    lo, hi = par
    good = torch.rand(n, generator=gen, device=dev) < 0.5
    return torch.where(good, 0.8 * torch.rand(n, generator=gen, device=dev), lo + (hi - lo) * torch.rand(n, generator=gen, device=dev))


def actions(S, e):
    dx = e - S['g']
    z = torch.zeros_like(dx)
    o = torch.ones_like(dx)
    a = torch.stack([torch.stack([dx, z, z], 1), torch.stack([z, ZG - S['gz0'], o], 1),
                     torch.stack([z, S['L'] / 2, o], 1), torch.stack([z, S['L'] / 2, o], 1)], 1)
    return a / ACT_SCALE.to(a.device)  # [n, 4, 3] normalised


def frame_params(S, e, t):
    """state of frame t (0..4) for scenes S under alignment error e."""
    cap = e.abs() < S['w']
    z = torch.zeros_like(e)
    if t == 0:
        return dict(gx=S['g'], gz=S['gz0'], closed=torch.zeros_like(cap), ox=z, oz=z)
    if t == 1:
        return dict(gx=e, gz=S['gz0'], closed=torch.zeros_like(cap), ox=z, oz=z)
    lift = {2: 0.0, 3: 0.5, 4: 1.0}[t] * S['L']
    return dict(gx=e, gz=ZG + lift, closed=torch.ones_like(cap), ox=torch.where(cap, e, z), oz=torch.where(cap, lift, z), cap=cap)


def _cov(lo, hi, g):
    return (torch.minimum(g[None] + 1, hi[:, None]) - torch.maximum(g[None], lo[:, None])).clamp(0, 1)


def _rect(img, x0, x1, z0, z1, rgb, g):
    px = lambda x: (x - XR[0]) / (XR[1] - XR[0]) * IMG
    py = lambda z: (ZR[1] - z) / (ZR[1] - ZR[0]) * IMG
    a = (_cov(py(z1), py(z0), g)[:, :, None] * _cov(px(x0), px(x1), g)[:, None, :]).unsqueeze(1)
    img.mul_(1 - a).add_(a * rgb.view(-1, 3, 1, 1))


def render(S, e, t):
    fp = frame_params(S, e, t)
    n, dev = e.shape[0], e.device
    g = torch.arange(IMG, device=dev, dtype=torch.float32)
    img = torch.full((n, 3, IMG, IMG), 0.88, device=dev)
    one = torch.ones(n, device=dev)
    grey = lambda v: torch.tensor([v, v, v], device=dev)
    w = S['w']
    _rect(img, -2 * one, 2 * one, -2 * one, 0 * one, grey(0.55), g)
    _rect(img, fp['ox'] - w, fp['ox'] + w, fp['oz'], fp['oz'] + 0.10, torch.tensor(PALETTE, device=dev)[S['c']], g)
    cap = fp.get('cap', torch.zeros(n, dtype=torch.bool, device=dev))
    inner = torch.where(fp['closed'], torch.where(cap, w, 0.01 * one), OPEN_IN * one)
    outer = inner + FW
    gx, gz, gr = fp['gx'], fp['gz'], grey(0.15)
    _rect(img, gx - outer, gx + outer, gz + 0.06, gz + 0.09, gr, g)
    _rect(img, gx - outer, gx - inner, gz, gz + 0.06, gr, g)
    _rect(img, gx + inner, gx + outer, gz, gz + 0.06, gr, g)
    _rect(img, gx - 0.006, gx + 0.006, gz + 0.09, gz + 0.16, gr, g)
    return img


class Encoder:
    def __init__(self, dev):
        self.dev = dev
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        self.mu = self.V = None

    @torch.no_grad()
    def pooled(self, img):
        with torch.autocast('cuda', dtype=torch.float16):
            t = self.model.forward_features((img - self.mean) / self.std)['x_norm_patchtokens'].float()
        t = t.reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
        return F.avg_pool2d(t, 2).flatten(2).transpose(1, 2)  # [n, 64, 384]

    @torch.no_grad()
    def fit_pca(self, S, e, bs=256):
        T = []
        for t in range(5):
            for i in range(0, len(e), bs):
                T.append(self.pooled(render(sub(S, slice(i, i + bs)), e[i:i + bs], t)).reshape(-1, 384))
        T = torch.cat(T)
        self.mu = T.mean(0)
        ev, evec = torch.linalg.eigh((T - self.mu).T @ (T - self.mu) / len(T))
        self.V = evec[:, -D_TOK:].flip(-1).contiguous()
        return float(ev[-D_TOK:].sum() / ev.sum())

    @torch.no_grad()
    def frames(self, S, e, t, bs=384):
        out = []
        for i in range(0, len(e), bs):
            p = self.pooled(render(sub(S, slice(i, i + bs)), e[i:i + bs], t))
            out.append(((p - self.mu) @ self.V).half())
        return torch.cat(out)  # [n, 64, 64]


def build_seed(enc, seed, dev, N=N_TRAIN):
    """frame bank and episode tables for one data seed (paired across p within the seed)."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(2000 + seed)
    S = sample_scene(N, gen, dev)
    fail = S['u'] < P_LOW
    alt = (S['u'] >= P_LOW) & (S['u'] < P_HIGH)
    r_base = torch.where(fail, 1 + S['v'], S['v'])
    e_base = S['sign'] * r_base * S['w']
    bank, cursor = [], 0

    def add(x):
        nonlocal cursor
        bank.append(x)
        idx = torch.arange(cursor, cursor + x.shape[0], device=dev)
        cursor += x.shape[0]
        return idx

    f0 = add(enc.frames(S, e_base, 0))
    base = torch.stack([add(enc.frames(S, e_base, t)) for t in range(1, 5)], 1)          # [N, 4]
    ia = alt.nonzero().squeeze(1)
    Sa = sub(S, ia)
    e_alt = Sa['sign'] * (1 + Sa['v']) * Sa['w']
    altf = torch.stack([add(enc.frames(Sa, e_alt, t)) for t in range(1, 5)], 1)
    ifl = fail.nonzero().squeeze(1)
    Sf = sub(S, ifl)
    e_nmf = Sf['sign'] * (NEAR[0] + (NEAR[1] - NEAR[0]) * Sf['v2']) * Sf['w']                 # near-miss failures
    nmf = torch.stack([add(enc.frames(Sf, e_nmf, t)) for t in range(1, 5)], 1)
    e_cps = Sf['sign'] * (0.8 + 0.2 * Sf['v3']) * Sf['w']                                    # paired boundary successes
    cps = torch.stack([add(enc.frames(Sf, e_cps, t)) for t in range(1, 5)], 1)
    succ_pool = (S['u'] >= P_HIGH).nonzero().squeeze(1)
    perm = torch.randperm(len(succ_pool), generator=gen, device=dev)
    ius = succ_pool[perm[:len(ifl)]]                                                         # unpaired boundary successes
    idrop = succ_pool[perm[len(ifl):2 * len(ifl)]]                                           # dropped base successes (cp)
    Su = sub(S, ius)
    e_us = Su['sign'] * (0.8 + 0.2 * Su['v3']) * Su['w']
    usf = torch.stack([add(enc.frames(Su, e_us, t)) for t in range(1, 5)], 1)
    F_all = torch.cat(bank)
    act_base = actions(S, e_base)
    rr = lambda e, SS: e.abs() / SS['w']

    def table(f0i, fi, act, succ, r, group, partner):
        return dict(f0=f0i, f=fi, act=act, succ=succ, r=r, group=group, partner=partner)

    tabs = {}
    grp = torch.where(fail, 1, 0)
    neg = torch.full((N,), -1, dtype=torch.long, device=dev)
    tabs['std'] = table(f0, base, act_base, ~fail, r_base, grp, neg)                         # p = 0.03
    f_hi, act_hi, r_hi = base.clone(), act_base.clone(), r_base.clone()
    f_hi[ia], act_hi[ia], r_hi[ia] = altf, actions(Sa, e_alt), rr(e_alt, Sa)
    tabs['p03'] = table(f0, f_hi, act_hi, ~(fail | alt), r_hi, torch.where(fail | alt, 1, 0), neg)  # p = 0.3
    # near-miss variants (same 900 failure scenes now near-miss; 900 boundary successes added, N kept)
    f_nm, act_nm, r_nm, g_nm = base.clone(), act_base.clone(), r_base.clone(), grp.clone()
    f_nm[ifl], act_nm[ifl], r_nm[ifl], g_nm[ifl] = nmf, actions(Sf, e_nmf), rr(e_nmf, Sf), 2
    nm = table(f0, f_nm.clone(), act_nm.clone(), ~fail, r_nm.clone(), g_nm.clone(), neg.clone())
    nm['f'][ius], nm['act'][ius], nm['r'][ius], nm['group'][ius] = usf, actions(Su, e_us), rr(e_us, Su), 3
    tabs['nm'] = nm
    keep = torch.ones(N, dtype=torch.bool, device=dev)
    keep[idrop] = False
    kept = keep.nonzero().squeeze(1)
    n_k = len(kept)
    cp = table(torch.cat([f0[kept], f0[ifl]]), torch.cat([f_nm[kept], cps]), torch.cat([act_nm[kept], actions(Sf, e_cps)]),
               torch.cat([(~fail)[kept], torch.ones(len(ifl), dtype=torch.bool, device=dev)]),
               torch.cat([r_nm[kept], rr(e_cps, Sf)]), torch.cat([g_nm[kept], torch.full((len(ifl),), 3, device=dev)]),
               torch.full((n_k + len(ifl),), -1, dtype=torch.long, device=dev))
    pos_fail = torch.searchsorted(kept, ifl)                                                  # failure episodes inside kept
    cp['partner'][pos_fail] = n_k + torch.arange(len(ifl), device=dev)
    cp['partner'][n_k + torch.arange(len(ifl), device=dev)] = pos_fail
    tabs['cp'] = cp
    info = dict(N=N, n_fail=int(fail.sum()), n_alt=int(alt.sum()), n_frames=int(F_all.shape[0]))
    return F_all, tabs, info


def build_eval(enc, dev):
    gen = torch.Generator(device=dev)
    gen.manual_seed(12345)
    E = {}
    n = 1024
    S = sample_scene(n, gen, dev)
    E['pol_f0'] = enc.frames(S, torch.zeros(n, device=dev), 0)
    E['pol'] = {}
    for name in POLICIES:
        r = policy_r(name, n, gen, dev)
        e = S['sign'] * r * S['w']
        E['pol'][name] = dict(act=actions(S, e), r=r, succ=r < 1, f4=enc.frames(S, e, 4))
    # r* grid
    n2 = 256
    S2 = sample_scene(n2, gen, dev)
    E['grid_f0'] = enc.frames(S2, torch.zeros(n2, device=dev), 0)
    grid = torch.arange(0.6, 1.8001, 0.05, device=dev)
    E['grid_r'] = grid
    E['grid_act'] = torch.stack([actions(S2, S2['sign'] * rv * S2['w']) for rv in grid], 0)  # [G, n2, 4, 3]
    # success / failure episodes with real frames (latent prediction error)
    n3 = 512
    S3 = sample_scene(n3, gen, dev)
    for nm_, r in (('succ', 0.9 * S3['v']), ('fail', 1 + S3['v'])):
        e = S3['sign'] * r * S3['w']
        E[f'err_{nm_}'] = dict(f=torch.stack([enc.frames(S3, e, t) for t in range(5)], 1), act=actions(S3, e))
    # counterfactual pairs for n50
    n4 = 64
    S4 = sample_scene(n4, gen, dev)
    E['pair_f0'] = enc.frames(S4, torch.zeros(n4, device=dev), 0)
    E['pair_act_s'] = actions(S4, S4['sign'] * (0.05 + 0.65 * S4['v']) * S4['w'])
    E['pair_act_f'] = actions(S4, S4['sign'] * (1.3 + 0.6 * S4['v2']) * S4['w'])
    # classifier data: real final frames, r ~ U[0, 2)
    n5 = 22000
    S5 = sample_scene(n5, gen, dev)
    r5 = 2 * S5['v']
    E['clf_f4'] = enc.frames(S5, S5['sign'] * r5 * S5['w'], 4)
    E['clf_y'] = r5 < 1
    return E


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(FEAT_DIR, exist_ok=True)
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2').split(',')]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else N_TRAIN
    t0 = time.time()
    enc = Encoder(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(999)
    Sp = sample_scene(800, gen, dev)
    expl = enc.fit_pca(Sp, Sp['sign'] * 2 * Sp['v'] * Sp['w'])
    print(f'PCA explained {expl:.3f} ({time.time() - t0:.0f}s)', flush=True)
    E = build_eval(enc, dev)
    cpu = lambda x: {k: cpu(v) for k, v in x.items()} if isinstance(x, dict) else x.cpu()
    torch.save(dict(E=cpu(E), pca_mu=enc.mu.cpu(), pca_V=enc.V.cpu(), explained=expl), os.path.join(FEAT_DIR, 'eval.pt'))
    print(f'eval bank done ({time.time() - t0:.0f}s)', flush=True)
    for s in seeds:
        F_all, tabs, info = build_seed(enc, s, dev, N)
        if s == seeds[0]:
            x = F_all[:200000].float()
            torch.save(dict(mean=x.mean((0, 1)).cpu(), std=x.std((0, 1)).clamp_min(1e-3).cpu()), os.path.join(FEAT_DIR, 'norm.pt'))
        torch.save(dict(F=F_all.cpu(), tabs=cpu(tabs), info=info), os.path.join(FEAT_DIR, f'bank_s{s}_N{N}.pt'))
        print(f'seed {s}: {info} ({time.time() - t0:.0f}s)', flush=True)
    ex = torch.cat([render(sub(Sp, slice(0, 2)), Sp['sign'][:2] * r * Sp['w'][:2], t) for r in (0.5, 1.1) for t in range(5)])
    torch.save(ex.cpu(), os.path.join(FEAT_DIR, 'example_frames.pt'))
    print(f'done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
