"""Visual version of the grasp world (R2-6).
The initial scene is rendered by an object-centred side camera (224x224, anti-aliased box coverage):
table, object (width 2w, colour = observable property; colour 5 = slippery), gripper (palm, two fingers, wrist marker)
at horizontal offset gx0-ox and height gz0. Frozen DINOv2 ViT-S/14 patch tokens (16x16x384) are projected by PCA
to 32 dims per patch -> 8192-dim observation; the action (dx, L) is appended. Targets are object-relative trajectories.
Run as a script to build the per-seed feature banks and the visual test sets."""
import os
import sys
import time

import torch
import torch.nn.functional as F

from common import (N_COLORS, SLIP_COLOR, SLIP_SHARE, norm_stats, realize, rel_y, sample_scene, test_sets, train_data)

IMG = 224
XR = (-0.6, 0.6)
ZR = (-0.08, 0.80)
PALETTE = [[0.85, 0.25, 0.25], [0.25, 0.70, 0.30], [0.25, 0.40, 0.85], [0.90, 0.75, 0.20], [0.60, 0.30, 0.75],
           [0.20, 0.80, 0.85]]
N_PCA = 32
FEAT_DIM = 256 * N_PCA
FEAT_DIR = '/workspace/feat'
N_VIS = 100000


def _cov(lo, hi, g):
    return (torch.minimum(g[None] + 1, hi[:, None]) - torch.maximum(g[None], lo[:, None])).clamp(0, 1)


def _px_x(x):
    return (x - XR[0]) / (XR[1] - XR[0]) * IMG


def _px_y(z):
    return (ZR[1] - z) / (ZR[1] - ZR[0]) * IMG


def _rect(img, x0, x1, z0, z1, rgb, g):
    cx = _cov(_px_x(x0), _px_x(x1), g)
    cy = _cov(_px_y(z1), _px_y(z0), g)
    a = (cy[:, :, None] * cx[:, None, :]).unsqueeze(1)
    img.mul_(1 - a).add_(a * rgb.view(-1, 3, 1, 1))


def render(X_state):
    rel_gx = X_state[:, 0] - X_state[:, 2]
    gz0, w = X_state[:, 1], X_state[:, 3]
    color = X_state[:, 6:6 + N_COLORS].argmax(1)
    n, dev = X_state.shape[0], X_state.device
    g = torch.arange(IMG, device=dev, dtype=torch.float32)
    img = torch.full((n, 3, IMG, IMG), 0.88, device=dev)
    one = torch.ones(n, device=dev)
    grey = lambda v: torch.tensor([v, v, v], device=dev)
    _rect(img, -2 * one, 2 * one, -2 * one, 0 * one, grey(0.55), g)                     # table
    _rect(img, -w, w, 0 * one, 0.10 * one, torch.tensor(PALETTE, device=dev)[color], g)  # object
    gr = grey(0.15)
    _rect(img, rel_gx - 0.09, rel_gx + 0.09, gz0 + 0.06, gz0 + 0.09, gr, g)              # palm
    _rect(img, rel_gx - 0.09, rel_gx - 0.07, gz0, gz0 + 0.06, gr, g)                     # fingers
    _rect(img, rel_gx + 0.07, rel_gx + 0.09, gz0, gz0 + 0.06, gr, g)
    _rect(img, rel_gx - 0.006, rel_gx + 0.006, gz0 + 0.09, gz0 + 0.16, gr, g)            # wrist marker
    return img


class Encoder:
    def __init__(self, dev):
        self.dev = dev
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        self.mu, self.V = None, None

    @torch.no_grad()
    def tokens(self, X_state):
        x = (render(X_state) - self.mean) / self.std
        with torch.autocast('cuda', dtype=torch.float16):
            out = self.model.forward_features(x)
        return out['x_norm_patchtokens'].float()  # [n, 256, 384]

    @torch.no_grad()
    def fit_pca(self, X_state, bs=256):
        T = torch.cat([self.tokens(X_state[i:i + bs]).reshape(-1, 384) for i in range(0, len(X_state), bs)])
        self.mu = T.mean(0)
        C = (T - self.mu).T @ (T - self.mu) / len(T)
        ev, evec = torch.linalg.eigh(C)
        self.V = evec[:, -N_PCA:].flip(-1).contiguous()
        return float(ev[-N_PCA:].sum() / ev.sum())

    @torch.no_grad()
    def features(self, X_state, bs=512):
        out = []
        for i in range(0, len(X_state), bs):
            t = self.tokens(X_state[i:i + bs])
            out.append(((t - self.mu) @ self.V).reshape(t.shape[0], -1).half())
        return torch.cat(out)


def vis_X(feat, X_state):
    return torch.cat([feat.float(), X_state[:, 4:6]], 1)


def vis_set(enc, d):
    f = enc.features(d['X'])
    return dict(X=vis_X(f, d['X']), Y=rel_y(d['Y'], d['X']), outcome=d['outcome'], r=d['r'], L=d['L'],
                X_state=d['X'])


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(FEAT_DIR, exist_ok=True)
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2').split(',')]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else N_VIS
    t0 = time.time()
    enc = Encoder(dev)
    ref = train_data(0.3, 999, 4096, dev)['X']
    expl = enc.fit_pca(ref)
    torch.save(dict(mu=enc.mu.cpu(), V=enc.V.cpu(), explained=expl), os.path.join(FEAT_DIR, 'pca.pt'))
    print(f'PCA fitted, explained variance {expl:.3f} ({time.time() - t0:.0f}s)', flush=True)
    for s in seeds:
        gen = torch.Generator(device=dev)
        gen.manual_seed(1000 + s)
        S = sample_scene(N, gen, dev)                       # identical draws to train_data(p, s, N)
        base = realize(S, torch.zeros(N, dtype=torch.long, device=dev))
        slip_ids = (S['u'] < 0.3 * SLIP_SHARE).nonzero().squeeze(1)
        X5 = base['X'][slip_ids].clone()
        X5[:, 6:6 + N_COLORS] = 0
        X5[:, 6 + SLIP_COLOR] = 1
        feat = torch.cat([enc.features(base['X']), enc.features(X5)])
        slip_rank = torch.full((N,), -1, dtype=torch.long, device=dev)
        slip_rank[slip_ids] = torch.arange(len(slip_ids), device=dev)
        torch.save(dict(feat=feat.cpu(), slip_rank=slip_rank.cpu(), N=N), os.path.join(FEAT_DIR, f'bank_s{s}_N{N}.pt'))
        print(f'seed {s}: bank {tuple(feat.shape)} ({time.time() - t0:.0f}s)', flush=True)
        if s == seeds[0]:
            # normalisation: feature stats from this bank's base images, action and target stats from the
            # p=0.3 reference sample used by the state version
            fb = feat[:N].float()
            ref_d = train_data(0.3, 999, 200000, dev)
            nm_state = norm_stats(dev)
            yr = rel_y(ref_d['Y'], ref_d['X'])
            nm = dict(xm=torch.cat([fb.mean(0), nm_state['xm'][4:6]]), xs=torch.cat([fb.std(0).clamp_min(1e-3), nm_state['xs'][4:6]]),
                      ym=yr.mean(0), ys=yr.std(0).clamp_min(0.05))
            torch.save({k: v.cpu() for k, v in nm.items()}, os.path.join(FEAT_DIR, 'norm_visual.pt'))
            del fb
    TS = test_sets(dev)
    VT = {k: vis_set(enc, TS[k]) for k in ('bal', 'near')}
    VT['pairs'] = {k: (vis_set(enc, a), vis_set(enc, b)) for k, (a, b) in TS['pairs'].items()}
    cpu = lambda d: {k: v.cpu() for k, v in d.items()}
    torch.save(dict(bal=cpu(VT['bal']), near=cpu(VT['near']),
                    pairs={k: (cpu(a), cpu(b)) for k, (a, b) in VT['pairs'].items()}),
               os.path.join(FEAT_DIR, 'test_visual.pt'))
    # a few example renders for the record
    ex = render(TS['bal']['X'][[0, 1, 4096, 4097, 7168, 7169]])
    torch.save(ex.cpu(), os.path.join(FEAT_DIR, 'example_renders.pt'))
    print(f'visual features done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
