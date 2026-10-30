"""Shared code for pilot R2-4 (r2n4): 2D grasp-and-lift world, batched MLP world model, test sets, metrics.

World (side view, x horizontal, z vertical), horizon H=10 steps, fully deterministic:
  steps 1-3  gripper moves laterally by dx (from gx0 to gx0+dx) at height gz0
  steps 4-5  gripper descends to grasp height ZG
  step  6    gripper closes. If |e| < w (e = gx0+dx-ox, alignment error) the object is captured
             and centred under the gripper (object x snaps to gripper x).
  steps 7-10 gripper lifts by L (L/4 per step).
Outcomes (deterministic given inputs):
  success (0): captured and not slippery -> object rises with the gripper (final object z = L)
  miss    (1): |e| >= w -> object stays on the table (final z = 0)
  slip    (2): captured but colour == SLIP_COLOR -> object rises 2 steps then drops (final z = 0)
Model input  (12): gx0, gz0, ox, w, dx, L, colour one-hot(6)
Model output (40): for each step t: gripper x, gripper z, object x, object z (absolute positions)
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

H = 10
D_IN = 12
D_OUT = 4 * H
N_COLORS = 6
SLIP_COLOR = 5
ZG = 0.05
WIDTH = 256
DEPTH = 4                 # hidden layers; linear maps W0..W3 (hidden) and W4 (readout)
SLIP_SHARE = 0.25         # share of the failure rate p that is the rarer slip mode
TAU_WEAK = 0.1            # weak/strong split for regime analysis (|a| < TAU_WEAK * m_j)

IDX_OZ_LIFT = [t * 4 + 3 for t in range(6, 10)]   # object z during lift steps 7..10
IDX_OZ_FINAL = 9 * 4 + 3                           # object z at step 10
IDX_GZ_APPROACH = [t * 4 + 1 for t in range(0, 5)] # gripper z during steps 1..5 (approach/descent)
IDX_OBJ = [t * 4 + k for t in range(H) for k in (2, 3)]


# ----------------------------------------------------------------------------- world
def make_inputs(gx0, gz0, ox, w, dx, L, color):
    oh = F.one_hot(color, N_COLORS).to(gx0.dtype)
    return torch.cat([torch.stack([gx0, gz0, ox, w, dx, L], 1), oh], 1)


def rollout(gx0, gz0, ox, w, dx, L, color):
    n = gx0.shape[0]
    dev, dt = gx0.device, gx0.dtype
    e = gx0 + dx - ox
    aligned = e.abs() < w
    slip = color == SLIP_COLOR
    outcome = torch.zeros(n, dtype=torch.long, device=dev)
    outcome[aligned & slip] = 2
    outcome[~aligned] = 1
    t = torch.arange(1, H + 1, device=dev, dtype=dt)
    gx = gx0[:, None] + dx[:, None] * torch.clamp(t / 3.0, max=1.0)[None]
    gz = torch.empty(n, H, device=dev, dtype=dt)
    gz[:, 0:3] = gz0[:, None]
    gz[:, 3] = 0.5 * (gz0 + ZG)
    gz[:, 4:6] = ZG
    frac = torch.arange(1, 5, device=dev, dtype=dt) / 4.0
    lift = L[:, None] * frac[None]
    gz[:, 6:10] = ZG + lift
    gxe = gx0 + dx
    ox_t = ox[:, None].repeat(1, H)
    ox_t[:, 5:] = torch.where(aligned[:, None], gxe[:, None], ox[:, None])
    oz_t = torch.zeros(n, H, device=dev, dtype=dt)
    succ = (outcome == 0)[:, None]
    oz_t[:, 6:10] = torch.where(succ, lift, torch.zeros_like(lift))
    slipm = (outcome == 2)[:, None]
    oz_t[:, 6:8] = torch.where(slipm, lift[:, :2], oz_t[:, 6:8])
    Y = torch.stack([gx, gz, ox_t, oz_t], -1).reshape(n, D_OUT)
    return Y, outcome


def sample_scene(n, gen, dev):
    def U(a, b):
        return a + (b - a) * torch.rand(n, generator=gen, device=dev)
    S = {}
    S['ox'] = U(-1.0, 1.0)
    S['w'] = U(0.05, 0.15)
    S['gz0'] = U(0.4, 0.6)
    S['L'] = U(0.1, 0.5)
    S['gx0'] = S['ox'] + U(-0.4, 0.4)
    S['v'] = U(0.0, 1.0)                                   # alignment draw
    S['sign'] = 1.0 - 2.0 * (torch.rand(n, generator=gen, device=dev) < 0.5).float()
    S['c'] = torch.randint(0, 5, (n,), generator=gen, device=dev)  # non-slippery colours 0..4
    S['u'] = U(0.0, 1.0)                                   # class draw for the data policy
    return S


def policy_classes(u, p):
    """data policy: success w.p. 1-p, miss w.p. (1-SLIP_SHARE)p, slip w.p. SLIP_SHARE*p."""
    cls = torch.zeros(u.shape[0], dtype=torch.long, device=u.device)
    cls[u < p] = 1
    cls[u < p * SLIP_SHARE] = 2
    return cls


def realize(S, cls, r=None, e=None, L=None, gz0=None, color=None):
    """policy alignment noise: |e|/w ~ U[0,1) for success/slip, U[1,2) for miss (same support for every p)."""
    if e is None:
        if r is None:
            r = torch.where(cls == 1, 1.0 + S['v'], S['v'])
        e = S['sign'] * r * S['w']
    if color is None:
        color = torch.where(cls == 2, torch.full_like(S['c'], SLIP_COLOR), S['c'])
    L = S['L'] if L is None else L
    gz0 = S['gz0'] if gz0 is None else gz0
    dx = S['ox'] + e - S['gx0']
    X = make_inputs(S['gx0'], gz0, S['ox'], S['w'], dx, L, color)
    Y, outcome = rollout(S['gx0'], gz0, S['ox'], S['w'], dx, L, color)
    ratio = (S['gx0'] + dx - S['ox']).abs() / S['w']
    return dict(X=X, Y=Y, outcome=outcome, r=ratio, L=L)


def train_data(p, seed, N, dev):
    """same scene draws for every p within a seed (paired design); only class membership differs."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(1000 + seed)
    S = sample_scene(N, gen, dev)
    return realize(S, policy_classes(S['u'], p))


def norm_stats(dev):
    gen_d = train_data(0.3, 999, 200000, dev)
    return dict(xm=gen_d['X'].mean(0), xs=gen_d['X'].std(0).clamp_min(0.05),
                ym=gen_d['Y'].mean(0), ys=gen_d['Y'].std(0).clamp_min(0.05))


def nx(X, nm):
    return (X - nm['xm']) / nm['xs']


def ny(Y, nm):
    return (Y - nm['ym']) / nm['ys']


def raw_y(y, nm):
    return y * nm['ys'] + nm['ym']


def test_sets(dev, n_pairs=512):
    gen = torch.Generator(device=dev)
    gen.manual_seed(12345)
    T = {}
    n_s, n_m, n_sl = 4096, 3072, 1024
    S = sample_scene(n_s + n_m + n_sl, gen, dev)
    cls = torch.cat([torch.zeros(n_s), torch.ones(n_m), 2 * torch.ones(n_sl)]).long().to(dev)
    T['bal'] = realize(S, cls)
    S = sample_scene(4096, gen, dev)
    T['near'] = realize(S, torch.zeros(4096, dtype=torch.long, device=dev), r=0.8 + 0.4 * S['v'])
    # paired counterfactual sets (same base scene within a pair)
    n = n_pairs
    S = sample_scene(n, gen, dev)
    rnd = lambda: torch.rand(n, generator=gen, device=dev)
    zero = torch.zeros(n, dtype=torch.long, device=dev)
    r_a = 0.05 + 0.65 * S['v']            # aligned (success)
    r_b = 1.3 + 0.6 * rnd()               # misaligned, same side (miss)
    P = {}
    P['miss'] = (realize(S, zero, r=r_a), realize(S, zero, r=r_b))
    de = (r_b - r_a) * S['w']             # input-copy control: same |delta e| but both aligned
    sg = 1.0 - 2.0 * (rnd() < 0.5).float()
    P['copy'] = (realize(S, zero, e=-0.5 * de * sg), realize(S, zero, e=0.5 * de * sg))
    L_hi, L_lo = 0.35 + 0.15 * rnd(), 0.10 + 0.05 * rnd()
    P['succ'] = (realize(S, zero, r=r_a, L=L_hi), realize(S, zero, r=r_a, L=L_lo))
    P['slip'] = (realize(S, zero, r=r_a), realize(S, zero, r=r_a, color=torch.full_like(S['c'], SLIP_COLOR)))
    gz_hi, gz_lo = 0.55 + 0.05 * rnd(), 0.40 + 0.05 * rnd()
    P['unrel'] = (realize(S, zero, r=r_a, gz0=gz_hi), realize(S, zero, r=r_a, gz0=gz_lo))
    T['pairs'] = P
    return T


# ----------------------------------------------------------------------------- model
def init_single(seed, dev):
    gen = torch.Generator(device=dev)
    gen.manual_seed(50000 + seed)
    dims = [D_IN] + [WIDTH] * DEPTH + [D_OUT]
    P = {}
    for l in range(DEPTH + 1):
        fi, fo = dims[l], dims[l + 1]
        std = math.sqrt(2.0 / fi) if l < DEPTH else math.sqrt(1.0 / fi)
        P[f'W{l}'] = torch.randn(fi, fo, generator=gen, device=dev) * std
        P[f'b{l}'] = torch.zeros(fo, device=dev)
    return P


def init_ensemble(seeds, dev):
    singles = {s: init_single(s, dev) for s in sorted(set(seeds))}
    keys = list(singles[seeds[0]].keys())
    return {k: torch.stack([singles[s][k] for s in seeds]).contiguous() for k in keys}


PKEYS = [f'{t}{l}' for l in range(DEPTH + 1) for t in ('W', 'b')]


def forward(P, x, hook=None, return_acts=False):
    """P: single model (W [fi,fo]) or ensemble (W [M,fi,fo]); x matches leading dims.
    hook(l, a, z) -> a' is applied to the post-GELU activation of hidden layer l (0-based)."""
    acts = []
    h = x
    for l in range(DEPTH):
        z = torch.matmul(h, P[f'W{l}']) + P[f'b{l}'].unsqueeze(-2)
        a = F.gelu(z)
        if hook is not None:
            a = hook(l, a, z)
        if return_acts:
            acts.append((a, z))
        h = a
    y = torch.matmul(h, P[f'W{DEPTH}']) + P[f'b{DEPTH}'].unsqueeze(-2)
    return (y, acts) if return_acts else y


def forward_from(P, l, a):
    """continue the forward pass from the post-activation of hidden layer l."""
    h = a
    for k in range(l + 1, DEPTH):
        h = F.gelu(torch.matmul(h, P[f'W{k}']) + P[f'b{k}'].unsqueeze(-2))
    return torch.matmul(h, P[f'W{DEPTH}']) + P[f'b{DEPTH}'].unsqueeze(-2)


def slice_model(P, m):
    return {k: v[m] for k, v in P.items()}


# ----------------------------------------------------------------------------- metrics
def R_obj(yr):
    return yr[..., IDX_OZ_LIFT].sum(-1)


def R_gz(yr):
    return yr[..., IDX_GZ_APPROACH].sum(-1)


def lifted_pred(yr, d):
    return yr[..., IDX_OZ_FINAL] > 0.5 * d['L']


def outcome_metrics(yr, d):
    """yr: raw predictions [..., n, D_OUT]. Returns tensors with the leading dims of yr."""
    lifted = lifted_pred(yr, d)
    out = d['outcome']
    succ, miss, slip, fail = out == 0, out == 1, out == 2, out != 0
    m = {}
    m['acc_s'] = lifted[..., succ].float().mean(-1)
    m['acc_f'] = (~lifted[..., fail]).float().mean(-1)
    if miss.any():
        m['acc_miss'] = (~lifted[..., miss]).float().mean(-1)
    if slip.any():
        m['acc_slip'] = (~lifted[..., slip]).float().mean(-1)
    m['opt'] = 1.0 - m['acc_f']
    if miss.any():
        m['opt_miss'] = 1.0 - m['acc_miss']
    if slip.any():
        m['opt_slip'] = 1.0 - m['acc_slip']
    ozf = yr[..., IDX_OZ_FINAL]
    m['opt_cont'] = (ozf[..., fail] / d['L'][fail]).mean(-1)       # predicted lift fraction on failures
    m['pes_cont'] = 1.0 - (ozf[..., succ] / d['L'][succ]).mean(-1)  # missing lift fraction on successes
    se = (yr - d['Y']) ** 2
    m['mse_s'] = se.mean(-1)[..., succ].mean(-1)
    m['mse_s_obj'] = se[..., IDX_OBJ].mean(-1)[..., succ].mean(-1)
    m['mse_f'] = se.mean(-1)[..., fail].mean(-1)
    return m


def binned_lift_prob(yr, d, edges):
    lifted = lifted_pred(yr, d).float()
    r = d['r']
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r >= lo) & (r < hi) & (d['outcome'] != 2)
        out.append(lifted[..., sel].mean(-1) if sel.any() else torch.full(lifted.shape[:-1], float('nan'), device=r.device))
    return torch.stack(out, -1)


def to_py(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
