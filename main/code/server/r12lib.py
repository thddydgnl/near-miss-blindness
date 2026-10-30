"""R2-12 library: near-miss optimism of the public DINO-WM PushT checkpoint when it is used to evaluate policies.
Builds on r21lib (copied unchanged from R2-1: model loading, env, SDPA attention patch). Runs on the pod only.

Pieces
- episodes: start state + expert action segment (dataset actions) from the PushT dataset; goal = sim outcome of the
  expert segment (same as DINO-WM goal_source='dset').
- policies: perturbations of the expert segment in env action units (relative displacement / 100).
- sim: fast PushT replay (verified against env.rollout in the smoke test) + final-frame rendering.
- world model: open-loop DINO-WM rollout from the 1-frame start observation (as in DINO-WM planning).
- judges: (a) goal-conditioned success classifier on DINOv2 patch features, (b) goal-distance thresholds.
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21lib as L

import bpaths as BP    # B E0 port: server paths
RES = BP.RES
FEAT = BP.FEAT
FS = 5                       # frameskip of the checkpoint
POS_THR, ANG_THR = 20.0, np.pi / 9   # PushT evaluator (env.eval_state)


# ------------------------------------------------------------------------------------------------ evaluator
OBJ_POS_THR = 20.0          # B E0: object-pose-only definition uses the evaluator's 20 px on the block position alone


def margin_parts(g, f):
    """Same formulas as PushTWrapper.eval_state. Returns r = max(pos/20, ang/(pi/9)) (success iff r < 1) and parts.
    B E0 addition: object-pose-only failure distance r_obj = max(|block xy diff|/20, ang/(pi/9)) (agent ignored)."""
    g = np.asarray(g, np.float64)
    f = np.asarray(f, np.float64)
    pos = float(np.linalg.norm(g[:4] - f[:4]))
    ang = abs(g[4] - f[4])
    ang = float(min(ang, 2 * np.pi - ang))
    agent = float(np.linalg.norm(g[:2] - f[:2]))
    block = float(np.linalg.norm(g[2:4] - f[2:4]))
    return dict(r=max(pos / POS_THR, ang / ANG_THR), pos=pos, ang=ang, agent=agent, block=block,
                r_pos=pos / POS_THR, r_ang=ang / ANG_THR,
                r_obj=max(block / OBJ_POS_THR, ang / ANG_THR), r_obj_pos=block / OBJ_POS_THR)


def r_bin(r):
    if r < 1:
        return 'success'
    if r < 2:
        return 'near'
    if r < 3:
        return 'mid'
    return 'far'


# ------------------------------------------------------------------------------------------------ episodes
_DS = {}


def load_traj_dsets():
    """Trajectory-level PushT datasets (train, valid) without the model (CPU workers)."""
    if 'train' not in _DS:
        L.setup_paths()
        from datasets.pusht_dset import PushTDataset
        root = os.environ['DATASET_DIR'] + '/pusht_noise'
        _DS['train'] = PushTDataset(data_path=root + '/train', with_velocity=True)
        _DS['valid'] = PushTDataset(data_path=root + '/val', with_velocity=True)
    return _DS


def sample_episodes(dset, n, H, seed, exclude=None):
    """n episodes: (trajectory, offset) uniform as in R2-1 sample_pairs (offset <= len - H*FS - 1).
    exclude: trajectory indices not to use. Returns list of dicts: traj, offset, shape, init_state (7,),
    expert (H*FS, 2) env-unit actions."""
    rng = np.random.RandomState(seed)
    T = H * FS
    lens = np.array([dset.get_seq_length(i) for i in range(len(dset))])
    ok = np.where(lens - T - 1 >= 0)[0]
    if exclude is not None:
        ok = np.setdiff1d(ok, np.asarray(list(exclude), dtype=int))
    amean, astd = dset.action_mean.numpy(), dset.action_std.numpy()
    eps = []
    for j in range(n):
        bi = int(ok[rng.randint(len(ok))])
        off = int(rng.randint(0, lens[bi] - T))      # 0 .. len-T-1
        acts = dset.actions[bi, off:off + T].numpy() * astd + amean
        eps.append(dict(traj=bi, offset=off, shape=dset.shapes[bi], H=H,
                        init_state=dset.states[bi, off].numpy().astype(np.float64),
                        expert=acts.astype(np.float64)))
    return eps


# ------------------------------------------------------------------------------------------------ policies
FAMILIES = ('iid', 'offset', 'scale', 'gross', 'clean', 'mix', 'expert')


def make_actions(expert, fam, p, rng):
    """expert (T,2) env units (displacement/100 per control step). p: dict of parameters in world units.
    iid: + N(0, sigma) per step; offset: + beta*u, beta~U(0,beta_max), u random unit vector (per episode);
    scale: * (1+k), k~U(-k_max,k_max) (per episode); gross: with prob q + N(0, sig_big) per step, else + N(0, sig_small);
    clean: + N(0, sig_small). Returns (actions, info)."""
    T = expert.shape[0]
    if fam == 'expert':
        return expert.copy(), {}
    if fam == 'iid':
        return expert + rng.normal(0, p['sigma'], size=(T, 2)) / 100.0, {}
    if fam == 'offset':
        b = rng.uniform(0, p['beta_max'])
        u = rng.normal(size=2)
        u /= np.linalg.norm(u) + 1e-12
        return expert + (b * u)[None] / 100.0, {'beta': float(b)}
    if fam == 'scale':
        k = rng.uniform(-p['k_max'], p['k_max'])
        return expert * (1.0 + k), {'k': float(k)}
    if fam == 'gross':
        bad = rng.uniform() < p['q']
        s = p['sig_big'] if bad else p['sig_small']
        return expert + rng.normal(0, s, size=(T, 2)) / 100.0, {'corrupted': bool(bad)}
    if fam == 'clean':
        return expert + rng.normal(0, p['sig_small'], size=(T, 2)) / 100.0, {}
    if fam == 'mix':     # with prob q a gross corruption (far-miss), else offset(beta_max) + N(0, sig_small)
        if rng.uniform() < p['q']:
            return expert + rng.normal(0, p['sig_big'], size=(T, 2)) / 100.0, {'corrupted': True}
        a = expert.copy()
        b = 0.0
        if p.get('beta_max', 0) > 0:
            b = rng.uniform(0, p['beta_max'])
            u = rng.normal(size=2)
            u /= np.linalg.norm(u) + 1e-12
            a = a + (b * u)[None] / 100.0
        if p.get('sig_small', 0) > 0:
            a = a + rng.normal(0, p['sig_small'], size=(T, 2)) / 100.0
        return a, {'corrupted': False, 'beta': float(b)}
    raise ValueError(fam)


# ------------------------------------------------------------------------------------------------ simulator
_ENV = {}


def get_env():
    if 'e' not in _ENV:
        L.setup_paths()
        _ENV['e'] = L.make_env('pusht')
    return _ENV['e']


def sim_reset(env, shape, init_state):
    """Same as env.prepare(seed, init_state) (reset -> _setup, _set_state incl. its one physics step)."""
    env.update_env({'shape': shape})
    env._setup()
    env._set_state(np.asarray(init_state, dtype=np.float64))
    env.n_contact_points = 0


def sim_run(env, shape, init_state, acts, render=True):
    """Fast replay (R2-1 replay_final_states without per-step reward/render). Returns final state (7,), image or None."""
    from pymunk.vec2d import Vec2d
    sim_reset(env, shape, init_state)
    dt = 1.0 / env.sim_hz
    n_sub = env.sim_hz // env.control_hz
    for a in acts:
        act = np.array(a) * env.action_scale
        if env.relative:
            act = env.agent.position + act
        for _ in range(n_sub):
            acc = env.k_p * (act - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acc * dt
            env.space.step(dt)
    st = env._get_obs().astype(np.float64)
    img = env._render_frame('rgb_array') if render else None
    return st, img


def start_obs(env, shape, init_state):
    sim_reset(env, shape, init_state)
    st = env._get_obs().astype(np.float64)
    img = env._render_frame('rgb_array')
    proprio = np.concatenate([st[:2], st[5:]]).astype(np.float32)
    return st, img, proprio


def episode_job(args):
    """One episode in a CPU worker. args: (ep dict, list of (policy name, actions (T,2)), render).
    Returns dict with start img/proprio, goal state/img/proprio, per-policy final states/imgs/proprios, margins."""
    ep, seqs, render = args
    torch.set_num_threads(1)
    env = get_env()
    out = {}
    if render:
        s0, img0, p0 = start_obs(env, ep['shape'], ep['init_state'])
        out['start_img'], out['start_proprio'] = img0, p0
    gst, gimg = sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=render)
    out['goal_state'], out['goal_img'] = gst, gimg
    out['goal_proprio'] = np.concatenate([gst[:2], gst[5:]]).astype(np.float32)
    fin, imgs, pros, mps = [], [], [], []
    for name, acts in seqs:
        st, img = sim_run(env, ep['shape'], ep['init_state'], acts, render=render)
        fin.append(st)
        imgs.append(img)
        pros.append(np.concatenate([st[:2], st[5:]]).astype(np.float32))
        mps.append(margin_parts(gst, st))
    out['final_states'] = np.stack(fin)
    out['final_imgs'] = np.stack(imgs) if render else None
    out['final_proprios'] = np.stack(pros)
    out['margins'] = mps
    return out


def pool(n=None):
    from multiprocessing import get_context
    n = n or BP.NPROC          # B E0 port: CPU shared on the server (was min(24, cpu//2) on the pod)
    return get_context('spawn').Pool(n)


# ------------------------------------------------------------------------------------------------ world model
def load_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    L.patch_attention(True)
    T = L.load_task('pusht', load_dset=True)
    return T


def to_model_actions(pre, acts):
    """env-unit actions (N, H*FS, 2) -> normalized model actions (N, H, FS*2)."""
    a = torch.as_tensor(np.asarray(acts), dtype=torch.float32)
    a = pre.normalize_actions(a)
    N, TT, D = a.shape
    return a.reshape(N, TT // FS, FS * D)


@torch.no_grad()
def encode_frames(m, pre, imgs, proprios, bs=256):
    """imgs (N,H,W,3) uint8, proprios (N,4) raw -> visual (N,196,384) fp16 cpu, proprio-emb (N,10) fp32 cpu."""
    vs, ps = [], []
    for i in range(0, len(imgs), bs):
        o = pre.transform_obs({'visual': np.asarray(imgs[i:i + bs])[:, None],
                               'proprio': np.asarray(proprios[i:i + bs])[:, None]})
        o = {k: v.float().cuda() for k, v in o.items()}
        z = m.encode_obs(o)
        vs.append(z['visual'][:, 0].half().cpu())
        ps.append(z['proprio'][:, 0].float().cpu())
    return torch.cat(vs), torch.cat(ps)


@torch.no_grad()
def imagine(m, pre, start_imgs, start_proprios, acts, bs=128):
    """Open-loop DINO-WM rollout from 1-frame start observations (as in DINO-WM planning).
    start_imgs (N,H,W,3), start_proprios (N,4), acts (N,H*FS,2) env units.
    Returns predicted final visual (N,196,384) fp16 cpu and proprio-emb (N,10) fp32 cpu."""
    vs, ps = [], []
    for i in range(0, len(acts), bs):
        o = pre.transform_obs({'visual': np.asarray(start_imgs[i:i + bs])[:, None],
                               'proprio': np.asarray(start_proprios[i:i + bs])[:, None]})
        o = {k: v.float().cuda() for k, v in o.items()}
        a = to_model_actions(pre, acts[i:i + bs]).cuda()
        z_obses, _ = m.rollout(obs_0=o, act=a)
        vs.append(z_obses['visual'][:, -1].half().cpu())
        ps.append(z_obses['proprio'][:, -1].float().cpu())
    return torch.cat(vs), torch.cat(ps)


# ------------------------------------------------------------------------------------------------ judges
def d_vis(z, zg):
    """mean squared patch-feature distance (N,) ; z, zg (N,P,C)."""
    return ((z.float() - zg.float()) ** 2).mean((1, 2))


def d_full(z, zg, p, pg, alpha=1.0):
    """DINO-WM planning objective (mode last): visual MSE + alpha * proprio-emb MSE."""
    return d_vis(z, zg) + alpha * ((p.float() - pg.float()) ** 2).mean(-1)


class PairJudge(nn.Module):
    """Goal-conditioned success classifier on DINOv2 patch features: tokens [z, zg, z-zg] -> 2-layer transformer -> CLS.
    Outputs (success logit, predicted log-margin)."""

    def __init__(self, C=384, d=128, P=196, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(3 * C), nn.Linear(3 * C, d))
        self.pos = nn.Parameter(torch.zeros(1, P, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, z, zg):
        z, zg = z.float(), zg.float()
        x = self.inp(torch.cat([z, zg, z - zg], -1)) + self.pos
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        h = self.enc(x)[:, 0]
        o = self.head(h)
        return o[:, 0], o[:, 1]


@torch.no_grad()
def judge_prob(net, z, zg, bs=512):
    net.eval()
    out = []
    for i in range(0, len(z), bs):
        lg, _ = net(z[i:i + bs].cuda(), zg[i:i + bs].cuda())
        out.append(torch.sigmoid(lg).float().cpu())
    return torch.cat(out).numpy()


def train_judge(z_tr, zg_tr, y_tr, lr_tr, z_va, zg_va, y_va, epochs=30, d=128, layers=2, seed=0, log=print):
    """BCE (class-balanced) on true success + 0.3 * SmoothL1 on clipped log-margin (auxiliary, same ground truth).
    Model selection: epoch with best held-out balanced accuracy. Returns (net, history)."""
    torch.manual_seed(seed)
    net = PairJudge(d=d, layers=layers).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    n = len(y_tr)
    steps = epochs * math.ceil(n / 128)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    ytr = torch.as_tensor(y_tr, dtype=torch.float32)
    lrt = torch.as_tensor(np.clip(lr_tr, -2.5, 2.5), dtype=torch.float32)
    pw = torch.tensor(float((1 - ytr.mean()) / ytr.mean().clamp(min=1e-3))).cuda()
    best, best_state, hist = -1, None, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tl = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            lg, lr = net(z_tr[idx].cuda(), zg_tr[idx].cuda())
            yb = ytr[idx].cuda()
            loss = F.binary_cross_entropy_with_logits(lg, yb, pos_weight=pw) + 0.3 * F.smooth_l1_loss(lr, lrt[idx].cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            tl += float(loss) * len(idx)
        pv = judge_prob(net, z_va, zg_va)
        ba = balanced_acc(pv > 0.5, y_va)
        acc = float(((pv > 0.5) == (np.asarray(y_va) > 0.5)).mean())
        hist.append(dict(epoch=ep, train_loss=tl / n, val_bacc=ba, val_acc=acc))
        log(f'  judge epoch {ep} loss {tl / n:.4f} val acc {acc:.4f} bacc {ba:.4f}')
        if ba > best:
            best = ba
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


def balanced_acc(pred, y):
    pred = np.asarray(pred).astype(bool)
    y = np.asarray(y) > 0.5
    tpr = (pred & y).sum() / max(1, y.sum())
    tnr = (~pred & ~y).sum() / max(1, (~y).sum())
    return float(0.5 * (tpr + tnr))


def best_threshold(d, y):
    """tau maximizing balanced accuracy of (d < tau) vs success (real frames)."""
    d = np.asarray(d, np.float64)
    y = np.asarray(y) > 0.5
    cand = np.unique(np.quantile(d, np.linspace(0.001, 0.999, 999)))
    bas = [balanced_acc(d < t, y) for t in cand]
    k = int(np.argmax(bas))
    return float(cand[k]), float(bas[k])


def jdump(obj, path):
    L.jdump(obj, path)
