"""B E4/E5 (M1) shared helpers: the fine-tuning trajectory pool F, DINO-WM training-unit windows (4 frames at frameskip 5,
20 actions), window simulation / rendering with per-frame block and agent patch masks, and the fine-tuning model wrapper
that trains the public checkpoint's predictor + proprio / action encoders on precomputed DINOv2 features with the DINO-WM
loss (visual + proprio embedding MSE, teacher forcing on num_hist 3 -> next frame).
Frozen by the E4/E5 operational definitions (worklogs/E4E5_server.md)."""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E

FS = R.FS
N_FRAMES = 4
W_ACT = N_FRAMES * FS                 # 20 actions per training window (the last chunk is not used by the loss)
LABEL_STEP = (N_FRAMES - 1) * FS       # outcome measured at step 15 (frame 3)
E1_RES = os.path.expanduser('~/pmx/B/results/E1/m1')
BLOCK_RGB = np.array([119, 136, 153])   # LightSlateGray
AGENT_RGB = np.array([65, 105, 225])    # RoyalBlue
PATCH = 16                              # 224 px / 14 patches


def f_split(dset_train, min_len=52):
    """train-split trajectories not used by E1 (J judges, C calibration, T auxiliary evaluation)."""
    sp = json.load(open(f'{E1_RES}/split_e1.json'))
    used = set(sp['J']) | set(sp['C']) | set(sp['T'])
    lens = np.array([dset_train.get_seq_length(i) for i in range(len(dset_train))])
    return sorted(int(i) for i in np.where(lens >= min_len)[0] if int(i) not in used)


def sample_windows(dset, trajs, n, seed):
    """n training-unit windows: trajectory uniform from trajs, start uniform in [0, len - 21]; expert = dataset actions."""
    rng = np.random.RandomState(seed)
    amean, astd = dset.action_mean.numpy(), dset.action_std.numpy()
    out = []
    for _ in range(n):
        bi = int(trajs[rng.randint(len(trajs))])
        L = dset.get_seq_length(bi)
        off = int(rng.randint(0, L - W_ACT))
        acts = dset.actions[bi, off:off + W_ACT].numpy() * astd + amean
        out.append(dict(traj=bi, offset=off, shape=dset.shapes[bi], H=3, init_state=dset.states[bi, off].numpy().astype(np.float64),
                        expert=acts.astype(np.float64)))
    return out


def patch_masks(img):
    """img (224, 224, 3) uint8 -> block / agent masks on the 14 x 14 patch grid (any pixel of that colour in the cell)."""
    def m(rgb):
        px = (np.abs(img.astype(np.int16) - rgb[None, None]).sum(-1) <= 30)
        return px.reshape(14, PATCH, 14, PATCH).any((1, 3)).reshape(-1)
    return m(BLOCK_RGB), m(AGENT_RGB)


def run_frames(env, shape, init, acts, render=True):
    """simulate acts (W_ACT, 2) env units from init; states at steps 0, 5, 10, 15 (+ final at 20); renders and masks of
    frames 1..3 (frame 0 is the shared start frame)."""
    from pymunk.vec2d import Vec2d
    R.sim_reset(env, shape, init)
    dt = 1.0 / env.sim_hz
    n_sub = env.sim_hz // env.control_hz
    states, imgs, masks = [env._get_obs().astype(np.float64)], [], []
    for t, a in enumerate(acts):
        act = env.agent.position + np.array(a) * env.action_scale
        for _ in range(n_sub):
            acc = env.k_p * (act - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acc * dt
            env.space.step(dt)
        if (t + 1) % FS == 0 and (t + 1) // FS < N_FRAMES:
            states.append(env._get_obs().astype(np.float64))
            if render:
                im = env._render_frame('rgb_array')
                imgs.append(im)
                masks.append(patch_masks(im))
    return np.stack(states), imgs, masks


def proprio_of(st):
    return np.concatenate([st[..., :2], st[..., 5:]], -1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ fine-tuning model
def trainable_modules(m):
    return dict(predictor=m.predictor, proprio_encoder=m.proprio_encoder, action_encoder=m.action_encoder)


def load_variant(m, path):
    """load a fine-tuned state (predictor + proprio / action encoders) into the model in place."""
    sd = torch.load(path, map_location='cuda', weights_only=False)
    for k, mod in trainable_modules(m).items():
        mod.load_state_dict(sd[k])


def snapshot(m):
    return {k: {kk: vv.detach().clone() for kk, vv in mod.state_dict().items()} for k, mod in trainable_modules(m).items()}


def restore(m, snap):
    for k, mod in trainable_modules(m).items():
        mod.load_state_dict(snap[k])


def window_loss(m, vis, pro, act, w_patch=None):
    """DINO-WM loss on precomputed features. vis (B, 4, 196, 384), pro (B, 4, 4) normalised, act (B, 4, 10) normalised.
    z = [visual | proprio-emb (tiled) | action-emb (tiled)]; predict frames 1..3 from 0..2; MSE on visual + proprio dims
    (action dims excluded), target detached (as VWorldModel.forward). w_patch (B, 3, 196): optional per-patch weights
    (mean 1 per frame) for the boundary-weighted loss (v)."""
    zp = m.proprio_encoder(pro)
    za = m.action_encoder(act)
    P = vis.shape[2]
    z = torch.cat([vis, zp[:, :, None].expand(-1, -1, P, -1), za[:, :, None].expand(-1, -1, P, -1)], -1)
    z_pred = m.predict(z[:, :3])
    d_act = za.shape[-1]
    e = (z_pred[..., :-d_act] - z[:, 1:, :, :-d_act].detach()) ** 2          # (B, 3, P, 394)
    if w_patch is None:
        return e.mean()
    return (e.mean(-1) * w_patch).mean()
