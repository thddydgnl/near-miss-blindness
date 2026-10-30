"""Generate R2-11 datasets: simulate (rotating-box env), render 112x112, encode with frozen DINOv2 ViT-S/14
(8x8 patch tokens), cache features + full ground-truth state logs.

Sets (episodes of T=40 control steps, 41 frames; OU teleop-like policy as in R2-2):
  train : 2500 episodes, world-model training
  id    : 150 held-out episodes, loss evaluation and CEM tasks
  ana   : 400 held-out episodes, probes / confusion / neuron analysis
GT logged per frame: pa, va, pb, vb (2), th, wb (1); per step: a (2), contact (step-level touch flag).
Usage: python gen_data.py --out /workspace/data [--scale 0.05]
"""
import argparse
import json
import math
import os
import time

import torch

import env as E

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dino(dev):
    m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    return m.to(dev).eval()


@torch.no_grad()
def encode(dino, pa, pb, th, bs=1024):
    """pa, pb: [M,2], th: [M] -> patch tokens [M,64,384] fp16 on CPU."""
    out = []
    mean, std = MEAN.to(pa.device), STD.to(pa.device)
    for i in range(0, pa.shape[0], bs):
        img = E.render(pa[i:i + bs], pb[i:i + bs], th[i:i + bs])
        x = (img - mean) / std
        f = dino.forward_features(x)['x_norm_patchtokens']
        out.append(f.half().cpu())
    return torch.cat(out, 0)


def encode_episodes(dino, sim):
    n, T1 = sim['pa'].shape[:2]
    Z = encode(dino, sim['pa'].reshape(-1, 2), sim['pb'].reshape(-1, 2), sim['th'].reshape(-1))
    return Z.view(n, T1, Z.shape[1], Z.shape[2])


def wrap_q(x):
    """wrap an angle difference to (-pi/4, pi/4] (square symmetry)."""
    q = math.pi / 2
    return x - q * torch.round(x / q)


def phys_stats(sim):
    pa, pb, th = sim['pa'], sim['pb'], sim['th']
    dth = wrap_q(th[:, 1:] - th[:, :-1]).abs() * 180 / math.pi
    gap = E.box_gap(pa, pb, th)
    wg = E.wall_gap_box(pb, th)
    bsp = (pb[:, 1:] - pb[:, :-1]).norm(dim=-1)
    return dict(
        agent_disp_px=float((pa[:, 1:] - pa[:, :-1]).norm(dim=-1).mean()),
        box_moving_frac=float((bsp > 0.5).float().mean()),
        box_speed_mean=float(bsp.mean()), box_speed_p95=float(bsp.flatten().quantile(0.95)),
        contact_step_frac=float(sim['contact'].float().mean()),
        contact_frame_frac=float((gap <= 0.5).float().mean()),
        gap_min=float(gap.min()),
        box_rot_frac_1deg=float((dth > 1).float().mean()), box_rot_frac_5deg=float((dth > 5).float().mean()),
        box_rot_p99_deg=float(dth.flatten().quantile(0.99)), box_rot_max_deg=float(dth.max()),
        box_at_wall_frac=float((wg <= 0.5).float().mean()), wall_gap_mean=float(wg.mean()),
        action_abs=float(sim['a'].abs().mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/workspace/data')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--T', type=int, default=40)
    ap.add_argument('--sets', default='train,id,ana')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    dino = load_dino(dev)
    sizes = dict(train=2500, id=150, ana=400)
    seeds = dict(train=1111, id=3333, ana=7777)
    stats = {}
    for name in args.sets.split(','):
        n = max(8, int(round(sizes[name] * args.scale)))
        sim = E.simulate(n, args.T, 'ou', seeds[name], dev)
        stats[name] = phys_stats(sim)
        # determinism check: re-simulating logged actions from logged states reproduces the next states exactly
        st = [sim[k][:, :-1].reshape(-1, *sim[k].shape[2:]) for k in ('pa', 'va', 'pb', 'vb', 'th', 'wb')]
        nx = E.control_step(*st, sim['a'].reshape(-1, 2))
        err = max(float((nx[0] - sim['pa'][:, 1:].reshape(-1, 2)).abs().max()),
                  float((nx[2] - sim['pb'][:, 1:].reshape(-1, 2)).abs().max()),
                  float((nx[4] - sim['th'][:, 1:].reshape(-1)).abs().max()))
        stats[name]['resim_err'] = err
        Z = encode_episodes(dino, sim)
        torch.save(dict(Z=Z, **{k: sim[k].cpu() for k in ('a', 'pa', 'va', 'pb', 'vb', 'th', 'wb', 'contact')}),
                   f'{args.out}/{name}.pt')
        if name == 'train':
            # chunked statistics (the container has a 31 GB memory limit)
            s1 = torch.zeros(Z.shape[-1], dtype=torch.float64)
            s2 = torch.zeros(Z.shape[-1], dtype=torch.float64)
            for i in range(0, Z.shape[0], 100):
                zf = Z[i:i + 100].double().view(-1, Z.shape[-1])
                s1 += zf.sum(0)
                s2 += (zf ** 2).sum(0)
            m = Z.shape[0] * Z.shape[1] * Z.shape[2]
            mu = (s1 / m).float()
            sd = ((s2 / m - (s1 / m) ** 2) * m / (m - 1)).clamp_min(0).sqrt().float().clamp_min(1e-3)
            torch.save(dict(mu=mu, sd=sd), f'{args.out}/stats.pt')
            se, cnt = 0.0, 0
            for i in range(0, Z.shape[0], 100):
                zs = (Z[i:i + 100].float() - mu) / sd
                se += float(((zs[:, 3:] - zs[:, 2:-1]) ** 2).double().sum())
                cnt += zs[:, 3:].numel()
            stats['train']['copy_mse_std'] = se / cnt
            stats['train']['feat_raw_absmax'] = float(Z.abs().max())
        del Z
        print(name, n, json.dumps(stats[name]), f'{time.time() - t0:.0f}s', flush=True)
        del sim
        torch.cuda.empty_cache()
    fn = f'{args.out}/data_stats.json'
    if os.path.exists(fn):
        prev = json.load(open(fn))
        prev.update(stats)
        stats = prev
    json.dump(stats, open(fn, 'w'), indent=1)
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
