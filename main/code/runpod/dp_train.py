"""Train the per-task LIBERO Diffusion Policy (dp_model.py) on GPU-resident data (dp_data.py output).
50k steps, batch 64, AdamW 1e-4 (betas 0.95/0.999, wd 1e-6), cosine with 500 warm-up steps, EMA (power 0.75),
random-shift augmentation (+-6 px via affine grid), bf16 autocast. Saves EMA weights (fp16) + normalization stats at
the checkpoint steps (stages) and a resumable full state every 5k steps.
Run:  $VP /workspace/code/dp_train.py --task 8
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dp_model import A_DIM, H_ACT, IMG, DiffusionPolicy, make_schedulers, to_unit

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, required=True)
ap.add_argument("--steps", type=int, default=50000)
ap.add_argument("--ckpts", default="5000,10000,20000,50000")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="/workspace/dp/data")
ap.add_argument("--out", default="/workspace/dp/ckpt")
ap.add_argument("--stop_at", type=int, default=0, help="exit after this many steps (same LR schedule as --steps)")
args = ap.parse_args()
out = os.path.join(args.out, f"t{args.task}_s{args.seed}")
os.makedirs(out, exist_ok=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
ckpt_steps = [int(x) for x in args.ckpts.split(",")]
DIMS = (256, 512, 1024)

d = np.load(os.path.join(args.data, f"t{args.task}.npz"))
imgs = torch.from_numpy(d["images"]).to(dev)                        # (N, 2, 128, 128, 3) uint8
prop_raw, act_raw = d["proprio"].astype(np.float32), d["actions"].astype(np.float32)
stats = dict(p_lo=prop_raw.min(0), p_hi=prop_raw.max(0), a_lo=act_raw.min(0), a_hi=act_raw.max(0))
prop = torch.from_numpy(to_unit(prop_raw, stats["p_lo"], stats["p_hi"]).astype(np.float32)).to(dev)
act = torch.from_numpy(to_unit(act_raw, stats["a_lo"], stats["a_hi"]).astype(np.float32)).to(dev)
# chunk index table: actions[t : t + 16] inside the episode, padded with the episode's last action
idx = []
for s, L in zip(d["ep_start"], d["ep_len"]):
    for t in range(L):
        idx.append([s + min(t + j, L - 1) for j in range(H_ACT)])
idx = torch.tensor(idx, device=dev)
N = idx.shape[0]
print(f"task {args.task}: {len(d['ep_len'])} demos, {N} samples, stats a_lo {stats['a_lo'].round(3)} a_hi {stats['a_hi'].round(3)}", flush=True)

net = DiffusionPolicy(dims=DIMS).to(dev)
ema = DiffusionPolicy(dims=DIMS).to(dev).eval()
ema.load_state_dict(net.state_dict())
for p in ema.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
from diffusers.optimization import get_scheduler
sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=500, num_training_steps=args.steps)
ddpm, _ = make_schedulers()
start = 0
resume = os.path.join(out, "resume.pt")
if os.path.exists(resume):
    ck = torch.load(resume, map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start = ck["step"]
    torch.set_rng_state(ck["rng_cpu"]); torch.cuda.set_rng_state(ck["rng_cuda"])
    print("resumed at", start, flush=True)


def ema_decay(step, power=0.75, max_value=0.9999):
    v = 1 - (1 + step) ** -power
    return max(0.0, min(v, max_value))


def augment(x):
    """x: (B, 2, 3, H, W) in [-1, 1]; per-image random translation of up to +-6 px (border padding)."""
    B2 = x.shape[0] * x.shape[1]
    flat = x.reshape(B2, 3, IMG, IMG)
    shift = (torch.rand(B2, 2, device=dev) * 2 - 1) * (6.0 / IMG) * 2
    theta = torch.zeros(B2, 2, 3, device=dev)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    return F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False).reshape(x.shape)


ema_params = [p for p in ema.parameters()]
net_params = [p.detach() for p in net.parameters()]
assert not list(net.buffers()), "EMA update assumes no buffers (GroupNorm only)"
log = open(os.path.join(out, "train_log.jsonl"), "a")
t_last, losses = time.time(), []
for step in range(start, args.stop_at or args.steps):
    b = torch.randint(0, N, (args.batch,), device=dev)
    t0 = idx[b, 0]
    x = imgs[t0].permute(0, 1, 4, 2, 3).float().div(127.5).sub(1.0)
    x = augment(x)
    a0 = act[idx[b]]
    noise = torch.randn_like(a0)
    ts = torch.randint(0, ddpm.config.num_train_timesteps, (args.batch,), device=dev)
    xt = ddpm.add_noise(a0, noise, ts)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net.unet(xt, ts, net.cond(x, prop[t0]))
    loss = F.mse_loss(pred.float(), noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    with torch.no_grad():
        dcy = ema_decay(step)
        torch._foreach_mul_(ema_params, dcy)
        torch._foreach_add_(ema_params, net_params, alpha=1 - dcy)
    losses.append(loss.item() if (step + 1) % 50 == 0 else None)
    s1 = step + 1
    if s1 % 500 == 0:
        ls = [l for l in losses if l is not None]
        rate = 500 / (time.time() - t_last)
        t_last, losses = time.time(), []
        rec = dict(step=s1, loss=float(np.mean(ls)) if ls else None, lr=sched.get_last_lr()[0], it_s=rate)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
    if s1 in ckpt_steps:
        torch.save(dict(ema={k: v.half() for k, v in ema.state_dict().items()}, stats=stats,
                        config=dict(dims=list(DIMS), task=args.task, seed=args.seed, step=s1, h_act=H_ACT, img=IMG)),
                   os.path.join(out, f"ckpt_{s1}.pt"))
        print("saved ckpt", s1, flush=True)
    if s1 % 5000 == 0 and s1 < args.steps:
        torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=s1,
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), resume + ".tmp")
        os.replace(resume + ".tmp", resume)
print("TRAIN_DONE", args.task, flush=True)
