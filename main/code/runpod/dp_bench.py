"""Throughput micro-benchmark for the planned per-task LIBERO Diffusion Policy (Chi et al. 2023, CNN variant).
No training run: synthetic tensors only, to measure optimizer steps/s and inference latency on this GPU.
Architecture: ResNet-18 (GroupNorm) per camera -> avgpool features, + proprio, as global condition of a
ConditionalUnet1D (FiLM); DDPM 100 train steps, DDIM 10 inference steps; action horizon 16 (matches the Cosmos chunk),
n_obs_steps 1 (the world model only returns the state at chunk boundaries, needed for closed-loop mode C).
Run:  $VP /workspace/code/dp_bench.py
"""
import json
import math
import time

import torch
import torch.nn as nn
import torchvision

dev = "cuda"


class SinEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        h = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / (h - 1))
        x = t[:, None].float() * f[None]
        return torch.cat([x.sin(), x.cos()], -1)


class CondRes(nn.Module):
    def __init__(self, cin, cout, cdim, k=5, g=8):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(g, cout), nn.Mish())
        self.b2 = nn.Sequential(nn.Conv1d(cout, cout, k, padding=k // 2), nn.GroupNorm(g, cout), nn.Mish())
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cdim, 2 * cout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.b1(x)
        s, b = self.film(c).unsqueeze(-1).chunk(2, 1)
        h = self.b2(h * s + b)
        return h + self.res(x)


class CondUnet1D(nn.Module):
    def __init__(self, adim, gdim, dims=(256, 512, 1024), sdim=128):
        super().__init__()
        self.temb = nn.Sequential(SinEmb(sdim), nn.Linear(sdim, 4 * sdim), nn.Mish(), nn.Linear(4 * sdim, sdim))
        cdim = sdim + gdim
        ch = [adim] + list(dims)
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(zip(ch[:-1], ch[1:])):
            last = i == len(dims) - 1
            self.down.append(nn.ModuleList([CondRes(a, b, cdim), CondRes(b, b, cdim),
                                            nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([CondRes(dims[-1], dims[-1], cdim), CondRes(dims[-1], dims[-1], cdim)])
        self.up = nn.ModuleList()
        rev = list(reversed(dims))
        for i, (a, b) in enumerate(zip(rev[:-1], rev[1:])):
            self.up.append(nn.ModuleList([CondRes(2 * a, b, cdim), CondRes(b, b, cdim), nn.ConvTranspose1d(b, b, 4, 2, 1)]))
        self.final = nn.Sequential(nn.Conv1d(dims[0], dims[0], 5, padding=2), nn.GroupNorm(8, dims[0]), nn.Mish(),
                                   nn.Conv1d(dims[0], adim, 1))

    def forward(self, x, t, g):
        x = x.transpose(1, 2)
        c = torch.cat([self.temb(t), g], -1)
        hs = []
        for r1, r2, dn in self.down:
            x = r2(r1(x, c), c)
            hs.append(x)
            x = dn(x)
        for m in self.mid:
            x = m(x, c)
        for r1, r2, upm in self.up:
            x = torch.cat([x, hs.pop()], 1)
            x = upm(r2(r1(x, c), c))
        return self.final(x).transpose(1, 2)


def resnet18_gn():
    m = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
    m.fc = nn.Identity()
    return m


class DP(nn.Module):
    def __init__(self, n_cam=2, pdim=9, adim=7, dims=(256, 512, 1024)):
        super().__init__()
        self.enc = nn.ModuleList([resnet18_gn() for _ in range(n_cam)])
        self.unet = CondUnet1D(adim, 512 * n_cam + pdim, dims)

    def cond(self, imgs, prop):
        return torch.cat([e(imgs[:, i]) for i, e in enumerate(self.enc)] + [prop], -1)


def bench(dims, res, B=64, H=16, steps=30):
    torch.manual_seed(0)
    m = DP(dims=dims).to(dev)
    nparam = sum(p.numel() for p in m.parameters()) / 1e6
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-6)
    imgs = torch.rand(B, 2, 3, res, res, device=dev)
    prop = torch.randn(B, 9, device=dev)
    act = torch.randn(B, H, 7, device=dev)
    abar = torch.cos(torch.linspace(0, 1, 101, device=dev) * math.pi / 2) ** 2

    def step():
        t = torch.randint(0, 100, (B,), device=dev)
        n = torch.randn_like(act)
        a = abar[t].view(-1, 1, 1)
        x = a.sqrt() * act + (1 - a).sqrt() * n
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((m.unet(x, t, m.cond(imgs, prop)) - n) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(5):
        step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    sps = steps / (time.time() - t0)
    # inference latency: 1 observation, 10 DDIM steps
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g = m.cond(imgs[:1], prop[:1])
        x = torch.randn(1, H, 7, device=dev)
        for _ in range(3):
            for k in range(10):
                m.unet(x, torch.full((1,), 90 - 10 * k, device=dev), g)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            g = m.cond(imgs[:1], prop[:1])
            for k in range(10):
                x = m.unet(x, torch.full((1,), 90 - 10 * k, device=dev), g)
        torch.cuda.synchronize()
        lat = (time.time() - t0) / 10
    return dict(dims=list(dims), res=res, batch=B, params_M=nparam, train_steps_per_s=sps, infer_ddim10_s=lat,
                vram_peak_gib=torch.cuda.max_memory_allocated() / 2**30)


out = [bench((256, 512, 1024), 128), bench((128, 256, 512), 96), bench((256, 512, 1024), 96)]
for o in out:
    print(o, flush=True)
json.dump(out, open("/workspace/results/throughput/dp_bench.json", "w"), indent=1)
