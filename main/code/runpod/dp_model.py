"""Per-task LIBERO Diffusion Policy for B's M2 (CNN variant of Chi et al. 2023, as benchmarked in dp_bench.py).

Inputs: agentview + wrist images in the Cosmos Policy model view (flip -> JPEG95 -> 224 -> 90% centre crop -> 224,
see cosmos_utils.prepare_images_for_model) resized to 128 px, so the same network can read Cosmos-imagined frames
(mode C); proprio 9-d in Cosmos order (gripper 2, eef pos 3, eef quat 4). Output: 16-step action chunk, executed
open-loop (one Cosmos chunk). n_obs_steps = 1. DDPM 100 training steps (squaredcos_cap_v2, epsilon), DDIM 10 at test.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

H_ACT, A_DIM, P_DIM, IMG = 16, 7, 9, 128


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
        return self.b2(h * s + b) + self.res(x)


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
        for a, b in zip(rev[:-1], rev[1:]):
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


class DiffusionPolicy(nn.Module):
    def __init__(self, n_cam=2, dims=(256, 512, 1024)):
        super().__init__()
        self.enc = nn.ModuleList([resnet18_gn() for _ in range(n_cam)])
        self.unet = CondUnet1D(A_DIM, 512 * n_cam + P_DIM, dims)

    def cond(self, imgs, prop):
        """imgs: (B, n_cam, 3, H, W) float in [-1, 1]; prop: (B, 9) normalized."""
        return torch.cat([e(imgs[:, i]) for i, e in enumerate(self.enc)] + [prop], -1)


def make_schedulers():
    from diffusers import DDIMScheduler, DDPMScheduler
    kw = dict(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    return DDPMScheduler(**kw), DDIMScheduler(**kw, set_alpha_to_one=True, steps_offset=0)


def to_unit(x, lo, hi):
    return 2.0 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1.0


def from_unit(x, lo, hi):
    return (x + 1.0) / 2.0 * (hi - lo) + lo


def images_to_input(imgs_u8: torch.Tensor) -> torch.Tensor:
    """(B, n_cam, H, W, 3) uint8 -> (B, n_cam, 3, H, W) float in [-1, 1]."""
    return imgs_u8.permute(0, 1, 4, 2, 3).float().div(127.5).sub(1.0)


class PolicyRunner:
    """Inference wrapper: obs in the Cosmos model view (224 px uint8) -> 16 unnormalized LIBERO actions."""

    def __init__(self, ckpt_path, device="cuda", ddim_steps=10, seed=0):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in ck["stats"].items()}
        self.net = DiffusionPolicy(dims=tuple(ck["config"]["dims"])).to(device).eval()
        self.net.load_state_dict({k: v.float() for k, v in ck["ema"].items()})
        _, self.ddim = make_schedulers()
        self.ddim.set_timesteps(ddim_steps)
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    @torch.inference_mode()
    def act(self, primary224: np.ndarray, wrist224: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        im = np.stack([primary224, wrist224])[None]  # (1, 2, 224, 224, 3)
        x = torch.from_numpy(im).to(self.device)
        x = F.interpolate(x.permute(0, 1, 4, 2, 3).reshape(2, 3, 224, 224).float(), size=(IMG, IMG), mode="bilinear",
                          antialias=True, align_corners=False).clamp(0, 255).round()
        x = (x / 127.5 - 1.0).reshape(1, 2, 3, IMG, IMG)
        p = torch.from_numpy(to_unit(np.asarray(proprio, np.float32), self.stats["p_lo"], self.stats["p_hi"])[None]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = self.net.cond(x, p.float())
        a = torch.randn((1, H_ACT, A_DIM), device=self.device, generator=self.gen)
        for t in self.ddim.timesteps:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eps = self.net.unet(a, t.expand(1).to(self.device), g)
            a = self.ddim.step(eps.float(), t, a).prev_sample
        a = a[0].clamp(-1, 1).float().cpu().numpy()
        return from_unit(a, self.stats["a_lo"], self.stats["a_hi"])


def resize_224_to_128(img224: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(img224)).permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", antialias=True, align_corners=False)
    return x[0].permute(1, 2, 0).clamp(0, 255).round().byte().numpy()
