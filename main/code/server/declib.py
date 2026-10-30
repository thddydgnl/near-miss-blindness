"""B E3: the DINO-WM PushT checkpoint's own VQ-VAE decoder (10.1M parameters, stored in model_latest.pth, quantize=False):
DINOv2 patch features (N, 196, 384) -> 224 px RGB. Used to show frames to the VLM judge (d): real final frames are
encoded and decoded as well, so the decoder's blur enters real and imagined inputs alike (E0 report section 8.3)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r21lib as L

_DEC = {}


def load_decoder(device='cuda'):
    if 'd' not in _DEC:
        L.setup_paths()
        p = f"{BP.CKPT_BASE}/outputs/{L.TASKS['pusht']['model']}/checkpoints/model_latest.pth"
        payload = torch.load(p, map_location='cpu', weights_only=False)
        dec = payload['decoder'].to(device).eval()
        for q in dec.parameters():
            q.requires_grad_(False)
        _DEC['d'] = dec
        del payload
    return _DEC['d']


@torch.no_grad()
def decode(z, bs=64, device='cuda'):
    """z (N, 196, 384) any float dtype (cpu) -> uint8 (N, 224, 224, 3). The decoder outputs the normalised image
    (dataset transform Normalize(0.5, 0.5) -> [-1, 1]); inverted and rounded here."""
    dec = load_decoder(device)
    out = []
    for i in range(0, len(z), bs):
        x = torch.as_tensor(z[i:i + bs]).to(device).float()[:, None]
        img, _ = dec(x)
        img = ((img * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        out.append(img.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 224, 224, 3), np.uint8)
