"""B E4a / E5 (M1): fine-tune the public DINO-WM PushT predictor and proprio / action encoders on precomputed DINOv2 features
of the fine-tuning windows (e4_m1_ftdata.py), DINO-WM loss (visual + proprio embedding MSE, teacher forcing, frames 1-3
from 0-2). DINOv2 and the decoder stay frozen. Conditions (DESIGN_main v3.1 section 9):
  S97 / S50   E4a, natural composition (uniform sampling over the 10,000 samples)
  rw          E5 (i)   S97 successes + S97's 300 failures, 8 failures in every batch of 32 (25%)
  near        E5 (ii)  S97 successes + 300 near-boundary counterfactual failures (r* ~ U[1, 2)), 25%
  cp          E5 (iii) the 300 near failures + their pair successes (r* ~ U[0.8, 1)): 8 + 8 + 16 S97 successes
  full        E5 (iv)  S97 successes + 300 full-range counterfactual failures (r* ~ U[1, 6]), 25%
  bw          E5 (v)   rw + T-block patch weights (lambda from E4c, --lam)
AdamW (default weight decay), lr 1e-4 for all trained parts, linear warm-up 100 steps then cosine to 0, batch 32, 3,000
steps, seed = batch order + dropout. bf16 autocast for the forward pass (fp32 master weights; 04:15Z, for speed on the shared GPU). Attention dropout is applied in training mode (SDPA with the same causal mask).
Writes ckpt/E4/m1/<cond>_s<seed>.pt (predictor, proprio_encoder, action_encoder state dicts + log).
usage: python e4_m1_finetune.py --cond S97 --seed 0 [--steps 3000] [--lam 1.0]"""
import os, sys, time, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r21lib as L
import e4lib as E4

FT_DIR = os.path.expanduser('~/pmx/B/feat/E4/m1')
CK_DIR = os.path.expanduser('~/pmx/B/ckpt/E4/m1')
BATCH, N_FAIL_B = 32, 8


def _sdpa_train(self, x):
    """vit.Attention.forward with SDPA: same causal frame mask; attention dropout active in training mode (as the
    original masked_fill / softmax / dropout implementation)."""
    from einops import rearrange
    B, T, C = x.size()
    x = self.norm(x)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), self.to_qkv(x).chunk(3, dim=-1))
    mask = (self.bias[0, 0, :T, :T] != 0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0, scale=self.scale)
    return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


def patch_attention_train():
    L.setup_paths()
    from models import vit
    vit.Attention.forward = _sdpa_train


def load_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    patch_attention_train()
    return L.load_task('pusht', load_dset=False)['model']


class Data:
    def __init__(self):
        self.vis = np.load(f'{FT_DIR}/ft_vis.npy', mmap_mode='r')
        self.svis = np.load(f'{FT_DIR}/ft_start_vis.npy')
        z = np.load(f'{FT_DIR}/ft_meta.npz')
        self.pro, self.act, self.bmask, self.win_of = z['pro'], z['act'], z['bmask'], z['win_of']
        self.sel = {k[4:]: z[k] for k in z.files if k.startswith('sel_')}

    def batch(self, idx, dev='cuda'):
        idx = np.asarray(idx)
        o = np.argsort(idx)                       # sorted access into the memmap
        v = np.empty((len(idx), 3, 196, 384), np.float16)
        v[o] = self.vis[idx[o]]
        vis = torch.from_numpy(np.concatenate([self.svis[self.win_of[idx]][:, None], v], 1)).to(dev).float()
        return (vis, torch.from_numpy(self.pro[idx]).to(dev), torch.from_numpy(self.act[idx]).to(dev),
                torch.from_numpy(self.bmask[idx]).to(dev))


def sampler(cond, D, rng):
    s97 = np.concatenate([D.sel['S97_s'], D.sel['S97_f']])
    s50 = np.concatenate([D.sel['S50_s'], D.sel['S50_f']])
    S = D.sel['S97_s']
    fail = dict(rw=D.sel['S97_f'], bw=D.sel['S97_f'], near=D.sel['cf_near'], full=D.sel['cf_full'])

    def draw():
        if cond == 'S97':
            return rng.choice(s97, BATCH)
        if cond == 'S50':
            return rng.choice(s50, BATCH)
        if cond == 'cp':
            k = rng.randint(0, len(D.sel['cf_near']), N_FAIL_B)
            return np.concatenate([D.sel['cf_near'][k], D.sel['cf_pair'][k], rng.choice(S, BATCH - 2 * N_FAIL_B)])
        return np.concatenate([rng.choice(fail[cond], N_FAIL_B), rng.choice(S, BATCH - N_FAIL_B)])
    return draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True, choices=['S97', 'S50', 'rw', 'near', 'cp', 'full', 'bw'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--bf16', type=int, default=1)
    ap.add_argument('--bench', type=int, default=0, help='time n steps (data / step) and exit without saving')
    args = ap.parse_args()
    os.makedirs(CK_DIR, exist_ok=True)
    out = f'{CK_DIR}/{args.cond}_s{args.seed}.pt'
    if os.path.exists(out):
        print('exists', out)
        return
    t0 = time.time()
    D = Data()
    m = load_model()
    for p in m.parameters():
        p.requires_grad_(False)
    mods = E4.trainable_modules(m)
    params = [p for mod in mods.values() for p in mod.parameters()]
    for p in params:
        p.requires_grad_(True)
    m.train()
    m.encoder.eval()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(1000 + args.seed)
    draw = sampler(args.cond, D, rng)
    opt = torch.optim.AdamW(params, lr=args.lr)
    log, n_fail_seen = [], 0
    lab = np.load(f'{FT_DIR}/ft_meta.npz')['lab'][:, 0]
    if args.bench:
        import time as _t
        td, ts = [], []
        for i in range(args.bench):
            a0 = _t.time(); idx = draw(); vis, pro, act, bm = D.batch(idx); torch.cuda.synchronize(); a1 = _t.time()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(args.bf16)):
                loss = E4.window_loss(m, vis, pro, act, None)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize(); a2 = _t.time()
            td.append(a1 - a0); ts.append(a2 - a1)
        print('BENCH', json.dumps(dict(bf16=args.bf16, data_s=float(np.median(td[5:])), step_s=float(np.median(ts[5:])),
                                       peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)), flush=True)
        return
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * max(0, step - 100) / max(1, args.steps - 100))))
        for g in opt.param_groups:
            g['lr'] = lr
        idx = draw()
        n_fail_seen += int((lab[idx] >= 1).sum())
        vis, pro, act, bm = D.batch(idx)
        w = None
        if args.cond == 'bw':
            w = 1.0 + (args.lam - 1.0) * bm[:, :3].float()
            w = w / w.mean(-1, keepdim=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(args.bf16)):
            loss = E4.window_loss(m, vis, pro, act, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            log.append(dict(step=step + 1, loss=float(loss), lr=lr, sec=time.time() - t0))
        if (step + 1) % 500 == 0:
            print(f'{args.cond} s{args.seed} step {step + 1} loss {float(loss):.5f} ({time.time() - t0:.0f}s)', flush=True)
    sd = {k: {kk: vv.detach().cpu() for kk, vv in mod.state_dict().items()} for k, mod in mods.items()}
    meta = dict(cond=args.cond, seed=args.seed, steps=args.steps, lr=args.lr, lam=args.lam, batch=BATCH,
                fail_fraction_seen=n_fail_seen / (args.steps * BATCH), time_s=time.time() - t0,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    torch.save(dict(**sd, log=log, meta=meta), out + '.tmp')
    os.replace(out + '.tmp', out)
    print('E4_FT_DONE', json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
