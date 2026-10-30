"""B E1 M3: evaluation bank for the failure-distance curves (DESIGN_main v1 section 0.3).
 grid : 128 scenes (generator 31337, the R2-9 scene distribution) x r in {0.60, 0.65, ..., 3.00} (49 points); for r >= 3
        only scenes with w <= 0.13 are used (flag). Stores f0 tokens, normalised actions and real f4 tokens.
 ceil : 22,000 scenes (generator 31338) with r ~ U[0, 3.2] for the information-ceiling classifier (trained on each
        model's imagined f4 in m3_e1_eval.py); stores f0, actions, real f4, labels.
Uses the PCA basis and token normalisation of the R2-9 banks (eval.pt of the same feature dir); nothing is refitted.
usage: python world_e1.py"""
import os as _os
# B guard: GPU 1 only, torch hub inside ~/pmx (see ../bpaths.py)
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, time
import torch
from world import FEAT_DIR, Encoder, sample_scene, sub, actions

GRID = [round(0.60 + 0.05 * i, 2) for i in range(49)]
N_SCENE, N_CEIL = 128, 22000


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    ev = torch.load(os.path.join(FEAT_DIR, 'eval.pt'), weights_only=False)
    enc = Encoder(dev)
    enc.mu, enc.V = ev['pca_mu'].to(dev), ev['pca_V'].to(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(31337)
    S = sample_scene(N_SCENE, gen, dev)
    f0 = enc.frames(S, torch.zeros(N_SCENE, device=dev), 0)
    grid_act, grid_f4, use = [], [], []
    for r in GRID:
        e = S['sign'] * r * S['w']
        grid_act.append(actions(S, e))
        grid_f4.append(enc.frames(S, e, 4))
        use.append((S['w'] <= 0.13) if r >= 3.0 else torch.ones(N_SCENE, dtype=torch.bool, device=dev))
    gen.manual_seed(31338)
    S2 = sample_scene(N_CEIL, gen, dev)
    r2 = 3.2 * S2['v']
    e2 = S2['sign'] * r2 * S2['w']
    cpu = lambda x: {k: cpu(v) for k, v in x.items()} if isinstance(x, dict) else x.cpu()
    out = dict(grid_r=torch.tensor(GRID), grid_f0=f0.cpu(), grid_act=torch.stack(grid_act).cpu(),          # [G, n, 4, 3]
               grid_f4=torch.stack(grid_f4).cpu(), grid_use=torch.stack(use).cpu(), grid_scene=cpu(S),
               ceil_f0=enc.frames(S2, torch.zeros(N_CEIL, device=dev), 0).cpu(), ceil_act=actions(S2, e2).cpu(),
               ceil_f4=enc.frames(S2, e2, 4).cpu(), ceil_r=r2.cpu(), ceil_y=(r2 < 1).cpu())
    torch.save(out, os.path.join(FEAT_DIR, 'eval_e1.pt'))
    print(f'grid {len(GRID)} x {N_SCENE} (r=3.00 uses {int(out["grid_use"][-1].sum())} scenes), ceil {N_CEIL} '
          f'({time.time() - t0:.0f}s)', flush=True)
    print('WORLD_E1_DONE')


if __name__ == '__main__':
    main()
