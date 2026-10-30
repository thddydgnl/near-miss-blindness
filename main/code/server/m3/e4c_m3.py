"""B E4c (M3, descriptive; DESIGN_main v3.1 section 9): where the R2-9 predictor's training loss puts its weight.
Data = the p = 0.03 base table (the base data of E5) of each data seed; per episode the normalised token innovations
dz_t = z(t+1) - z(t), t = 0..3 (the loss targets are frames 1..4).
  1. shares of the innovation energy |dz|^2 on object tokens, gripper tokens (not object) and background (8 x 8 token
     masks from the scene geometry, target frame)
  2. the same shares of the E1 p = 0.03 model's own one-step prediction error (teacher forcing)
  3. lambda for the boundary-weighted loss (v): object tokens get weight lambda so that they hold 50% of the weighted
     innovation energy (lambda = E_rest / E_object, at least 1); pooled over the three seeds
Writes results/E4/m3/e4c_m3.json. usage: python e4c_m3.py"""
import os as _os
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, time, json
import torch
from wm import Predictor, load_norm
from world import FEAT_DIR, N_TRAIN

EXT = os.path.expanduser(os.environ.get('B_E5_FEAT', '~/pmx/B/feat/E5/m3'))
CK1 = os.path.expanduser('~/pmx/B/ckpt/E1/m3')
OUT = os.path.expanduser('~/pmx/B/results/E4/m3')


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    mu, sd = load_norm(dev)
    tot = dict(obj=0.0, grip=0.0, bg=0.0)
    err = dict(obj=0.0, grip=0.0, bg=0.0)
    per_seed = {}
    for s in (0, 1, 2):
        B = torch.load(os.path.join(FEAT_DIR, f'bank_s{s}_N{N_TRAIN}.pt'), weights_only=False)
        X = torch.load(f'{EXT}/ext_s{s}.pt', weights_only=False)
        Fb = B['F'].to(dev)
        tab = {k: v.to(dev) for k, v in X['tabs']['p0.03_base'].items()}
        obj, grip = (m.to(dev) for m in X['masks']['p0.03_base'])
        seq = torch.cat([tab['f0'][:, None], tab['f']], 1)
        model = Predictor().to(dev).eval()
        model.load_state_dict(torch.load(f'{CK1}/p0.03_std_1x_s{s}.pt', map_location=dev, weights_only=False)['state'])
        st = dict(obj=0.0, grip=0.0, bg=0.0)
        with torch.no_grad():
            for i in range(0, seq.shape[0], 2048):
                sl = slice(i, i + 2048)
                for t in range(4):
                    z0 = (Fb[seq[sl, t]].float() - mu) / sd
                    z1 = (Fb[seq[sl, t + 1]].float() - mu) / sd
                    e = ((z1 - z0) ** 2).sum(-1)                              # (n, 64)
                    pe = ((model(z0, tab['act'][sl, t]) - z1) ** 2).sum(-1)
                    o, g = obj[sl, t], grip[sl, t] & ~obj[sl, t]
                    bgm = ~(o | g)
                    for k, mk in (('obj', o), ('grip', g), ('bg', bgm)):
                        v = float(e[mk].sum())
                        tot[k] += v
                        st[k] += v
                        err[k] += float(pe[mk].sum())
        S_ = sum(st.values())
        per_seed[s] = {k: v / S_ for k, v in st.items()}
        del Fb, B, X
    T, P = sum(tot.values()), sum(err.values())
    lam = max(1.0, (tot['grip'] + tot['bg']) / tot['obj']) if tot['obj'] > 0 else 1.0
    out = dict(innovation_energy_share={k: v / T for k, v in tot.items()}, per_seed_innovation_share=per_seed,
               prediction_error_share={k: v / P for k, v in err.items()}, lambda_obj=float(lam),
               lambda_rule='object tokens hold 50% of the weighted innovation energy (lambda = E_rest / E_object, >= 1)',
               time_s=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f'{OUT}/e4c_m3.json', 'w'), indent=1)
    print('E4C_M3_DONE', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
