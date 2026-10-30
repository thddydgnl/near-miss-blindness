"""B E5 M3 evaluation: the E1 grid (49 x 128, eval_e1.pt) and the R2-9 policy bank (6 policies x 1,024 scenes) for the
prescription models, judged by the frozen R2-9 classifier (same as m3_e1_eval.py / m3_e2_eval.py, no ceiling).
Writes results/E5/m3/e5eval_<model>.npz (grid_imag_logit [G, n], pol_imag_logit [6, 1024]).
usage: python m3_e5_eval.py [model,...]  (default: every e5_*.pt in ckpt/E5/m3)"""
import os as _os
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, time, glob
import numpy as np
import torch
from wm import CK, Classifier, Predictor, load_norm
from world import FEAT_DIR, POLICIES
from evaluate import imagine

CK5 = os.path.expanduser(os.environ.get('B_E5_CK', '~/pmx/B/ckpt/E5/m3'))
OUT = os.path.expanduser(os.environ.get('B_E5_OUT', '~/pmx/B/results/E5/m3'))


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else sorted(os.path.basename(f)[:-3] for f in glob.glob(f'{CK5}/e5_*.pt'))
    os.makedirs(OUT, exist_ok=True)
    G = torch.load(os.path.join(FEAT_DIR, 'eval_e1.pt'), weights_only=False)
    Ev = torch.load(os.path.join(FEAT_DIR, 'eval.pt'), weights_only=False)['E']
    mu, sd = load_norm(dev)
    nz = lambda x: (x.to(dev).float() - mu) / sd
    clf = Classifier().to(dev).eval()
    clf.load_state_dict(torch.load(os.path.join(CK, 'classifier.pt'), map_location=dev))
    for p_ in clf.parameters():
        p_.requires_grad_(False)
    nG = G['grid_act'].shape[0]
    z0 = nz(G['grid_f0'])
    zp0 = nz(Ev['pol_f0'])
    pols = list(POLICIES)
    for name in names:
        o = f'{OUT}/e5eval_{name}.npz'
        if os.path.exists(o):
            continue
        t0 = time.time()
        path = f'{CK5}/{name}.pt' if os.path.exists(f'{CK5}/{name}.pt') else os.path.join(CK, f'{name}.pt')
        ck = torch.load(path, map_location=dev, weights_only=False)
        model = Predictor().to(dev).eval()
        model.load_state_dict(ck['state'])
        with torch.no_grad():
            gl = torch.stack([imagine(model, clf, z0, G['grid_act'][g].to(dev)) for g in range(nG)]).cpu().numpy()
            pl = torch.stack([imagine(model, clf, zp0, Ev['pol'][p]['act'].to(dev)) for p in pols]).cpu().numpy()
        np.savez_compressed(o, grid_imag_logit=gl, pol_imag_logit=pl)
        print(f'{name}: evaluated ({time.time() - t0:.0f}s)', flush=True)
    print('M3_E5_EVAL_DONE', flush=True)


if __name__ == '__main__':
    main()
