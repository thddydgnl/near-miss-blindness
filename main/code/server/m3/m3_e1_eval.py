"""B E1 M3: failure-distance curves of the 9 controlled models (p in {0.03, 0.1, 0.3} x seed {0, 1, 2}).
Per model: imagine the 49 x 128 grid (4-step autoregressive rollout from the real f0 tokens) and judge the imagined
final frame with the frozen R2-9 classifier; the same classifier on the real f4 gives the real-frame judgment.
Information ceiling: the same classifier architecture and recipe (R2-9 wm.train_classifier: Adam 1e-3, 3,000 steps,
batch 512, seed 0) trained on this model's IMAGINED f4 of the 22,000 ceiling scenes (20,000 train / 2,000 validation),
then applied to the grid's imagined f4.
Writes results/<stage>/m3/e1_<model>.npz (+ json summary). usage: python m3_e1_eval.py [model,...]"""
import os as _os
# B guard: GPU 1 only, torch hub inside ~/pmx (see ../bpaths.py)
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from wm import CK, OUT, Classifier, Predictor, load_norm
from world import FEAT_DIR

MODELS = [f'p{p}_std_1x_s{s}' for p in ('0.03', '0.1', '0.3') for s in (0, 1, 2)]


@torch.no_grad()
def rollout_final(model, z0, acts, bs=4096):
    out = []
    for i in range(0, z0.shape[0], bs):
        out.append(model.rollout(z0[i:i + bs], acts[i:i + bs])[-1])
    return torch.cat(out)


def train_ceiling(Xtr, ytr, Xva, yva, dev):
    torch.manual_seed(0)
    clf = Classifier().to(dev)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(3000):
        i = torch.randint(0, len(Xtr), (512,), device=dev)
        loss = F.binary_cross_entropy_with_logits(clf(Xtr[i]), ytr[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        acc = float(((clf(Xva) > 0).float() == yva).float().mean())
    return clf, acc


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    G = torch.load(os.path.join(FEAT_DIR, 'eval_e1.pt'), weights_only=False)
    mu, sd = load_norm(dev)
    nz = lambda x: (x.to(dev).float() - mu) / sd
    clf = Classifier().to(dev).eval()
    clf.load_state_dict(torch.load(os.path.join(CK, 'classifier.pt'), map_location=dev))
    for p_ in clf.parameters():
        p_.requires_grad_(False)
    nG, nS = G['grid_act'].shape[:2]
    z0 = nz(G['grid_f0'])
    real_logit = torch.stack([clf(nz(G['grid_f4'][g])) for g in range(nG)]).cpu().numpy()          # [G, n]
    for name in names:
        o = os.path.join(OUT, f'e1_{name}.npz')
        if os.path.exists(o):
            print('exists', o)
            continue
        t0 = time.time()
        ck = torch.load(os.path.join(CK, f'{name}.pt'), map_location=dev, weights_only=False)
        model = Predictor().to(dev).eval()
        model.load_state_dict(ck['state'])
        zi = torch.stack([rollout_final(model, z0, G['grid_act'][g].to(dev)) for g in range(nG)])     # [G, n, 64, 64]
        with torch.no_grad():
            imag_logit = torch.stack([clf(zi[g]) for g in range(nG)]).cpu().numpy()
            zc = rollout_final(model, nz(G['ceil_f0']), G['ceil_act'].to(dev))
        y = G['ceil_y'].to(dev).float()
        cclf, cacc = train_ceiling(zc[:20000], y[:20000], zc[20000:], y[20000:], dev)
        with torch.no_grad():
            ceil_logit = torch.stack([cclf(zi[g]) for g in range(nG)]).cpu().numpy()
            ceil_real_acc = float(((cclf(nz(G['ceil_f4'][20000:])) > 0).float() == y[20000:]).float().mean())
        np.savez_compressed(o, grid_r=G['grid_r'].numpy(), use=G['grid_use'].numpy(), w=G['grid_scene']['w'].numpy(),
                            real_logit=real_logit, imag_logit=imag_logit, ceil_logit=ceil_logit)
        summ = dict(name=name, ceiling_val_acc_imagined=cacc, ceiling_acc_on_real_f4=ceil_real_acc, sec=time.time() - t0)
        json.dump(summ, open(os.path.join(OUT, f'e1_{name}.json'), 'w'), indent=1)
        print(f'{name}: ceiling classifier val acc (imagined) {cacc:.4f} ({time.time() - t0:.0f}s)', flush=True)
    print('M3_E1_EVAL_DONE')


if __name__ == '__main__':
    main()
