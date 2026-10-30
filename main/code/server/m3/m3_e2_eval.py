"""B E2 M3: policy evaluation of the 9 E1 controlled models (p in {0.03, 0.1, 0.3} x seed {0, 1, 2}) with the R2-9
policies pi1-pi6 on the R2-9 evaluation bank (1,024 start scenes, generator seed 12345; DESIGN_main v1 section 0.3,
v2 section 5, v2.2 section 7). Per model, policy and scene: the frozen R2-9 classifier's logit on the imagined final
frame (4-step rollout from the real f0, as R2-9 evaluate.py) and on the real f4 (perception), the real alignment ratio
r and success (r < 1). Per-policy imagined rates are checked against the E0 evaluation of the overlapping models.
Writes results/E2/m3/e2_<model>.npz (+ e2_m3_meta.json). usage: python m3_e2_eval.py"""
import os as _os
# B guard: GPU 1 only, torch hub inside ~/pmx (see ../bpaths.py)
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, time, json
import numpy as np
import torch
from wm import CK, OUT, Classifier, Predictor, load_norm
from world import FEAT_DIR, POLICIES
from evaluate import imagine

MODELS = [f'p{p}_std_1x_s{s}' for p in ('0.03', '0.1', '0.3') for s in (0, 1, 2)]
E0_OUT = os.path.expanduser('~/pmx/B/results/E0/m3')


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    E = torch.load(os.path.join(FEAT_DIR, 'eval.pt'), weights_only=False)['E']
    mu, sd = load_norm(dev)
    nz = lambda x: (x.to(dev).float() - mu) / sd
    clf = Classifier().to(dev).eval()
    clf.load_state_dict(torch.load(os.path.join(CK, 'classifier.pt'), map_location=dev))
    for p_ in clf.parameters():
        p_.requires_grad_(False)
    names = list(POLICIES)
    z0 = nz(E['pol_f0'])
    r = np.stack([E['pol'][p]['r'].numpy() for p in names])
    succ = np.stack([E['pol'][p]['succ'].numpy() for p in names])
    with torch.no_grad():
        real_logit = np.stack([clf(nz(E['pol'][p]['f4'])).cpu().numpy() for p in names])
    os.makedirs(OUT, exist_ok=True)
    meta = dict(policies=names, n_scenes=int(r.shape[1]), real_acc=float(((real_logit > 0) == succ).mean()), check={})
    for name in MODELS:
        ck = torch.load(os.path.join(CK, f'{name}.pt'), map_location=dev, weights_only=False)
        model = Predictor().to(dev).eval()
        model.load_state_dict(ck['state'])
        imag_logit = np.stack([imagine(model, clf, z0, E['pol'][p]['act'].to(dev)).cpu().numpy() for p in names])
        np.savez_compressed(os.path.join(OUT, f'e2_{name}.npz'), policies=np.array(names), r=r, succ=succ,
                            real_logit=real_logit, imag_logit=imag_logit)
        rates = {p: float((imag_logit[k] > 0).mean()) for k, p in enumerate(names)}
        e0 = os.path.join(E0_OUT, f'eval_{name}.json')
        if os.path.exists(e0):
            ref = json.load(open(e0))['policies']
            meta['check'][name] = max(abs(rates[p] - ref[p]['imagined']) for p in names)
        print(f'{name}: ' + ' '.join(f'{p[:3]} {rates[p]:.3f}' for p in names) + f' ({time.time() - t0:.0f}s)', flush=True)
    meta['time_s'] = time.time() - t0
    json.dump(meta, open(os.path.join(OUT, 'e2_m3_meta.json'), 'w'), indent=1)
    print('M3_E2_EVAL_DONE', json.dumps(meta['check']), flush=True)


if __name__ == '__main__':
    main()
