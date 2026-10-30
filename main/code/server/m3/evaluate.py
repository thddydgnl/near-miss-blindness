"""R2-9 evaluation of trained world models: policy evaluation (imagined vs true success), boundary r*,
success-case latent prediction error, and n50 of the failure-swap effect on the predictor's MLP units
(paired counterfactual: same start frame, aligned vs misaligned action; unit patched at every rollout step;
readout = frozen classifier logit on the imagined final frame), with the untrained-predictor null."""
import os as _os
# B guard (added 14:43Z): GPU 1 only, torch hub inside ~/pmx (see ../bpaths.py)
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import json
import os
import sys
import time

import numpy as np
import torch

from wm import CK, OUT, Classifier, Predictor, load_norm
from world import FEAT_DIR, NEAR, POLICIES



def concentration(I):
    s = torch.sort(I, descending=True).values
    c = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
    return int((c < 0.5).sum().item() + 1)


def boundary(r, p):
    pts = list(zip(r, p))
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 >= 0.5 > y1:
            return x0 + (y0 - 0.5) / (y0 - y1) * (x1 - x0)
    return float('nan') if pts[0][1] < 0.5 else pts[-1][0]


@torch.no_grad()
def imagine(model, clf, z0, acts, bs=4096):
    out = []
    for i in range(0, z0.shape[0], bs):
        out.append(clf(model.rollout(z0[i:i + bs], acts[i:i + bs])[-1]))
    return torch.cat(out)


@torch.no_grad()
def n50_patch(model, clf, z0, acts_s, acts_f, chunk=64):
    torch.backends.cuda.matmul.allow_tf32 = True     # speed: TF32 only inside the patching analysis
    try:
        return _n50_patch(model, clf, z0, acts_s, acts_f, chunk)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def _n50_patch(model, clf, z0, acts_s, acts_f, chunk):
    P = z0.shape[0]

    def capture(acts):
        store = {}

        def hook_fn(t):
            def hk(l, u):
                store[(t, l)] = u
                return u
            return hk
        R = clf(model.rollout(z0, acts, hook_fn)[-1])
        return store, R
    Hs, Rs = capture(acts_s)
    Hf, Rf = capture(acts_f)
    D = (Rs - Rf).abs().mean()
    imp = torch.zeros(model.L, model.mlp, device=z0.device)
    for l in range(model.L):
        for j0 in range(0, model.mlp, chunk):
            J = torch.arange(j0, min(j0 + chunk, model.mlp), device=z0.device)
            C = len(J)
            for src, acts_dst, R_dst in ((Hs, acts_f, Rf), (Hf, acts_s, Rs)):
                def hook_fn(t, src=src):
                    def hk(ll, u):
                        if ll != l:
                            return u
                        uu = u.clone().view(C, P, u.shape[1], u.shape[2])
                        uu[torch.arange(C, device=u.device), :, :, J] = src[(t, l)][:, :, J].permute(2, 0, 1)
                        return uu.view(C * P, u.shape[1], u.shape[2])
                    return hk
                R = clf(model.rollout(z0.repeat(C, 1, 1), acts_dst.repeat(C, 1, 1), hook_fn)[-1]).view(C, P)
                imp[l, J] += 0.5 * (R - R_dst[None]).abs().mean(1) / D
    I = imp.flatten()
    return dict(n50=concentration(I), layer_share=[float(imp[l].sum() / imp.sum()) for l in range(model.L)],
                D=float(D), I=I.tolist())


def evaluate(name, dev, clf, E, mu, sd):
    t0 = time.time()
    ck = torch.load(os.path.join(CK, f'{name}.pt'), map_location=dev)
    model = Predictor().to(dev).eval()
    model.load_state_dict(ck['state'])
    nz = lambda x: (x.to(dev).float() - mu) / sd
    res = dict(name=name, cfg=ck['cfg'], train_log=ck['log'])
    # policy evaluation
    z0 = nz(E['pol_f0'])
    pol = {}
    for pname, P in E['pol'].items():
        acts = P['act'].to(dev)
        logit = imagine(model, clf, z0, acts)
        imag = (logit > 0).float()
        real = (clf(nz(P['f4'])) > 0).float()
        r, succ = P['r'].to(dev), P['succ'].to(dev).float()
        near = (r >= NEAR[0]) & (r < NEAR[1])
        far = r >= 1.5
        pol[pname] = dict(true=float(succ.mean()), imagined=float(imag.mean()), real_frame_clf=float(real.mean()),
                          over=float(imag.mean() - succ.mean()), near_share=float(near.float().mean()),
                          fail_share=float(1 - succ.mean()),
                          opt_near=float(imag[near].mean()) if near.any() else float('nan'),
                          opt_far=float(imag[far].mean()) if far.any() else float('nan'),
                          pess=float(1 - imag[succ > 0].mean()) if (succ > 0).any() else float('nan'))
    res['policies'] = pol
    res['mean_over'] = float(np.mean([v['over'] for v in pol.values()]))
    res['mean_abs_err'] = float(np.mean([abs(v['over']) for v in pol.values()]))
    # boundary r*
    G = E['grid_act'].shape[0]
    zg = nz(E['grid_f0'])
    with torch.no_grad():
        pg = [float((imagine(model, clf, zg, E['grid_act'][g].to(dev)) > 0).float().mean()) for g in range(G)]
    rg = E['grid_r'].tolist()
    res['grid'] = dict(r=rg, p_succ=pg)
    res['r_star'] = boundary(rg, pg)
    # latent prediction error on real success / failure episodes (autoregressive, normalised token MSE)
    with torch.no_grad():
        for key in ('succ', 'fail'):
            f = nz(E[f'err_{key}']['f'])
            zs = model.rollout(f[:, 0], E[f'err_{key}']['act'].to(dev))
            res[f'err_{key}'] = float(torch.stack([((zs[t] - f[:, t + 1]) ** 2).mean() for t in range(4)]).mean())
            res[f'err_{key}_final'] = float(((zs[-1] - f[:, 4]) ** 2).mean())
            pred_succ = (clf(zs[-1]) > 0).float()
            res[f'imag_acc_{key}'] = float(pred_succ.mean() if key == 'succ' else 1 - pred_succ.mean())
    # n50 on the failure swap, trained and untrained
    zp = nz(E['pair_f0'])
    a_s, a_f = E['pair_act_s'].to(dev), E['pair_act_f'].to(dev)
    res['swap'] = n50_patch(model, clf, zp, a_s, a_f)
    upath = os.path.join(OUT, f"untrained_s{ck['cfg']['seed']}.json")   # same init for every model of a seed
    if os.path.exists(upath):
        res['swap_untrained'] = json.load(open(upath))
    else:
        m0 = Predictor().to(dev).eval()
        m0.load_state_dict(ck['init'])
        res['swap_untrained'] = n50_patch(m0, clf, zp, a_s, a_f)
        os.makedirs(OUT, exist_ok=True)
        json.dump(res['swap_untrained'], open(upath, 'w'))
    res['n50'], res['n50_untrained'] = res['swap']['n50'], res['swap_untrained']['n50']
    res['seconds'] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f'eval_{name}.json'), 'w'))
    print(f"{name}: mean_over={res['mean_over']:.3f} " + ' '.join(f"{k[:3]}={v['over']:+.3f}" for k, v in pol.items())
          + f" r*={res['r_star']:.3f} n50={res['n50']} (untr {res['n50_untrained']}) err_s={res['err_succ']:.4f} ({res['seconds']:.0f}s)", flush=True)


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    names = sys.argv[1].split(',')
    E = torch.load(os.path.join(FEAT_DIR, 'eval.pt'))['E']
    mu, sd = load_norm(dev)
    clf = Classifier().to(dev).eval()
    clf.load_state_dict(torch.load(os.path.join(CK, 'classifier.pt'), map_location=dev))
    for p in clf.parameters():
        p.requires_grad_(False)
    for n in names:
        if os.path.exists(os.path.join(CK, f'{n}.pt')):
            evaluate(n, dev, clf, E, mu, sd)
        else:
            print('missing', n, flush=True)


if __name__ == '__main__':
    main()
