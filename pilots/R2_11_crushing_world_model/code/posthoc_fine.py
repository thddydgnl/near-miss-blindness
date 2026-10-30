"""R2-11 POST-HOC (사후) fine-vs-coarse distinctions read from the predicted next frame (written after the
pre-registered analysis, to test "fine distinctions are lost first" with crisp binary distinctions).

Same linear kernel ridge as analyze.py on inputs [z_hat_{t+1}, z_t] (model), [z_t] (copy), [z_{t+1}, z_t] (oracle).
Scores are evaluated with AUROC on held-out episodes:
  fine   touch_vs_near : at t+1, agent touching the box (gap <= 0.5 px) vs near but not touching (0.5 < gap <= 4 px)
  fine   onset         : among frames with the agent NOT touching at t but within 6 px, does it touch at t+1?
  fine   box_moves     : among frames with contact at t+1, does the box move (> 0.5 px) during the step?
  coarse near_vs_far   : at t+1, gap <= 4 px vs gap > 12 px
  coarse agent_moves   : agent displacement > 4 px vs < 2 px during the step
Usage: python posthoc_fine.py --data /workspace/data --ckpt /workspace/ckpt --out /workspace/results
"""
import argparse
import json
import os
import time

import torch

import analyze as A
import env as E
from model import CONFIGS, make_model

DEV = 'cuda'
NPER = 38


def auroc(score, label):
    """Mann-Whitney AUROC of scores for binary labels (ties averaged)."""
    s = score.double()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float64)
    # average ranks for ties
    uniq, inv, cnt = torch.unique(s, return_inverse=True, return_counts=True)
    sums = torch.zeros(uniq.numel(), dtype=torch.float64, device=s.device).index_add_(0, inv, ranks)
    ranks = (sums / cnt)[inv]
    pos = label.bool()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--seeds', default='0,1')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    st = torch.load(f'{args.data}/stats.pt')
    mu, sd = st['mu'].to(DEV), st['sd'].to(DEV)
    d = torch.load(f'{args.data}/ana.pt')
    n_ep = d['Z'].shape[0]
    Z = torch.empty(d['Z'].shape, dtype=torch.float16, device=DEV)
    for i in range(0, n_ep, 50):
        Z[i:i + 50] = ((d['Z'][i:i + 50].to(DEV).float() - mu) / sd).half()
    Aact = d['a'].to(DEV).float()
    ep = torch.arange(n_ep, device=DEV).repeat_interleave(NPER)
    tt = torch.arange(2, 2 + NPER, device=DEV).repeat(n_ep)
    tr, te = ep < 240, ep >= 240
    fit = (ep[tr] < 180).nonzero().flatten()
    val = (ep[tr] >= 180).nonzero().flatten()
    epc, tc = ep.cpu(), tt.cpu()
    g = lambda k, dt=0: d[k][epc, tc + dt].double().to(DEV)
    pa0, pa1, pb0, pb1, th0, th1 = g('pa'), g('pa', 1), g('pb'), g('pb', 1), g('th'), g('th', 1)
    gap0, gap1 = E.box_gap(pa0, pb0, th0), E.box_gap(pa1, pb1, th1)
    dpa = (pa1 - pa0).norm(dim=-1)
    dpb = (pb1 - pb0).norm(dim=-1)
    touch1 = (gap1 <= 0.5).double()
    near1 = (gap1 <= 4.0).double()
    tasks = {
        'touch_vs_near': ('fine', touch1, gap1 <= 4.0),
        'onset': ('fine', touch1, (gap0 > 0.5) & (gap0 <= 6.0)),
        'box_moves': ('fine', (dpb > 0.5).double(), gap1 <= 0.5),
        'near_vs_far': ('coarse', near1, (gap1 <= 4.0) | (gap1 > 12.0)),
        'agent_moves': ('coarse', (dpa > 4.0).double(), (dpa > 4.0) | (dpa < 2.0)),
    }
    Y = torch.stack([tasks[k][1] for k in tasks], 1)
    Ytr = Y[tr]
    sel_te = {k: tasks[k][2][te] for k in tasks}
    out = dict(note='POST-HOC fine vs coarse distinctions (AUROC on held-out episodes)',
               n_test={k: int(sel_te[k].sum()) for k in tasks},
               pos_rate={k: float(Y[te][sel_te[k], i].mean()) for i, k in enumerate(tasks)},
               kind={k: tasks[k][0] for k in tasks})

    def evaluate(pred):
        return {k: auroc(pred[sel_te[k], i], Y[te][sel_te[k], i]) for i, k in enumerate(tasks)}
    zt = Z[ep, tt].reshape(ep.numel(), -1)
    z1 = Z[ep, tt + 1].reshape(ep.numel(), -1)
    zt_c = zt - zt[tr].float().mean(0).half()
    z1_c = z1 - z1[tr].float().mean(0).half()
    Kt_tr, Kt_te = A.gram(zt_c[tr], zt_c[tr]), A.gram(zt_c[te], zt_c[tr])
    K1_tr, K1_te = A.gram(z1_c[tr], z1_c[tr]), A.gram(z1_c[te], z1_c[tr])
    del zt_c, z1_c
    p, _ = A.kridge(Kt_tr, Kt_te, Ytr, fit, val)
    out['copy'] = evaluate(p)
    p, _ = A.kridge(Kt_tr + K1_tr, Kt_te + K1_te, Ytr, fit, val)
    out['oracle'] = evaluate(p)
    del K1_tr, K1_te
    print('n_test', out['n_test'], 'pos', {k: round(v, 3) for k, v in out['pos_rate'].items()}, flush=True)
    print('copy', {k: round(v, 3) for k, v in out['copy'].items()}, flush=True)
    print('oracle', {k: round(v, 3) for k, v in out['oracle'].items()}, f'{time.time() - t0:.0f}s', flush=True)
    out['models'] = {}
    for s in [int(x) for x in args.seeds.split(',')]:
        for name in CONFIGS:
            fn = f'{args.ckpt}/{name}_s{s}_final.pt'
            if not os.path.exists(fn):
                continue
            model = make_model(name).to(DEV)
            model.load_state_dict(torch.load(fn, map_location=DEV))
            model.eval()
            zh = torch.empty_like(zt)
            with torch.no_grad():
                for i in range(0, ep.numel(), 500):
                    e, t_ = ep[i:i + 500], tt[i:i + 500]
                    fr = t_[:, None] + torch.arange(-2, 1, device=DEV)[None]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pz = model(Z[e[:, None], fr].float(), Aact[e, t_])
                    zh[i:i + 500] = pz.float().reshape(pz.shape[0], -1).half()
            zh_c = zh - zh[tr].float().mean(0).half()
            Kh_tr, Kh_te = A.gram(zh_c[tr], zh_c[tr]), A.gram(zh_c[te], zh_c[tr])
            del zh, zh_c
            p, _ = A.kridge(Kh_tr + Kt_tr, Kh_te + Kt_te, Ytr, fit, val)
            del Kh_tr, Kh_te
            out['models'][f'{name}_s{s}'] = evaluate(p)
            print(name, s, {k: round(v, 3) for k, v in out['models'][f'{name}_s{s}'].items()},
                  f'{time.time() - t0:.0f}s', flush=True)
            json.dump(out, open(f'{args.out}/analysis/posthoc_fine.json', 'w'), indent=1)
            del model
            torch.cuda.empty_cache()
    print('done', flush=True)


if __name__ == '__main__':
    main()
