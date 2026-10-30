"""R2-6 analyses per model: near/far optimism, boundary r*, failure-swap concentration n50 (and gain-free n50_dact,
untrained null), last-layer share, regime of the top failure units, detector ablation (top-n50 units, mean-ablation)
vs layer-matched random units, gain-free overlap with success-lift units (+ untrained null).
--timeseries: the same core metrics on parameter snapshots of the std models."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from common import (DEPTH, EDGES_BAL, EDGES_NEAR, TAU_WEAK, WIDTH, R_obj, binned_lift_prob, boundary, forward,
                    forward_from, nx, outcome_metrics, raw_y, slice_model, test_sets, to_py, train_data)

TOPK_FRAC = 0.10
CAT_NAMES = ['neg', 'weak+', 'strong+']
N_RAND = 20
_BANKS = {}


def train_inputs(cfg, n, dev):
    d = train_data(cfg['p'], cfg['seed'], cfg['N'], dev)
    if cfg['version'] == 'state':
        return d['X'][:n]
    from visual import FEAT_DIR
    key = (cfg['seed'], cfg['N'])
    if key not in _BANKS:
        b = torch.load(os.path.join(FEAT_DIR, f'bank_s{cfg["seed"]}_N{cfg["N"]}.pt'))
        _BANKS[key] = (b['feat'], b['slip_rank'])
    feat, rank = _BANKS[key]
    X = d['X'][:n]
    n = X.shape[0]
    col5 = (X[:, 11] > 0.5).cpu()
    ii = torch.where(col5, cfg['N'] + rank[:n].clamp_min(0), torch.arange(n))
    return torch.cat([feat[ii].float().to(dev), X[:, 4:6]], 1)


def act_stats(Pm, X):
    _, acts = forward(Pm, X, return_acts=True)
    mj = torch.stack([torch.quantile(a.abs()[:20000], 0.999, dim=0) for a, z in acts]).clamp_min(1e-6)
    mean = torch.stack([a.mean(0) for a, z in acts])
    return mj, mean


def patch_importance(Pm, nm, pa, pb, readout, keep=False):
    Xa, Xb = nx(pa['X'], nm), nx(pb['X'], nm)
    ya, acts_a = forward(Pm, Xa, return_acts=True)
    yb, acts_b = forward(Pm, Xb, return_acts=True)
    Ra, Rb = readout(raw_y(ya, nm)), readout(raw_y(yb, nm))
    D = (Rb - Ra).abs().mean()
    ar = torch.arange(WIDTH, device=Xa.device)
    I_ab, I_ba, eff_ba = [], [], []
    for l in range(DEPTH):
        A, B = acts_a[l][0], acts_b[l][0]
        for src, dst, Rd, store, is_ba in ((B, A, Ra, I_ab, False), (A, B, Rb, I_ba, True)):
            Ap = dst.unsqueeze(0).repeat(WIDTH, 1, 1)
            Ap[ar, :, ar] = src[:, ar].T
            eff = readout(raw_y(forward_from(Pm, l, Ap), nm)) - Rd[None]
            store.append(eff.abs().mean(-1) / D)
            if is_ba and keep:
                eff_ba.append(eff)
    out = dict(I=0.5 * (torch.cat(I_ab) + torch.cat(I_ba)), D=float(D), acts_a=acts_a, acts_b=acts_b)
    if keep:
        out['eff_ba'] = torch.cat(eff_ba)
    return out


def delta_act(pi, mj):
    return torch.cat([(pi['acts_b'][l][0] - pi['acts_a'][l][0]).abs().mean(0) / mj[l] for l in range(DEPTH)])


def concentration(I):
    s = torch.sort(I, descending=True).values
    c = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
    return int((c < 0.5).sum().item() + 1)


def layer_share(I):
    return [float(I[l * WIDTH:(l + 1) * WIDTH].sum() / I.sum()) for l in range(DEPTH)]


def topk_set(I, k):
    return set(torch.topk(I, k).indices.tolist())


def exp_jacc_layer(ca, cb, rng, draws=4000):
    inter = np.zeros(draws)
    for l in range(DEPTH):
        if ca[l] and cb[l]:
            inter += rng.hypergeometric(ca[l], WIDTH - ca[l], cb[l], size=draws)
    return float(np.mean(inter / (sum(ca) + sum(cb) - inter)))


def overlap(Ia, Ib, k, rng):
    A, B = topk_set(Ia, k), topk_set(Ib, k)
    lc = lambda s: [sum(1 for i in s if i // WIDTH == l) for l in range(DEPTH)]
    J = len(A & B) / len(A | B)
    E = exp_jacc_layer(lc(A), lc(B), rng)
    return dict(J=J, E=E, ratio=J / max(E, 1e-9))


def regime(pi, mj, sel):
    def cats(acts):
        a = torch.cat([acts[l][0] for l in range(DEPTH)], 1)
        z = torch.cat([acts[l][1] for l in range(DEPTH)], 1)
        c = torch.ones_like(a, dtype=torch.long)
        c[z < 0] = 0
        c[(z >= 0) & (a >= TAU_WEAK * mj.reshape(-1)[None])] = 2
        return c.T
    trans = 3 * cats(pi['acts_a']) + cats(pi['acts_b'])
    idx = torch.tensor(sorted(sel), device=trans.device)
    w = pi['eff_ba'].abs()[idx]
    tt = trans[idx]
    mass = torch.stack([(w * (tt == t)).sum() for t in range(9)])
    mass = mass / mass.sum().clamp_min(1e-12)
    return {f'{CAT_NAMES[t // 3]}->{CAT_NAMES[t % 3]}': float(mass[t]) for t in range(9)}


def ablate_eval(Pm, X, d, nm, sets, value):
    def hk(l, a, z):
        ii = sets[l]
        if len(ii) == 0:
            return a
        a = a.clone()
        a[:, ii] = value[l][ii]
        return a
    m = outcome_metrics(raw_y(forward(Pm, X, hk), nm), d)
    return {k: float(v) for k, v in m.items()}


def detector_ablation(Pm, nm, TS, I_fail, n_det, mean_act, gen):
    order = torch.argsort(I_fail, descending=True)[:n_det]
    det = [order[order // WIDTH == l] % WIDTH for l in range(DEPTH)]
    X = nx(TS['bal']['X'], nm)
    zero = torch.zeros_like(mean_act)
    res = dict(n_det=n_det, det_layers=[len(x) for x in det])
    res['det_mean'] = ablate_eval(Pm, X, TS['bal'], nm, det, mean_act)
    res['det_zero'] = ablate_eval(Pm, X, TS['bal'], nm, det, zero)
    rs_m, rs_z = [], []
    for _ in range(N_RAND):
        rset = []
        for l in range(DEPTH):
            pool = torch.ones(WIDTH, dtype=torch.bool, device=X.device)
            pool[det[l]] = False
            cand = pool.nonzero().squeeze(1)
            rset.append(cand[torch.randperm(len(cand), generator=gen, device=X.device)[:len(det[l])]])
        rs_m.append(ablate_eval(Pm, X, TS['bal'], nm, rset, mean_act))
        rs_z.append(ablate_eval(Pm, X, TS['bal'], nm, rset, zero))
    res['rand_mean'] = {k: float(np.mean([r[k] for r in rs_m])) for k in rs_m[0]}
    res['rand_zero'] = {k: float(np.mean([r[k] for r in rs_z])) for k in rs_z[0]}
    return res


def core(Pm, nm, TS, mj):
    """metrics used by both the full analysis and the time series."""
    yb = raw_y(forward(Pm, nx(TS['bal']['X'], nm)), nm)
    yn = raw_y(forward(Pm, nx(TS['near']['X'], nm)), nm)
    m = {k: float(v) for k, v in outcome_metrics(yb, TS['bal']).items()}
    cb, cn = binned_lift_prob(yb, TS['bal'], EDGES_BAL), binned_lift_prob(yn, TS['near'], EDGES_NEAR)
    m['r_star'] = boundary(to_py(cb), to_py(cn), EDGES_BAL, EDGES_NEAR)
    pm = patch_importance(Pm, nm, *TS['pairs']['miss'], R_obj, keep=True)
    m['n50'] = concentration(pm['I'])
    m['n50_dact'] = concentration(delta_act(pm, mj))
    m['layer_share'] = layer_share(pm['I'])
    m['miss_D'] = pm['D']
    return m, pm


def analyze_model(Pm, P0m, cfg, nm, TS, dev, gen, rng):
    t0 = time.time()
    Xtr = nx(train_inputs(cfg, 20000 if cfg['version'] == 'state' else 10000, dev), nm)
    mj, mean_act = act_stats(Pm, Xtr)
    res, pm = core(Pm, nm, TS, mj)
    res['cfg'] = cfg
    k = int(round(TOPK_FRAC * DEPTH * WIDTH))
    res['regime_top'] = regime(pm, mj, topk_set(pm['I'], k))
    res['ablation'] = detector_ablation(Pm, nm, TS, pm['I'], res['n50'], mean_act, gen)
    ps = patch_importance(Pm, nm, *TS['pairs']['succ'], R_obj)
    res['succ_n50'] = concentration(ps['I'])
    res['overlap_dact'] = overlap(delta_act(pm, mj), delta_act(ps, mj), k, rng)
    res['overlap_gain'] = overlap(pm['I'], ps['I'], k, rng)
    mj0, _ = act_stats(P0m, Xtr)
    pm0 = patch_importance(P0m, nm, *TS['pairs']['miss'], R_obj)
    ps0 = patch_importance(P0m, nm, *TS['pairs']['succ'], R_obj)
    res['untrained'] = dict(n50=concentration(pm0['I']), n50_dact=concentration(delta_act(pm0, mj0)),
                            layer_share=layer_share(pm0['I']), overlap_dact=overlap(delta_act(pm0, mj0), delta_act(ps0, mj0), k, rng),
                            overlap_gain=overlap(pm0['I'], ps0['I'], k, rng))
    res['importance_fail'] = to_py(pm['I'])
    res['seconds'] = time.time() - t0
    return res


def load_tests(version, dev):
    if version == 'state':
        return test_sets(dev)
    from visual import FEAT_DIR
    VT = torch.load(os.path.join(FEAT_DIR, 'test_visual.pt'))
    mv = lambda d: {k: v.to(dev).float() if v.is_floating_point() else v.to(dev) for k, v in d.items()}
    return dict(bal=mv(VT['bal']), near=mv(VT['near']), pairs={k: (mv(a), mv(b)) for k, (a, b) in VT['pairs'].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--modes', default='std,rw,hn')
    ap.add_argument('--timeseries', action='store_true')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(os.path.join(args.ckpt, f'{args.name}.pt'), map_location=dev)
    nm = {k: v.to(dev) for k, v in ck['norm'].items()}
    cfgs = ck['cfgs']
    TS = load_tests(cfgs[0]['version'], dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(777)
    rng = np.random.default_rng(0)
    od = os.path.join(args.out, args.name)
    os.makedirs(od, exist_ok=True)
    modes = args.modes.split(',')
    sel = [m for m, c in enumerate(cfgs) if c['mode'] in modes]
    with torch.no_grad():
        if args.timeseries:
            snaps = torch.load(os.path.join(args.ckpt, f'{args.name}_snaps.pt'))
            out = []
            for m in sel:
                c = cfgs[m]
                Xtr = None
                for sn in snaps:
                    Pm = {k: v[m].to(dev) for k, v in sn['P'].items()}
                    if Xtr is None:
                        Xtr = nx(train_inputs(c, 10000, dev), nm)
                    mj, _ = act_stats(Pm, Xtr)
                    r, _ = core(Pm, nm, TS, mj)
                    r.update(step=sn['step'], model=m, p=c['p'], mode=c['mode'], seed=c['seed'], N=c['N'])
                    out.append(r)
                print(f'timeseries model {m} {c} done', flush=True)
            json.dump(to_py(out), open(os.path.join(od, 'timeseries.json'), 'w'))
        else:
            P = {k: v.to(dev) for k, v in ck['P'].items()}
            P0 = {k: v.to(dev) for k, v in ck['P0'].items()}
            out = []
            for m in sel:
                r = analyze_model(slice_model(P, m), slice_model(P0, m), cfgs[m], nm, TS, dev, gen, rng)
                r['model'] = m
                out.append(r)
                ab = r['ablation']
                print(f"model {m} {cfgs[m]['p']} {cfgs[m]['mode']} s{cfgs[m]['seed']} opt={r['opt']:.3f} near={r.get('opt_near', float('nan')):.3f} "
                      f"far={r.get('opt_far', float('nan')):.3f} r*={r['r_star']:.3f} n50={r['n50']} (untr {r['untrained']['n50']}) "
                      f"L3={r['layer_share'][3]:.2f} abl det={ab['det_mean']['opt']:.3f} rand={ab['rand_mean']['opt']:.3f} ({r['seconds']:.1f}s)", flush=True)
            json.dump(to_py(out), open(os.path.join(od, 'analysis.json'), 'w'))
    print('analysis done', flush=True)


if __name__ == '__main__':
    main()
