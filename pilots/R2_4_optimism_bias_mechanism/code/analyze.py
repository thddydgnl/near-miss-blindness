"""Mechanism analyses for every model in a checkpoint (S2: weak-activation ablation, S3: neuron overlap, regimes).
Outputs <out>/<name>/analysis.json (per model) and analysis_rows.csv (flat)."""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from common import (DEPTH, TAU_WEAK, WIDTH, R_gz, R_obj, forward, forward_from, nx, outcome_metrics, raw_y,
                    slice_model, test_sets, to_py, train_data)

ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
TOPK_FRAC = 0.10
CAT_NAMES = ['neg', 'weak+', 'strong+']   # GELU negative pole (z<0), weak positive, strong positive (|a|>=TAU*m_j)


def neuron_scale(Pm, cfg, nm, dev, n=20000):
    d = train_data(cfg['p'], cfg['seed'], cfg['N'], dev)
    _, acts = forward(Pm, nx(d['X'][:n], nm), return_acts=True)
    return torch.stack([torch.quantile(a.abs(), 0.999, dim=0) for a, z in acts]).clamp_min(1e-6)  # [DEPTH, WIDTH]


def masked_eval(Pm, X, d, nm, mask_fn, gen):
    """apply mask_fn at every hidden layer (sequentially) and a count-matched random mask as control."""
    counts = []
    st = dict(mass_removed=0.0, mass_total=0.0, n_removed=0, n_total=0)

    def hk(l, a, z):
        mask = mask_fn(l, a, z)
        counts.append(mask.sum(-1))
        st['mass_removed'] += float((a.abs() * mask).sum())
        st['mass_total'] += float(a.abs().sum())
        st['n_removed'] += int(mask.sum())
        st['n_total'] += mask.numel()
        return a.masked_fill(mask, 0.0)

    mw = outcome_metrics(raw_y(forward(Pm, X, hk), nm), d)

    def hr(l, a, z):
        rnk = torch.rand(a.shape, generator=gen, device=a.device).argsort(-1).argsort(-1)
        return a.masked_fill(rnk < counts[l][:, None], 0.0)

    mr = outcome_metrics(raw_y(forward(Pm, X, hr), nm), d)
    return dict(target={k: float(v) for k, v in mw.items()}, random={k: float(v) for k, v in mr.items()},
                frac_entries=st['n_removed'] / max(st['n_total'], 1),
                frac_mass=st['mass_removed'] / max(st['mass_total'], 1e-12))


def patch_importance(Pm, nm, pa, pb, readout, denom=None, basis=None, keep=False):
    """paired counterfactual swap per neuron (or per rotated direction if basis is given).
    I_ab: patch b's value into a's run; I_ba: patch a's value into b's run. Normalised by mean |R_b - R_a|."""
    Xa, Xb = nx(pa['X'], nm), nx(pb['X'], nm)
    ya, acts_a = forward(Pm, Xa, return_acts=True)
    yb, acts_b = forward(Pm, Xb, return_acts=True)
    Ra, Rb = readout(raw_y(ya, nm)), readout(raw_y(yb, nm))
    D = (Rb - Ra).abs().mean() if denom is None else denom
    ar = torch.arange(WIDTH, device=Xa.device)
    I_ab, I_ba, eff_ba = [], [], []
    for l in range(DEPTH):
        A, B = acts_a[l][0], acts_b[l][0]
        for src, dst, Rd, store, is_ba in ((B, A, Ra, I_ab, False), (A, B, Rb, I_ba, True)):
            if basis is None:
                Ap = dst.unsqueeze(0).repeat(WIDTH, 1, 1)
                Ap[ar, :, ar] = src[:, ar].T
            else:
                Q = basis[l]
                Dc = ((src - dst) @ Q).T                      # [WIDTH, n] coefficient differences
                Ap = dst.unsqueeze(0) + Dc.unsqueeze(-1) * Q.T.unsqueeze(1)
            eff = readout(raw_y(forward_from(Pm, l, Ap), nm)) - Rd[None]   # [WIDTH, n]
            store.append(eff.abs().mean(-1) / D)
            if is_ba and keep:
                eff_ba.append(eff)
    out = dict(I=0.5 * (torch.cat(I_ab) + torch.cat(I_ba)), I_ab=torch.cat(I_ab), I_ba=torch.cat(I_ba),
               D=float(D), dR_mean=float((Rb - Ra).mean()))
    if keep:
        out['eff_ba'] = torch.cat(eff_ba)
        out['acts_a'], out['acts_b'] = acts_a, acts_b
    return out


def delta_act(pi, mj):
    return torch.cat([(pi['acts_b'][l][0] - pi['acts_a'][l][0]).abs().mean(0) / mj[l] for l in range(DEPTH)])


def zero_importance(Pm, nm, d, readout):
    X = nx(d['X'], nm)
    y, acts = forward(Pm, X, return_acts=True)
    R0 = readout(raw_y(y, nm))
    D = R0.abs().mean()
    ar = torch.arange(WIDTH, device=X.device)
    I = []
    for l in range(DEPTH):
        Ap = acts[l][0].unsqueeze(0).repeat(WIDTH, 1, 1)
        Ap[ar, :, ar] = 0.0
        eff = readout(raw_y(forward_from(Pm, l, Ap), nm)) - R0[None]
        I.append(eff.abs().mean(-1) / D)
    return torch.cat(I)


def topk_set(I, k):
    return set(torch.topk(I, k).indices.tolist())


def jacc(a, b):
    return len(a & b) / max(len(a | b), 1)


def layer_counts(s):
    return [sum(1 for i in s if i // WIDTH == l) for l in range(DEPTH)]


def exp_jacc_uniform(n, k):
    tot = math.comb(n, k)
    e = 0.0
    for i in range(0, k + 1):
        pr = math.comb(k, i) * math.comb(n - k, k - i) / tot
        e += pr * i / (2 * k - i)
    return e


def exp_jacc_layer(ca, cb, rng, draws=4000):
    inter = np.zeros(draws)
    for l in range(DEPTH):
        if ca[l] == 0 or cb[l] == 0:
            continue
        inter += rng.hypergeometric(ca[l], WIDTH - ca[l], cb[l], size=draws)
    ka, kb = sum(ca), sum(cb)
    return float(np.mean(inter / (ka + kb - inter)))


def overlap_stats(Ia, Ib, k, rng):
    A, B = topk_set(Ia, k), topk_set(Ib, k)
    j = jacc(A, B)
    ej = exp_jacc_layer(layer_counts(A), layer_counts(B), rng)
    return dict(J=j, E_layer=ej, ratio_layer=j / max(ej, 1e-9), E_unif=exp_jacc_uniform(DEPTH * WIDTH, k),
                ratio_unif=j / exp_jacc_uniform(DEPTH * WIDTH, k), share_A_in_B=len(A & B) / k,
                layers_A=layer_counts(A), layers_B=layer_counts(B))


def categories(acts, mj):
    a = torch.cat([acts[l][0] for l in range(DEPTH)], 1)
    z = torch.cat([acts[l][1] for l in range(DEPTH)], 1)
    m = mj.reshape(-1)[None]
    c = torch.ones_like(a, dtype=torch.long)
    c[z < 0] = 0
    c[(z >= 0) & (a >= TAU_WEAK * m)] = 2
    return c.T, (a / m).T   # [1024, n]


def regime_analysis(pi, mj, sel):
    """contribution-weighted distribution of (category in success run -> category in failure run)."""
    cs, us = categories(pi['acts_a'], mj)
    cf, uf = categories(pi['acts_b'], mj)
    w = pi['eff_ba'].abs()
    trans = 3 * cs + cf
    idx = torch.tensor(sorted(sel), device=w.device)
    res = {}
    for name, ii in (('top', idx), ('all', torch.arange(w.shape[0], device=w.device))):
        ww, tt = w[ii], trans[ii]
        mass = torch.stack([(ww * (tt == t)).sum() for t in range(9)])
        mass = mass / mass.sum().clamp_min(1e-12)
        res[name] = {f'{CAT_NAMES[t // 3]}->{CAT_NAMES[t % 3]}': float(mass[t]) for t in range(9)}
    # share of failure information carried by the success run being weak/neg vs strong, and failure run
    res['top_mean_u_succ'] = float(us[idx].mean())
    res['top_mean_u_fail'] = float(uf[idx].mean())
    res['top_frac_pairs_regime_change'] = float((cs[idx] != cf[idx]).float().mean())
    return res


def concentration(I):
    s = torch.sort(I, descending=True).values
    c = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
    return int((c < 0.5).sum().item() + 1)


def analyze_model(Pm, P0m, cfg, nm, TS, dev, gen, rng, basis):
    t0 = time.time()
    res = dict(cfg=cfg)
    Xb = nx(TS['bal']['X'], nm)
    res['bal'] = {k: float(v) for k, v in outcome_metrics(raw_y(forward(Pm, Xb), nm), TS['bal']).items()}
    mj = neuron_scale(Pm, cfg, nm, dev)
    res['mj_median_per_layer'] = to_py(mj.median(-1).values)
    # ---- S2: weak-activation ablation curve with count-matched random control
    curve = []
    for alpha in ALPHAS:
        r = masked_eval(Pm, Xb, TS['bal'], nm, lambda l, a, z, al=alpha: a.abs() < al * mj[l], gen)
        r['alpha'] = alpha
        curve.append(r)
    res['weak_curve'] = curve
    regimes = {
        'neg_pole': lambda l, a, z: z < 0,
        'weak_pos': lambda l, a, z: (z >= 0) & (a < TAU_WEAK * mj[l]),
        'strong_pos': lambda l, a, z: a >= TAU_WEAK * mj[l],
    }
    res['regime_ablation'] = {k: masked_eval(Pm, Xb, TS['bal'], nm, f, gen) for k, f in regimes.items()}
    # ---- S3: paired counterfactual swaps
    PR = TS['pairs']
    k = int(round(TOPK_FRAC * DEPTH * WIDTH))
    pm = patch_importance(Pm, nm, *PR['miss'], R_obj, keep=True)
    pcopy = patch_importance(Pm, nm, *PR['copy'], R_obj, denom=pm['D'])
    ps = patch_importance(Pm, nm, *PR['succ'], R_obj, keep=True)
    psl = patch_importance(Pm, nm, *PR['slip'], R_obj, keep=True)
    pu = patch_importance(Pm, nm, *PR['unrel'], R_gz)
    I_zero = zero_importance(Pm, nm, PR['miss'][0], R_obj)
    I_fail, I_succ = pm['I'], ps['I']
    I_spec = (I_fail - pcopy['I']).clamp_min(0)
    ov = {}
    ov['fail_succ'] = overlap_stats(I_fail, I_succ, k, rng)
    ov['fail_succzero'] = overlap_stats(I_fail, I_zero, k, rng)
    ov['fail_unrel'] = overlap_stats(I_fail, pu['I'], k, rng)
    ov['succ_unrel'] = overlap_stats(I_succ, pu['I'], k, rng)
    ov['failspec_succ'] = overlap_stats(I_spec, I_succ, k, rng)
    ov['fail_copy'] = overlap_stats(I_fail, pcopy['I'], k, rng)
    ov['slip_succ'] = overlap_stats(psl['I'], I_succ, k, rng)
    ov['slip_fail'] = overlap_stats(psl['I'], I_fail, k, rng)
    ov['slip_unrel'] = overlap_stats(psl['I'], pu['I'], k, rng)
    # nulls: random-rotation basis (trained model) and untrained model (neuron basis)
    pmr = patch_importance(Pm, nm, *PR['miss'], R_obj, basis=basis)
    psr = patch_importance(Pm, nm, *PR['succ'], R_obj, basis=basis)
    ov['rot_fail_succ'] = overlap_stats(pmr['I'], psr['I'], k, rng)
    pm0 = patch_importance(P0m, nm, *PR['miss'], R_obj, keep=True)
    ps0 = patch_importance(P0m, nm, *PR['succ'], R_obj, keep=True)
    ov['untrained_fail_succ'] = overlap_stats(pm0['I'], ps0['I'], k, rng)
    # supplementary (added after smoke test, before main run): gain-free representational overlap,
    # top-k by mean |a_b - a_a| / m_j, which removes the readout-gain factor shared by both swap contrasts
    mj0 = neuron_scale(P0m, cfg, nm, dev)
    ov['dact_fail_succ'] = overlap_stats(delta_act(pm, mj), delta_act(ps, mj), k, rng)
    ov['dact_slip_fail'] = overlap_stats(delta_act(psl, mj), delta_act(pm, mj), k, rng)
    ov['untrained_dact_fail_succ'] = overlap_stats(delta_act(pm0, mj0), delta_act(ps0, mj0), k, rng)
    res['overlap'] = ov
    res['contrast'] = dict(miss_D=pm['D'], miss_dR=pm['dR_mean'], copy_D=pcopy['D'], succ_D=ps['D'], slip_D=psl['D'],
                           unrel_D=pu['D'], copy_over_miss=float(pcopy['I'].sum() / I_fail.sum().clamp_min(1e-12)))
    res['concentration'] = dict(fail_n50=concentration(I_fail), succ_n50=concentration(I_succ),
                                slip_n50=concentration(psl['I']), unrel_n50=concentration(pu['I']),
                                fail_rot_n50=concentration(pmr['I']))
    res['layer_share'] = {nme: to_py(torch.stack([I[l * WIDTH:(l + 1) * WIDTH].sum() for l in range(DEPTH)]) / I.sum())
                          for nme, I in (('fail', I_fail), ('succ', I_succ), ('slip', psl['I']), ('unrel', pu['I']))}
    res['regime_fail'] = regime_analysis(pm, mj, topk_set(I_fail, k))
    res['regime_slip'] = regime_analysis(psl, mj, topk_set(psl['I'], k))
    shared = topk_set(I_fail, k) & topk_set(I_succ, k)
    if shared:
        cs, _ = categories(pm['acts_a'], mj)
        cf, _ = categories(pm['acts_b'], mj)
        ii = torch.tensor(sorted(shared), device=dev)
        res['shared_regime_change'] = float((cs[ii] != cf[ii]).float().mean())
        res['shared_n'] = len(shared)
    res['importance'] = dict(fail=to_py(I_fail), succ=to_py(I_succ), slip=to_py(psl['I']), unrel=to_py(pu['I']),
                             copy=to_py(pcopy['I']), succ_zero=to_py(I_zero))
    res['seconds'] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='main')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--models', default='all')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(os.path.join(args.ckpt, f'{args.name}.pt'), map_location=dev)
    nm = {k: v.to(dev) for k, v in ck['norm'].items()}
    P = {k: v.to(dev) for k, v in ck['P'].items()}
    P0 = {k: v.to(dev) for k, v in ck['P0'].items()}
    cfgs = ck['cfgs']
    TS = test_sets(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(777)
    rng = np.random.default_rng(0)
    gq = torch.Generator(device=dev)
    gq.manual_seed(4242)
    basis = [torch.linalg.qr(torch.randn(WIDTH, WIDTH, generator=gq, device=dev))[0] for _ in range(DEPTH)]
    sel = range(len(cfgs)) if args.models == 'all' else [int(x) for x in args.models.split(',')]
    results = []
    with torch.no_grad():
        for m in sel:
            r = analyze_model(slice_model(P, m), slice_model(P0, m), cfgs[m], nm, TS, dev, gen, rng, basis)
            r['model'] = m
            results.append(r)
            ov = r['overlap']
            print(f"model {m} {cfgs[m]} opt={r['bal']['opt']:.3f} J(fail,succ)={ov['fail_succ']['J']:.3f} "
                  f"x{ov['fail_succ']['ratio_layer']:.2f} rot x{ov['rot_fail_succ']['ratio_layer']:.2f} "
                  f"untr x{ov['untrained_fail_succ']['ratio_layer']:.2f} ({r['seconds']:.1f}s)", flush=True)
    od = os.path.join(args.out, args.name)
    os.makedirs(od, exist_ok=True)
    json.dump(to_py(results), open(os.path.join(od, 'analysis.json'), 'w'))
    rows = []
    for r in results:
        c = r['cfg']
        base = dict(model=r['model'], p=c['p'], mode=c['mode'], seed=c['seed'])
        for wc in r['weak_curve']:
            row = dict(base, kind='weak', alpha=wc['alpha'], frac_entries=wc['frac_entries'], frac_mass=wc['frac_mass'])
            for src in ('target', 'random'):
                for kk in ('acc_s', 'acc_f', 'acc_miss', 'acc_slip', 'mse_s'):
                    row[f'{src}_{kk}'] = wc[src][kk]
            rows.append(row)
        for rn, wc in r['regime_ablation'].items():
            row = dict(base, kind=rn, alpha=float('nan'), frac_entries=wc['frac_entries'], frac_mass=wc['frac_mass'])
            for src in ('target', 'random'):
                for kk in ('acc_s', 'acc_f', 'acc_miss', 'acc_slip', 'mse_s'):
                    row[f'{src}_{kk}'] = wc[src][kk]
            rows.append(row)
    with open(os.path.join(od, 'ablation_rows.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('analysis done', flush=True)


if __name__ == '__main__':
    main()
