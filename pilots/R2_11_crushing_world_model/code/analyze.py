"""R2-11 analysis (PLAN.md section 4): resolution probes, confusion, neuron locality with rotation / untrained nulls,
merged-neuron check, variance shares. JSON is written incrementally to <out>/analysis/.

Usage: python analyze.py --data /workspace/data --ckpt /workspace/ckpt --out /workspace/results [--seeds 0,1]
"""
import argparse
import json
import math
import os
import time

import torch

import env as E
from model import CONFIGS, make_model

VARS = ['A', 'B', 'R', 'C', 'W', 'S']
# v1 = matrix registered at 08:22Z; v2 (primary, PLAN section 10, 08:50Z) marks A-S and R-C as physically linked
DMAT_V1 = torch.tensor([[0, .5, 1, .5, 1, 1],
                        [.5, 0, 0, .5, 0, .5],
                        [1, 0, 0, 1, .5, .5],
                        [.5, .5, 1, 0, .5, .5],
                        [1, 0, .5, .5, 0, .5],
                        [1, .5, .5, .5, .5, 0]], dtype=torch.float64)
DMAT = DMAT_V1.clone()
DMAT[0, 5] = DMAT[5, 0] = 0.5
DMAT[2, 3] = DMAT[3, 2] = 0.5
UNREL = {i: [k for k in range(6) if DMAT_V1[i, k] == 1] for i in range(6)}       # v1 confusion (all samples)
REL = {i: [k for k in range(6) if k != i and DMAT_V1[i, k] <= 0.5] for i in range(6)}
IU = torch.triu_indices(6, 6, 1)
DBAR = float(DMAT[IU[0], IU[1]].mean())
SPREAD_U = float(DMAT.sum() / 36)
CONF_VARS = [i for i in range(6) if UNREL[i]]     # A R C W S (v1)
# v2 confusion (primary): non-interacting samples, own value + own innovation controls
CONF2 = [('A', ['R', 'W'], ['B']), ('R', ['A'], ['B', 'W']), ('W', ['A'], ['B', 'R'])]
NPER = 38
DEV = 'cuda'


def jdump(obj, fn):
    tmp = fn + '.tmp'
    json.dump(obj, open(tmp, 'w'), indent=1)
    os.replace(tmp, fn)


# ------------------------------------------------------------------------------------------------ GT and bases
def rbf1(x, lo, hi, n):
    c = torch.linspace(lo, hi, n, device=x.device, dtype=torch.float64)
    s = (hi - lo) / (n - 1)
    return torch.exp(-0.5 * ((x[:, None] - c[None]) / s) ** 2)


def rbf2(p, lo, hi, n):
    gx = rbf1(p[:, 0], lo, hi, n)
    gy = rbf1(p[:, 1], lo, hi, n)
    return (gx[:, :, None] * gy[:, None, :]).reshape(p.shape[0], n * n)


def gt_tables(d, ep, t):
    """GT variables (frame t+1) and auxiliary quantities for samples (ep, t)."""
    g = lambda k, dt=0: d[k][ep, t + dt].double().to(DEV)
    pa0, pa1, pb0, pb1 = g('pa'), g('pa', 1), g('pb'), g('pb', 1)
    th0, th1 = g('th'), g('th', 1)
    gap1 = E.box_gap(pa1, pb1, th1)
    wg1 = E.wall_gap_box(pb1, th1).clamp(0, 20)
    spd = (pb1 - pb0).norm(dim=-1)
    dth = th1 - th0
    dth = dth - (math.pi / 2) * torch.round(dth / (math.pi / 2))
    con1 = (gap1 <= 0.5).double()
    near1 = (wg1 <= 0.5).double()
    c4, s4, c8, s8 = torch.cos(4 * th1), torch.sin(4 * th1), torch.cos(8 * th1), torch.sin(8 * th1)
    gap0 = E.box_gap(pa0, pb0, th0)
    free = (gap0 > 1.0) & (gap1 > 1.0) & (spd < 0.05) & (dth.abs() < math.radians(0.05))
    dpa = pa1 - pa0
    Y = {'A': pa1, 'B': pb1, 'R': torch.stack([c4, s4], 1), 'C': con1[:, None], 'W': wg1[:, None], 'S': spd[:, None]}
    # confusion bases (values at t+1)
    LOB, HIB = 16.0, 96.0
    basis = {
        'A': torch.cat([rbf2(pa1, E.LO_A, E.HI_A, 7), pa1 / 112], 1),
        'B': torch.cat([rbf2(pb1, LOB, HIB, 7), pb1 / 112], 1),
        'R': torch.stack([c4, s4, c8, s8], 1),
        'C': torch.cat([con1[:, None], rbf1(gap1.clamp(0, 12), 0, 12, 4)], 1),
        'W': torch.cat([rbf1(wg1, 0, 20, 5), near1[:, None]], 1),
        'S': torch.cat([rbf1(spd.clamp(0, 5), 0, 5, 5), spd[:, None] / 5], 1),
    }
    # encoding families (state at t and t+1) for the neuron analysis and variance shares
    ba4 = rbf2(pa1, E.LO_A, E.HI_A, 4)
    bb4 = rbf2(pb1, LOB, HIB, 4)
    fam = {
        'A': torch.cat([rbf2(pa0, E.LO_A, E.HI_A, 7), rbf2(pa1, E.LO_A, E.HI_A, 7), (pa1 - pa0) / 4], 1),
        'B': torch.cat([rbf2(pb0, LOB, HIB, 7), rbf2(pb1, LOB, HIB, 7)], 1),
        'R': torch.cat([torch.stack([c4, s4, c8, s8], 1), c4[:, None] * bb4, s4[:, None] * bb4], 1),
        'C': torch.cat([con1[:, None], rbf1(gap1.clamp(0, 12), 0, 12, 4), con1[:, None] * ba4], 1),
        'W': torch.cat([rbf1(wg1, 0, 20, 5), near1[:, None], near1[:, None] * bb4], 1),
        'S': torch.cat([(pb1 - pb0) / 4, rbf1(spd.clamp(0, 5), 0, 5, 5), spd[:, None] * bb4, dth.abs()[:, None]], 1),
    }
    own2 = {
        'A': torch.cat([basis['A'], dpa / 4, rbf1(dpa[:, 0].clamp(-8, 8), -8, 8, 5), rbf1(dpa[:, 1].clamp(-8, 8), -8, 8, 5),
                        dpa.norm(dim=-1, keepdim=True) / 4], 1),
        'R': basis['R'], 'W': basis['W']}
    return Y, basis, fam, own2, free


# ------------------------------------------------------------------------------------------------ probes
def gram(X1, X2, ch=2048):
    """X1 [n1,D], X2 [n2,D] (fp16 on GPU) -> X1 X2^T / D in float64."""
    out = torch.empty(X1.shape[0], X2.shape[0], dtype=torch.float64, device=DEV)
    for i in range(0, X1.shape[0], ch):
        a = X1[i:i + ch].float()
        for j in range(0, X2.shape[0], ch):
            out[i:i + ch, j:j + ch] = (a @ X2[j:j + ch].float().T).double()
    return out / X1.shape[1]


def chol(K, lam):
    I = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    for jit in (1.0, 10.0, 100.0):
        L, info = torch.linalg.cholesky_ex(K + lam * jit * I)
        if int(info) == 0:
            return L
    raise RuntimeError('cholesky failed')


def kridge(Ktr, Kte, Ytr, fit, val, rels=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Kernel ridge with per-column lambda chosen on an inner split. Ytr [n,q] float64. Returns test preds [m,q]."""
    Ktr = 0.5 * (Ktr + Ktr.T)
    md = float(Ktr.diagonal().mean())
    Kff = Ktr[fit][:, fit]
    Kvf = Ktr[val][:, fit]
    Yf = Ytr[fit]
    mf = Yf.mean(0)
    errs = []
    for r in rels:
        L = chol(Kff, r * md)
        al = torch.cholesky_solve(Yf - mf, L)
        errs.append(((Kvf @ al + mf - Ytr[val]) ** 2).mean(0))
    best = torch.stack(errs).argmin(0)                      # [q]
    out = torch.empty(Kte.shape[0], Ytr.shape[1], dtype=torch.float64, device=DEV)
    m = Ytr.mean(0)
    for bi in best.unique().tolist():
        cols = (best == bi).nonzero().flatten()
        L = chol(Ktr, rels[bi] * md)
        al = torch.cholesky_solve(Ytr[:, cols] - m[cols], L)
        out[:, cols] = Kte @ al + m[cols]
    return out, [rels[b] for b in best.tolist()]


def ridge_cv_r2(Y, X, folds, lam_rel=1e-3):
    """2-fold cross-fitted ridge R^2 (pooled over columns of Y)."""
    pred = torch.empty_like(Y)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        Xt = X[tr]
        mx, my = Xt.mean(0), Y[tr].mean(0)
        Xc = Xt - mx
        G = Xc.T @ Xc + lam_rel * Xt.shape[0] * torch.eye(X.shape[1], dtype=X.dtype, device=DEV)
        W = torch.linalg.solve(G, Xc.T @ (Y[tr] - my))
        pred[te] = (X[te] - mx) @ W + my
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum().clamp_min(1e-30))


def std_cols(X):
    return (X - X.mean(0)) / X.std(0).clamp_min(1e-6)


def confusion(err, basis_te, folds, idx=None, with_rel=True):
    """Partial R^2 of probe errors on unrelated (and related) variables' bases beyond the variable's own basis."""
    out = {}
    for i in CONF_VARS:
        v = VARS[i]
        e = err[v] if idx is None else err[v][idx]
        bb = (lambda k: basis_te[VARS[k]] if idx is None else basis_te[VARS[k]][idx])
        fo = folds if idx is None else folds[idx]
        O = bb(i)
        r_own = ridge_cv_r2(e, O, fo)
        U = torch.cat([O] + [bb(k) for k in UNREL[i]], 1)
        out[v] = dict(unrel=ridge_cv_r2(e, U, fo) - r_own, own=r_own)
        if with_rel:
            Rl = torch.cat([O] + [bb(k) for k in REL[i]], 1)
            out[v]['rel'] = ridge_cv_r2(e, Rl, fo) - r_own
    out['CI'] = sum(out[VARS[i]]['unrel'] for i in CONF_VARS) / len(CONF_VARS)
    if with_rel:
        out['CI_rel'] = sum(out[VARS[i]]['rel'] for i in CONF_VARS) / len(CONF_VARS)
    return out


def confusion2(err, basis_te, own2_te, free_te, folds, idx=None):
    """Primary confusion (v2): non-interacting samples only (no contact at t, t+1; box static), so that physics
    couples none of the pairs; controls = own value + own innovation. Unrelated pairs: A-R, A-W."""
    sel = free_te.nonzero().flatten() if idx is None else idx[free_te[idx]]
    out = dict(n=int(sel.numel()))
    for v, un, rl in CONF2:
        e, O, fo = err[v][sel], own2_te[v][sel], folds[sel]
        r_own = ridge_cv_r2(e, O, fo)
        out[v] = dict(unrel=ridge_cv_r2(e, torch.cat([O] + [basis_te[k][sel] for k in un], 1), fo) - r_own,
                      rel=ridge_cv_r2(e, torch.cat([O] + [basis_te[k][sel] for k in rl], 1), fo) - r_own, own=r_own)
    out['CI'] = sum(out[v]['unrel'] for v, _, _ in CONF2) / len(CONF2)
    out['CI_rel'] = sum(out[v]['rel'] for v, _, _ in CONF2) / len(CONF2)
    return out


# ------------------------------------------------------------------------------------------------ neurons
@torch.no_grad()
def activations(model, Z, A, ep, t, bs=400):
    """MLP hidden (post-GELU) activations at current-frame tokens for all blocks: list of [N,64,h] fp32."""
    acts = [[] for _ in model.blocks]
    hooks = []
    for b, blk in enumerate(model.blocks):
        hooks.append(blk.mlp[1].register_forward_hook(
            lambda m, i, o, b=b: acts[b].append(o[:, -64:].float())))
    for i in range(0, ep.numel(), bs):
        e, tt = ep[i:i + bs], t[i:i + bs]
        fr = tt[:, None] + torch.arange(-2, 1, device=DEV)[None]
        model(Z[e[:, None], fr].float(), A[e, tt])
    for h in hooks:
        h.remove()
    return [torch.cat(a, 0) for a in acts]


def unit_metrics(Emat, R2f, Dm=None):
    """Locality metrics over units. Emat [u,6] unique variance fractions, R2f [u] full R^2."""
    Dm = DMAT if Dm is None else Dm
    dbar = float(Dm[IU[0], IU[1]].mean())
    spread_u = float(Dm.sum() / 36)
    inc = R2f >= 0.1
    Em, n_inc = Emat[inc], int(inc.sum())
    res = dict(n_units=int(R2f.numel()), n_incl=n_inc)
    if n_inc == 0:
        return res
    Ec = (Em - 0.005).clamp_min(0)
    ssum = Ec.sum(1)
    ok = ssum > 0
    p = Ec[ok] / ssum[ok, None]
    spread = torch.einsum('ui,ij,uj->u', p, Dm.to(p), p)
    Lw = 1 - spread / spread_u
    thr = torch.maximum(torch.full_like(Em[:, 0], 0.01), 0.3 * Em.max(1).values)
    S = Em >= thr[:, None]
    nset = S.sum(1)
    multi = nset >= 2
    iu = IU.to(Em.device)
    pairD = Dm.to(Em)[iu[0], iu[1]]                                    # [15]
    Sp = (S[:, iu[0]] & S[:, iu[1]]).double()                          # [u,15]
    npair = Sp.sum(1)
    meanD = (Sp * pairD).sum(1) / npair.clamp_min(1)
    sli = 1 - meanD / dbar
    pack = ((Sp * (pairD == 1).double()).sum(1) > 0)
    dom = Em.argmax(1)
    res.update(n_lw=int(ok.sum()), Lw=float(Lw.mean()) if ok.any() else None,
               frac_multi=float(multi.double().mean()),
               SLI_multi=float(sli[multi].mean()) if multi.any() else None, n_multi=int(multi.sum()),
               pack_rate=float(pack.double().mean()), mean_E={v: float(Em[:, k].mean()) for k, v in enumerate(VARS)},
               dom_hist={v: int((dom == k).sum()) for k, v in enumerate(VARS)},
               mean_R2full=float(R2f[inc].mean()), mean_nset=float(nset.double().mean()))
    return res


def encode_units(acts, Ftr, Fte, fam_cols, n_rot=10, rot_seed=0):
    """Per-token ridge encoding of each channel. Returns per-block E [h,6], R2 [h] (+ rotated versions)."""
    out = []
    ntr = Ftr.shape[0]
    nfit = int(0.8 * ntr)
    for b, a in enumerate(acts):
        h = a.shape[-1]
        Atr, Ate = a[:ntr].double(), a[ntr:].double()
        mu = Atr.mean(0, keepdim=True)
        Ytr = (Atr - mu).reshape(ntr, -1)
        Yte = (Ate - mu).reshape(Ate.shape[0], -1)
        # lambda by inner split (full model)
        best, bl = None, None
        for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            X1, Y1 = Ftr[:nfit], Ytr[:nfit]
            G = X1.T @ X1 + lam * torch.eye(X1.shape[1], dtype=X1.dtype, device=DEV)
            W = torch.linalg.solve(G, X1.T @ (Y1 - Y1.mean(0)))
            sse = float(((Ytr[nfit:] - Y1.mean(0) - Ftr[nfit:] @ W) ** 2).sum())
            if best is None or sse < best:
                best, bl = sse, lam
        res_list = []
        for drop in [None] + list(range(6)):
            cols = torch.cat([c for k, c in enumerate(fam_cols) if k != drop])
            X = Ftr[:, cols]
            G = X.T @ X + bl * torch.eye(X.shape[1], dtype=X.dtype, device=DEV)
            W = torch.linalg.solve(G, X.T @ Ytr)
            res_list.append((Yte - Fte[:, cols] @ W).reshape(-1, 64, h))
        Yte3 = Yte.reshape(-1, 64, h)

        def metrics(Q):
            if Q is None:
                sst = (Yte3 ** 2).sum((0, 1))
                sse = [(r ** 2).sum((0, 1)) for r in res_list]
            else:
                sst = ((Yte3 @ Q.T) ** 2).sum((0, 1))
                sse = [((r @ Q.T) ** 2).sum((0, 1)) for r in res_list]
            sst = sst.clamp_min(1e-12)
            R2 = 1 - sse[0] / sst
            Em = torch.stack([(sse[k + 1] - sse[0]) / sst for k in range(6)], 1)
            live = sst / (Yte3.shape[0] * 64) > 1e-8
            R2 = torch.where(live, R2, torch.zeros_like(R2))
            return Em, R2
        Em, R2 = metrics(None)
        rots = []
        g = torch.Generator(device='cpu')
        g.manual_seed(1000 * rot_seed + b)
        for r in range(n_rot):
            M = torch.randn(h, h, generator=g, dtype=torch.float64).to(DEV)
            Q, Rr = torch.linalg.qr(M)
            Q = Q * torch.sign(torch.diagonal(Rr))[None]
            rots.append(metrics(Q))
        out.append(dict(E=Em, R2=R2, rots=rots, lam=bl))
        del res_list
    return out


def pooled_unit_metrics(enc):
    E_all = torch.cat([o['E'] for o in enc])
    R_all = torch.cat([o['R2'] for o in enc])
    res = dict(pooled=unit_metrics(E_all, R_all), per_block=[unit_metrics(o['E'], o['R2']) for o in enc],
               pooled_v1=unit_metrics(E_all, R_all, DMAT_V1))
    n_rot = len(enc[0]['rots'])
    rot, rot1 = [], []
    for r in range(n_rot):
        Er = torch.cat([o['rots'][r][0] for o in enc])
        Rr = torch.cat([o['rots'][r][1] for o in enc])
        rot.append(unit_metrics(Er, Rr))
        rot1.append(unit_metrics(Er, Rr, DMAT_V1))
    res['rot'] = rot
    res['rot_v1'] = rot1
    return res


# ------------------------------------------------------------------------------------------------ merged neurons
def omp_r2(Xtr, Ytr, Xte, Yte, k=5):
    """Greedy OMP in the Gram domain, vectorised over targets. X [n,p] standardised wide units, Y [n,q] centred
    narrow units. Returns per-target test R^2 for top-1 and top-k, selected indices [q,k] and |coefficients| [q,k]."""
    G = Xtr.T @ Xtr / Xtr.shape[0]
    C = Xtr.T @ Ytr / Xtr.shape[0]
    Gt = Xte.T @ Xte / Xte.shape[0]
    Ct = Xte.T @ Yte / Xte.shape[0]
    yyt = (Yte ** 2).mean(0).clamp_min(1e-12)
    p, q = G.shape[0], C.shape[1]
    qa = torch.arange(q, device=DEV)
    S = torch.empty(q, 0, dtype=torch.long, device=DEV)
    out = {}
    for step in range(k):
        cand = torch.arange(p, device=DEV)[None, :, None].expand(q, p, 1)
        idx = torch.cat([S[:, None, :].expand(q, p, S.shape[1]), cand], 2)                 # [q,p,s+1]
        eye = torch.eye(idx.shape[2], device=DEV, dtype=G.dtype)
        Gs = G[idx[..., :, None], idx[..., None, :]] + 1e-6 * eye                           # [q,p,s+1,s+1]
        cs = C[idx, qa[:, None, None]]                                                      # [q,p,s+1]
        bet = torch.linalg.solve(Gs, cs[..., None])[..., 0]
        expl = (bet * cs).sum(-1)                                                           # [q,p]
        if S.shape[1] > 0:
            expl = expl.scatter(1, S, float('-inf'))
        S = torch.cat([S, expl.argmax(1)[:, None]], 1)
        if step == 0 or step == k - 1:
            eye = torch.eye(S.shape[1], device=DEV, dtype=G.dtype)
            Gsel = G[S[:, :, None], S[:, None, :]] + 1e-6 * eye
            bt = torch.linalg.solve(Gsel, C[S, qa[:, None]][..., None])[..., 0]            # [q,s]
            sse = yyt - 2 * (bt * Ct[S, qa[:, None]]).sum(1) + torch.einsum('qi,qij,qj->q', bt, Gt[S[:, :, None], S[:, None, :]], bt)
            out[step] = (1 - sse / yyt).tolist()
            if step == k - 1:
                out['beta'] = bt.abs()
    return out[0], out[k - 1], S, out['beta']


def ridge_all_r2(Xtr, Ytr, Xte, Yte, lam_rel=1e-3):
    n = Xtr.shape[0]
    G = Xtr.T @ Xtr + lam_rel * n * torch.eye(Xtr.shape[1], device=DEV, dtype=Xtr.dtype)
    W = torch.linalg.solve(G, Xtr.T @ Ytr)
    return (1 - ((Yte - Xte @ W) ** 2).sum(0) / (Yte ** 2).sum(0).clamp_min(1e-12)).tolist()


def merged_check(narrow, wide, wide_prof, ntr, n_rot=3, max_targets=64, seed=0):
    """narrow, wide: lists over blocks of [N,64,h]. wide_prof: list over blocks of [256,6] normalised GT profiles."""
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    out = []
    for b in range(len(narrow)):
        Xa, Ya = wide[b].double(), narrow[b].double()
        hn = Ya.shape[-1]
        tsel = torch.randperm(hn, generator=g)[:max_targets].to(DEV)
        Xtr, Xte = Xa[:ntr].reshape(-1, Xa.shape[-1]), Xa[ntr:].reshape(-1, Xa.shape[-1])
        Ytr, Yte = Ya[:ntr].reshape(-1, hn)[:, tsel], Ya[ntr:].reshape(-1, hn)[:, tsel]
        mx, sx = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        my = Ytr.mean(0)
        Xtr, Xte = (Xtr - mx) / sx, (Xte - mx) / sx
        Ytr, Yte = Ytr - my, Yte - my
        live = (Ytr.std(0) > 1e-4)
        r_all = ridge_all_r2(Xtr, Ytr, Xte, Yte)
        r1, r5, sel, bet = omp_r2(Xtr, Ytr, Xte, Yte)
        # semantic coherence of the selected wide units: expected D between profiles, coefficient-weighted
        P = wide_prof[b]
        valid = P.sum(1) > 0
        coh = []
        for S_t, w_t in zip(sel, bet):
            m = valid[S_t]
            if m.sum() < 2:
                coh.append(None)
                continue
            Ps, ws = P[S_t[m]], w_t[m]
            Dp = Ps @ DMAT.to(Ps) @ Ps.T
            W2 = ws[:, None] * ws[None]
            W2.fill_diagonal_(0)
            coh.append(float((W2 * Dp).sum() / W2.sum().clamp_min(1e-12)))
        vi = valid.nonzero().flatten()
        rnd = []
        for _ in range(200 if vi.numel() >= 5 else 0):
            S_t = vi[torch.randperm(vi.numel(), generator=g)[:5].to(DEV)]
            Dp = P[S_t] @ DMAT.to(P) @ P[S_t].T
            rnd.append(float((Dp.sum() - Dp.diagonal().sum()) / 20))
        rot1, rot5 = [], []
        for r in range(n_rot):
            M = torch.randn(Xtr.shape[1], Xtr.shape[1], generator=g, dtype=torch.float64).to(DEV)
            Q, _ = torch.linalg.qr(M)
            a1, a5, _, _ = omp_r2(Xtr @ Q, Ytr, Xte @ Q, Yte)
            rot1.append(a1)
            rot5.append(a5)
        lv = live.tolist()
        f = lambda xs: float(sum(x for x, l in zip(xs, lv) if l) / max(1, sum(lv)))
        cv = [c for c, l in zip(coh, lv) if l and c is not None]
        out.append(dict(n=int(sum(lv)), R2_1=f(r1), R2_5=f(r5), R2_all=f(r_all),
                        rot_R2_1=sum(f(x) for x in rot1) / n_rot, rot_R2_5=sum(f(x) for x in rot5) / n_rot,
                        frac_merged=float(sum(1 for a, c, l in zip(r1, r5, lv) if l and c - a >= 0.1) / max(1, sum(lv))),
                        coh_D=(sum(cv) / len(cv)) if cv else None, rand_D=(sum(rnd) / len(rnd)) if rnd else None,
                        n_coh=len(cv)))
    summ = {}
    for k in ['R2_1', 'R2_5', 'R2_all', 'rot_R2_1', 'rot_R2_5', 'frac_merged', 'rand_D', 'coh_D']:
        xs = [o[k] for o in out if o[k] is not None]
        summ[k] = sum(xs) / len(xs) if xs else None
    return dict(summary=summ, per_block=out)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/workspace/data')
    ap.add_argument('--ckpt', default='/workspace/ckpt')
    ap.add_argument('--out', default='/workspace/results')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--names', default=','.join(CONFIGS))
    ap.add_argument('--tag', default='final')
    ap.add_argument('--n_boot', type=int, default=200)
    ap.add_argument('--n_rot', type=int, default=10)
    ap.add_argument('--ep_split', type=int, default=240)
    ap.add_argument('--n_neu_tr', type=int, default=3200)
    ap.add_argument('--n_neu_te', type=int, default=1200)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    od = f'{args.out}/analysis'
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    st = torch.load(f'{args.data}/stats.pt')
    mu, sd = st['mu'].to(DEV), st['sd'].to(DEV)
    d = torch.load(f'{args.data}/ana.pt')
    n_ep = d['Z'].shape[0]
    Z = torch.empty(d['Z'].shape, dtype=torch.float16, device=DEV)
    for i in range(0, n_ep, 50):
        Z[i:i + 50] = ((d['Z'][i:i + 50].to(DEV).float() - mu) / sd).half()
    Aact = d['a'].to(DEV).float()
    split = min(args.ep_split, int(0.6 * n_ep))
    ep = torch.arange(n_ep, device=DEV).repeat_interleave(NPER)
    tt = torch.arange(2, 2 + NPER, device=DEV).repeat(n_ep)
    tr = ep < split
    te = ~tr
    fit = (ep[tr] < int(0.75 * split)).nonzero().flatten()
    val = (ep[tr] >= int(0.75 * split)).nonzero().flatten()
    Yd, basis, fam, own2, free = gt_tables(d, ep.cpu(), tt.cpu())
    folds = (ep[te] % 2).to(DEV)
    names_y = [(v, Yd[v].shape[1]) for v in VARS]
    Ycat = torch.cat([Yd[v] for v in VARS], 1)
    Ytr, Yte = Ycat[tr], Ycat[te]
    basis_te = {v: std_cols(basis[v][te]) for v in VARS}
    own2_te = {v: std_cols(own2[v][te]) for v in own2}
    free_te = free[te]
    print(f'data {time.time() - t0:.0f}s  samples {ep.numel()} train {int(tr.sum())} test {int(te.sum())}', flush=True)

    def per_var(pred):
        o, c = {}, 0
        for v, k in names_y:
            o[v] = pred[:, c:c + k]
            c += k
        return o

    def var_mse(pred):
        P = per_var(pred)
        T = per_var(Yte)
        return {v: float(((P[v] - T[v]) ** 2).sum(1).mean()) for v in VARS}, {v: P[v] - T[v] for v in VARS}

    var_y = {v: float(((per_var(Yte)[v] - per_var(Yte)[v].mean(0)) ** 2).sum(1).mean()) for v in VARS}
    # shared kernels: current frame (copy control) and true next frame (oracle)
    zt = Z[ep, tt].reshape(ep.numel(), -1)
    z1 = Z[ep, tt + 1].reshape(ep.numel(), -1)
    mzt, mz1 = zt[tr].float().mean(0).half(), z1[tr].float().mean(0).half()
    zt_c, z1_c = zt - mzt, z1 - mz1
    Kt_tr, Kt_te = gram(zt_c[tr], zt_c[tr]), gram(zt_c[te], zt_c[tr])
    K1_tr, K1_te = gram(z1_c[tr], z1_c[tr]), gram(z1_c[te], z1_c[tr])
    del z1_c, zt_c
    shared = dict(vars=VARS, D=DMAT.tolist(), D_v1=DMAT_V1.tolist(), DBAR=DBAR, SPREAD_U=SPREAD_U, var_y=var_y,
                  n_train=int(tr.sum()),
                  n_test=int(te.sum()))
    copy_pred, lam_copy = kridge(Kt_tr, Kt_te, Ytr, fit, val)
    mse_copy, err_copy = var_mse(copy_pred)
    orc_pred, lam_orc = kridge(Kt_tr + K1_tr, Kt_te + K1_te, Ytr, fit, val)
    mse_orc, err_orc = var_mse(orc_pred)
    shared['copy'] = dict(mse=mse_copy, raw_r2={v: 1 - mse_copy[v] / var_y[v] for v in VARS}, lam=lam_copy)
    shared['oracle'] = dict(mse=mse_orc, raw_r2={v: 1 - mse_orc[v] / var_y[v] for v in VARS},
                            r2_delta={v: 1 - mse_orc[v] / mse_copy[v] for v in VARS}, lam=lam_orc)
    shared['oracle']['confusion_v1'] = confusion(err_orc, basis_te, folds)
    shared['copy']['confusion_v1'] = confusion(err_copy, basis_te, folds)
    shared['oracle']['confusion'] = confusion2(err_orc, basis_te, own2_te, free_te, folds)
    shared['copy']['confusion'] = confusion2(err_copy, basis_te, own2_te, free_te, folds)
    shared['n_free_test'] = int(free_te.sum())
    print('copy/oracle probes', {v: round(shared['oracle']['r2_delta'][v], 3) for v in VARS},
          f'{time.time() - t0:.0f}s', flush=True)
    # variance shares of the true innovation dz = z_{t+1} - z_t
    fam_std = {v: std_cols(fam[v]) for v in VARS}
    dz_tr, dz_te = (z1[tr] - zt[tr]), (z1[te] - zt[te])
    mdz = dz_tr.float().mean(0)
    shares = {}
    for v in VARS + ['all']:
        X = torch.cat([fam_std[u] for u in VARS], 1) if v == 'all' else fam_std[v]
        Xtr_, Xte_ = X[tr], X[te]
        mx = Xtr_.mean(0)
        G = (Xtr_ - mx).T @ (Xtr_ - mx) + 1e-2 * Xtr_.shape[0] * torch.eye(X.shape[1], dtype=X.dtype, device=DEV)
        Gi = torch.linalg.inv(G)
        sse, sst = 0.0, 0.0
        for c0 in range(0, dz_tr.shape[1], 4096):
            Yc_tr = (dz_tr[:, c0:c0 + 4096].float() - mdz[c0:c0 + 4096]).double()
            Wc = Gi @ ((Xtr_ - mx).T @ Yc_tr)
            Yc_te = (dz_te[:, c0:c0 + 4096].float() - mdz[c0:c0 + 4096]).double()
            sse += float(((Yc_te - (Xte_ - mx) @ Wc) ** 2).sum())
            sst += float((Yc_te ** 2).sum())
        shares[v] = 1 - sse / sst
    shared['shares'] = shares
    shared['fam_sizes'] = {v: fam[v].shape[1] for v in VARS}
    del dz_tr, dz_te
    print('shares', {k: round(x, 4) for k, x in shares.items()}, f'{time.time() - t0:.0f}s', flush=True)
    jdump(shared, f'{od}/shared.json')

    # neuron subset
    g = torch.Generator(device='cpu')
    g.manual_seed(4242)
    tr_idx = tr.nonzero().flatten()
    te_idx = te.nonzero().flatten()
    n_tr = min(args.n_neu_tr, tr_idx.numel())
    n_te = min(args.n_neu_te, te_idx.numel())
    nidx = torch.cat([tr_idx[torch.randperm(tr_idx.numel(), generator=g)[:n_tr].to(DEV)],
                      te_idx[torch.randperm(te_idx.numel(), generator=g)[:n_te].to(DEV)]])
    F = torch.cat([fam_std[v][nidx] for v in VARS], 1)
    Fm, Fs = F[:n_tr].mean(0), F[:n_tr].std(0).clamp_min(1e-6)
    F = (F - Fm) / Fs
    Ftr, Fte = F[:n_tr], F[n_tr:]
    fam_cols, c = [], 0
    for v in VARS:
        fam_cols.append(torch.arange(c, c + fam[v].shape[1], device=DEV))
        c += fam[v].shape[1]
    ep_n, t_n = ep[nidx], tt[nidx]

    # bootstrap resamples of test episodes (shared by all models -> paired comparisons)
    te_eps = ep[te].unique()
    gb = torch.Generator(device='cpu')
    gb.manual_seed(777)
    boots = []
    ep_te = ep[te]
    pos = {int(e): (ep_te == e).nonzero().flatten() for e in te_eps.tolist()}
    for b in range(args.n_boot):
        pick = te_eps[torch.randint(te_eps.numel(), (te_eps.numel(),), generator=gb).to(DEV)]
        boots.append(torch.cat([pos[int(e)] for e in pick.tolist()]))
    gram_cache = {}
    wide_store = {}
    for s in [int(x) for x in args.seeds.split(',')]:
        for name in args.names.split(','):
            for kind in ('trained', 'untrained'):
                fn = f'{args.ckpt}/{name}_s{s}_{args.tag if kind == "trained" else "init"}.pt'
                if not os.path.exists(fn):
                    print('missing', fn, flush=True)
                    continue
                t1 = time.time()
                model = make_model(name).to(DEV)
                model.load_state_dict(torch.load(fn, map_location=DEV))
                model.eval()
                rec = dict(name=name, seed=s, kind=kind, d_model=CONFIGS[name][0], mlp=CONFIGS[name][1])
                if kind == 'trained':
                    # predictions on all ana samples
                    zh = torch.empty_like(zt)
                    with torch.no_grad():
                        for i in range(0, ep.numel(), 500):
                            e, t_ = ep[i:i + 500], tt[i:i + 500]
                            fr = t_[:, None] + torch.arange(-2, 1, device=DEV)[None]
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                p = model(Z[e[:, None], fr].float(), Aact[e, t_])
                            zh[i:i + 500] = p.float().reshape(p.shape[0], -1).half()
                    rec['pred_mse_test'] = float(((zh[te].float() - z1[te].float()) ** 2).mean())
                    rec['copy_mse_test'] = float(((zt[te].float() - z1[te].float()) ** 2).mean())
                    mzh = zh[tr].float().mean(0).half()
                    zh_c = zh - mzh
                    Kh_tr, Kh_te = gram(zh_c[tr], zh_c[tr]), gram(zh_c[te], zh_c[tr])
                    del zh, zh_c
                    pred, lam = kridge(Kh_tr + Kt_tr, Kh_te + Kt_te, Ytr, fit, val)
                    del Kh_tr, Kh_te
                    mse, err = var_mse(pred)
                    rec['probe'] = dict(mse=mse, lam=lam, raw_r2={v: 1 - mse[v] / var_y[v] for v in VARS},
                                        r2_delta={v: 1 - mse[v] / mse_copy[v] for v in VARS})
                    rec['confusion_v1'] = confusion(err, basis_te, folds)
                    rec['confusion'] = confusion2(err, basis_te, own2_te, free_te, folds)
                    rec['confusion_boot'] = [confusion2(err, basis_te, own2_te, free_te, folds, idx=bi)['CI'] for bi in boots]
                    rec['confusion_v1_boot'] = [confusion(err, basis_te, folds, idx=bi, with_rel=False)['CI'] for bi in boots]
                    print(f"[{name} s{s}] r2_delta", {v: round(rec['probe']['r2_delta'][v], 3) for v in VARS},
                          'CI', round(rec['confusion']['CI'], 4), f'{time.time() - t1:.0f}s', flush=True)
                # neurons
                acts = activations(model, Z, Aact, ep_n, t_n)
                enc = encode_units(acts, Ftr, Fte, fam_cols, n_rot=args.n_rot, rot_seed=s)
                rec['neurons'] = pooled_unit_metrics(enc)
                pm = rec['neurons']['pooled']
                rl = [r.get('Lw') for r in rec['neurons']['rot'] if r.get('Lw') is not None]
                print(f"[{name} s{s} {kind}] Lw {pm.get('Lw')} rotLw {sum(rl) / max(1, len(rl)):.3f} "
                      f"pack {pm.get('pack_rate')} n_incl {pm.get('n_incl')}/{pm.get('n_units')} "
                      f"{time.time() - t1:.0f}s", flush=True)
                if name == 'w256' and kind == 'trained':
                    prof = []
                    for o in enc:
                        Ec = (o['E'] - 0.005).clamp_min(0)
                        ssum = Ec.sum(1, keepdim=True)
                        P = torch.where(ssum > 0, Ec / ssum.clamp_min(1e-12), torch.zeros_like(Ec))
                        P[o['R2'] < 0.1] = 0
                        prof.append(P)
                    wide_store[s] = dict(acts=[a.half() for a in acts], prof=prof)
                elif s in wide_store and name != 'w256':
                    W = wide_store[s]
                    rec['merged'] = merged_check(acts, [a.float() for a in W['acts']], W['prof'], n_tr, seed=s)
                    print(f"[{name} s{s} {kind}] merged", {k: (round(v, 3) if v is not None else None)
                                                          for k, v in rec['merged']['summary'].items()}, flush=True)
                del acts, enc
                jdump(rec, f'{od}/{name}_s{s}_{kind}.json')
                del model
                torch.cuda.empty_cache()
    # cross-seed reference: wide seed 1 units explained by wide seed 0 units
    if 0 in wide_store and 1 in wide_store:
        ref = merged_check([a.float() for a in wide_store[1]['acts']], [a.float() for a in wide_store[0]['acts']],
                           wide_store[0]['prof'], n_tr, seed=5)
        jdump(ref, f'{od}/wide_cross_seed.json')
        print('cross-seed wide', {k: (round(v, 3) if v is not None else None) for k, v in ref['summary'].items()},
              flush=True)
    print(f'analysis done {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
