"""B E2 policy-level statistics (DESIGN_main v1 sections 4.3 / 6.2, v2.2 section 7), vectorised over bootstrap rows.
All functions take policy-level vectors of shape (B, N) (B bootstrap rows, N policies): s = real success rate,
sh = imagined (judged) success rate, n = near-miss share.
  spearman / pearson per row (Spearman = Pearson on average ranks)
  MMRV (SIMPLER, Li et al. 2024) = (1/N) sum_i max_j |s_i - s_j| * 1[(sh_i < sh_j) != (s_i < s_j)]
  FGRE (v2.2)                    = (1/N) sum_i max_{j: |s_i - s_j| <= 0.05} |n_i - n_j| * 1[(sh_i < sh_j) != (s_i < s_j)]
  H2b: pairs with |s_i - s_j| <= 0.05 and |sh_i - sh_j| >= 0.15; Kendall tau_b over the pairs with |s_i - s_j| <= 0.05,
       tau = (C - D) / sqrt((C + D + T_s)(C + D + T_sh)) (pairs tied in both are left out);
       sensitivity: tau_b inside the 5 pp window of real success that holds the most policies.
Tolerance 1e-9 on the 0.05 / 0.15 comparisons (rates are multiples of 1/n_episodes)."""
import itertools
import numpy as np
from scipy.stats import rankdata, kendalltau

EPS = 1e-9
DELTA = 0.05
GAP = 0.15


def _rowcorr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (x * y).sum(1) / den, np.nan)


def spearman(x, y):
    return _rowcorr(rankdata(x, axis=1), rankdata(y, axis=1))


def pearson(x, y):
    return _rowcorr(np.asarray(x, float), np.asarray(y, float))


def _viol(s, sh):
    return (sh[:, :, None] < sh[:, None, :]) != (s[:, :, None] < s[:, None, :])


def mmrv(s, sh):
    return (np.abs(s[:, :, None] - s[:, None, :]) * _viol(s, sh)).max(2).mean(1)


def fgre(s, sh, n):
    sim = np.abs(s[:, :, None] - s[:, None, :]) <= DELTA + EPS
    return (np.abs(n[:, :, None] - n[:, None, :]) * _viol(s, sh) * sim).max(2).mean(1)


def _pairs(N):
    i, j = np.triu_indices(N, 1)
    return i, j


def tau_sim(s, sh):
    """Kendall tau_b over the pairs with |s_i - s_j| <= 0.05, per row; also the number of such pairs."""
    i, j = _pairs(s.shape[1])
    ds, dh = s[:, i] - s[:, j], sh[:, i] - sh[:, j]
    sim = np.abs(ds) <= DELTA + EPS
    sgs, sgh = np.sign(np.where(np.abs(ds) <= EPS, 0, ds)), np.sign(np.where(np.abs(dh) <= EPS, 0, dh))
    C = ((sgs * sgh > 0) & sim).sum(1)
    D = ((sgs * sgh < 0) & sim).sum(1)
    Ts = ((sgs == 0) & (sgh != 0) & sim).sum(1)
    Th = ((sgs != 0) & (sgh == 0) & sim).sum(1)
    den = np.sqrt((C + D + Ts).astype(float) * (C + D + Th))
    with np.errstate(invalid='ignore', divide='ignore'):
        tau = np.where(den > 0, (C - D) / den, np.nan)
    return tau, sim.sum(1)


def h2b_pairs(s, sh):
    """per row: number of qualifying pairs (|ds| <= 0.05 and |dsh| >= 0.15) and the largest imagined gap among the
    similar pairs."""
    i, j = _pairs(s.shape[1])
    ds, dh = np.abs(s[:, i] - s[:, j]), np.abs(sh[:, i] - sh[:, j])
    sim = ds <= DELTA + EPS
    q = (sim & (dh >= GAP - EPS)).sum(1)
    mx = np.where(sim, dh, -np.inf).max(1)
    return q, np.where(np.isfinite(mx), mx, np.nan)


def densest_window(s1):
    """indices of the policies inside the 5 pp window of real success holding the most policies (first such window by
    increasing s) and the number of windows that tie for the maximum."""
    o = np.argsort(s1, kind='stable')
    ss = s1[o]
    best, cnt, start = -1, 0, 0
    sizes = []
    for a in range(len(ss)):
        b = np.searchsorted(ss, ss[a] + DELTA + EPS, 'right')
        sizes.append(b - a)
    sizes = np.array(sizes)
    mx = sizes.max()
    a = int(np.argmax(sizes))
    return o[a:a + mx], int((sizes == mx).sum())


def tau_window(s1, sh1):
    idx, nties = densest_window(s1)
    if len(idx) < 2:
        return np.nan, idx, nties
    t = kendalltau(sh1[idx], s1[idx]).correlation       # tau_b
    return float(t) if np.isfinite(t) else np.nan, idx, nties


def perm_spearman_p(x, y, n_perm=10000, seed=0, exact_max=8):
    """one-sided permutation p-value for H0: rho <= 0 (policy-level permutation of x). Exact for N <= exact_max."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = spearman(x[None], y[None])[0]
    N = len(x)
    if N <= exact_max:
        perms = np.array(list(itertools.permutations(range(N))))
        r = spearman(x[perms], np.repeat(y[None], len(perms), 0))
        return float(obs), float(np.mean(r >= obs - 1e-12)), 'exact'
    rng = np.random.RandomState(seed)
    P = np.stack([rng.permutation(N) for _ in range(n_perm)])
    r = spearman(x[P], np.repeat(y[None], n_perm, 0))
    return float(obs), float((1 + np.sum(r >= obs - 1e-12)) / (1 + n_perm)), 'mc'


def policy_stats(s, sh, shr, n):
    """all statistics for a set of rows; s, sh (imagined judge), shr (real-frame judge), n: (B, N)."""
    o, oi, oj = sh - s, sh - shr, shr - s
    q, mx = h2b_pairs(s, sh)
    ts, npair = tau_sim(s, sh)
    tr, _ = tau_sim(s, shr)
    return dict(spearman_n_oimag=spearman(n, oi), spearman_n_o=spearman(n, o), spearman_n_ojudge=spearman(n, oj),
                pearson_n_oimag=pearson(n, oi), pearson_sh_s=pearson(sh, s), spearman_sh_s=spearman(sh, s),
                pearson_shreal_s=pearson(shr, s), mmrv=mmrv(s, sh), mmrv_realjudge=mmrv(s, shr), fgre=fgre(s, sh, n),
                fgre_realjudge=fgre(s, shr, n), h2b_n_pairs=q.astype(float), h2b_max_gap_similar=mx, tau_similar=ts,
                tau_similar_realjudge=tr, n_similar_pairs=npair.astype(float), mean_o=o.mean(1), mean_abs_o=np.abs(o).mean(1),
                mean_oimag=oi.mean(1), mean_ojudge=oj.mean(1))
