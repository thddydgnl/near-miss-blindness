"""B E4/E5 analysis helpers (CPU analysis env): seed-pooled imagined judgments on the M1 E1 cells and the M3 E1 grid,
NM / SM and paired cluster-bootstrap differences (DESIGN_main v3.1 section 9)."""
import os, json, glob
import numpy as np

E1M1 = os.path.expanduser('~/pmx/B/results/E1/m1')
E1M3 = os.path.expanduser('~/pmx/B/results/E1/m3')
E4M1 = os.path.expanduser('~/pmx/B/results/E4/m1')
E5M3 = os.path.expanduser('~/pmx/B/results/E5/m3')
AKEY = {'r': 'p_full', 'r_obj': 'p_obj'}
B_BOOT = 10000


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None


def boot_w(n_units, B, seed):
    rng = np.random.RandomState(seed)
    W = np.zeros((B, n_units))
    np.add.at(W, (np.arange(B)[:, None], rng.randint(0, n_units, (B, n_units))), 1.0)
    return W


class M1Cell:
    """one E1 cell: counterfactual records of the cell's own grid, bands of the cell's definition, cluster = source
    trajectory. judged(variant) = seed-pooled imagined judged-success fraction per record (judge (a) of the definition)."""

    def __init__(self, tag):
        self.tag = tag
        st, H, d = tag.split('_', 2)
        self.d = d
        recs = [json.loads(l) for l in open(f'{E1M1}/eval_{tag}.jsonl')]
        self.cf = np.array([r['kind'] == 'cf' for r in recs])
        x = np.array([r[d] for r in recs])[self.cf]
        self.band = np.where(x < 1, 'success', np.where(x < 2, 'near', np.where(x < 3, 'mid', 'far')))
        tr = np.array([r['traj'] for r in recs])[self.cf]
        ut = np.unique(tr)
        self.unit = np.searchsorted(ut, tr)
        self.n_units = len(ut)
        self.real = (np.array([r[f'{AKEY[d]}_real'] for r in recs]) > 0.5)[self.cf].astype(float)
        self.e1_imag = (np.array([r[f'{AKEY[d]}_imag'] for r in recs]) > 0.5)[self.cf].astype(float)

    def judged(self, variant, seeds=(0, 1, 2)):
        names = ['orig'] if variant == 'orig' else [f'{variant}_s{s}' for s in seeds]
        J = []
        for n in names:
            z = np.load(f'{E4M1}/imag_{self.tag}_{n}.npz')
            J.append((z[f'{AKEY[self.d]}_imag'] > 0.5)[self.cf].astype(float))
        return np.mean(J, 0)


class M3Grid:
    """M3 E1 grid (seeds pooled): points (grid r x scene) used, bands, cluster = scene."""

    def __init__(self):
        z = np.load(f'{E1M3}/e1_p0.03_std_1x_s0.npz')
        self.use = z['use']
        g, s = np.where(self.use)
        self.g, self.s = g, s
        r = z['grid_r'][g]
        self.band = np.where(r < 1, 'success', np.where(r < 2, 'near', np.where(r < 3, 'mid', 'far')))
        self.unit = s
        self.n_units = int(s.max()) + 1
        self.real = (z['real_logit'][g, s] > 0).astype(float)

    def judged_e1(self, p):
        return np.mean([(np.load(f'{E1M3}/e1_p{p}_std_1x_s{k}.npz')['imag_logit'][self.g, self.s] > 0) for k in (0, 1, 2)], 0).astype(float)

    def judged_e5(self, name_fmt):
        return np.mean([(np.load(f'{E5M3}/e5eval_{name_fmt.format(s=k)}.npz')['grid_imag_logit'][self.g, self.s] > 0) for k in (0, 1, 2)], 0).astype(float)


def nm_sm(J, band, unit, W):
    """point NM / SM and bootstrap rows (W: (B, n_units) multiplicities)."""
    nm, sm = band == 'near', band == 'success'
    pt = dict(NM=float(J[nm].mean()), SM=float(1 - J[sm].mean()))
    Wn, Ws = W[:, unit[nm]], W[:, unit[sm]]
    bs = dict(NM=(Wn @ J[nm]) / Wn.sum(1), SM=1 - (Ws @ J[sm]) / Ws.sum(1))
    return pt, bs


def paired(JA, JB, band, unit, W):
    """A - B differences of NM and SM with 95% CI and one-sided bootstrap p-values."""
    pa, ba = nm_sm(JA, band, unit, W)
    pb, bb = nm_sm(JB, band, unit, W)
    out = {}
    for k in ('NM', 'SM'):
        d = ba[k] - bb[k]
        out[k] = dict(A=pa[k], B=pb[k], diff=pa[k] - pb[k], ci=ci(d), p_le0=float(np.mean(d <= 0)), p_ge0=float(np.mean(d >= 0)))
    return out
