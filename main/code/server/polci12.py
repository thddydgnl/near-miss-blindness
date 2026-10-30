"""Post-hoc: per-policy over-estimation (imagined - true) with cluster-bootstrap 95% CIs (main set), judges a, b, ai.
usage: python polci12.py"""
import sys, json
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from analyze12 import load, flags, POL, RES

J = json.load(open(f'{RES}/judges.json'))
recs = load('main')
F = flags(recs, J)
g = lambda k: np.array([x[k] for x in recs])
pol, succ, traj = g('policy'), g('success').astype(float), g('traj')
ut = np.unique(traj); tix = np.searchsorted(ut, traj)
rng = np.random.RandomState(4242)
W = [np.ones(len(recs))] + [np.bincount(rng.randint(0, len(ut), len(ut)), minlength=len(ut)).astype(float)[tix] for _ in range(1000)]
out = {}
for j, key in (('a', 'a_imag'), ('b', 'b_imag'), ('ai', 'ai_imag'), ('a_realjudge', 'a_real'), ('b_realjudge', 'b_real')):
    fl = F[key].astype(float)
    out[j] = {}
    for p in POL + ['expert']:
        mk = pol == p
        v = [float(((fl[mk] - succ[mk]) * w[mk]).sum() / w[mk].sum()) for w in W]
        out[j][p] = dict(over=v[0], ci=[float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))])
    print(j, {p: f"{100*d['over']:+.1f} [{100*d['ci'][0]:+.1f},{100*d['ci'][1]:+.1f}]" for p, d in out[j].items()})
json.dump(out, open(f'{RES}/policy_ci_main.json', 'w'), indent=1)
print('POLCI_DONE')
