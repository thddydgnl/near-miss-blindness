"""사후 분석 (after seeing the main results; not used for the verdict).
(a) same-k ablation: mean-ablate the top-k failure-swap units (k = 4, 8, 16, by each model's own importance) in the
    1x and 4x std p=0.01 models and compare miss-only optimism; layer-matched random-k control (10 draws).
    If consolidation means "a few units carry the failure prediction", removing the same k units should hurt the
    4x model more than the 1x model.
(b) slip-swap n50 for the visual 1x/4x p=0.01 models (the visual 4x optimism drop came mostly from the slip mode)."""
import json
import os

import numpy as np
import torch

from analyze import (R_obj, ablate_eval, act_stats, concentration, load_tests, patch_importance, train_inputs)
from common import DEPTH, WIDTH, nx, slice_model

RES, CK = '/workspace/results', '/workspace/ckpt'
PAIRS = [('state Nf1k', 's_n100k_T1', 's_n100k_T4'), ('state Nf3k', 's_n300k_T1', 's_n300k_T4'),
         ('state Nf200', 's_n20k_T1', 's_n20k_T4'), ('visual Nf1k', 'v_T1_std', 'v_T4_std')]
KS = [4, 8, 16]


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    out = []
    tests = {}
    with torch.no_grad():
        for label, r1, r4 in PAIRS:
            for run in (r1, r4):
                ck = torch.load(os.path.join(CK, f'{run}.pt'), map_location=dev)
                nm = {k: v.to(dev) for k, v in ck['norm'].items()}
                ver = ck['cfgs'][0]['version']
                if ver not in tests:
                    tests[ver] = load_tests(ver, dev)
                TS = tests[ver]
                A = {r['model']: r for r in json.load(open(os.path.join(RES, run, 'analysis.json')))}
                X = nx(TS['bal']['X'], nm)
                for m, c in enumerate(ck['cfgs']):
                    if c['mode'] != 'std' or abs(c['p'] - 0.01) > 1e-9:
                        continue
                    Pm = slice_model({k: v.to(dev) for k, v in ck['P'].items()}, m)
                    Xtr = nx(train_inputs(c, 10000, dev), nm)
                    mj, mean_act = act_stats(Pm, Xtr)
                    I = torch.tensor(A[m]['importance_fail'], device=dev)
                    row = dict(setting=label, run=run, T=c['T'], seed=c['seed'])
                    base = ablate_eval(Pm, X, TS['bal'], nm, [torch.empty(0, dtype=torch.long, device=dev)] * DEPTH, mean_act)
                    row['base_opt_miss'], row['base_opt'] = base['opt_miss'], base['opt']
                    for k in KS:
                        order = torch.argsort(I, descending=True)[:k]
                        det = [order[order // WIDTH == l] % WIDTH for l in range(DEPTH)]
                        d = ablate_eval(Pm, X, TS['bal'], nm, det, mean_act)
                        rr = []
                        for _ in range(10):
                            rs = []
                            for l in range(DEPTH):
                                pool = torch.ones(WIDTH, dtype=torch.bool, device=dev)
                                pool[det[l]] = False
                                cand = pool.nonzero().squeeze(1)
                                rs.append(cand[torch.randperm(len(cand), generator=gen, device=dev)[:len(det[l])]])
                            rr.append(ablate_eval(Pm, X, TS['bal'], nm, rs, mean_act)['opt_miss'])
                        row[f'top{k}_opt_miss'] = d['opt_miss']
                        row[f'rand{k}_opt_miss'] = float(np.mean(rr))
                    if ver == 'visual':
                        ps = patch_importance(Pm, nm, *TS['pairs']['slip'], R_obj)
                        row['slip_n50'] = concentration(ps['I'])
                    out.append(row)
                    print(row, flush=True)
    json.dump(out, open(os.path.join(RES, 'posthoc_topk_ablation.json'), 'w'), indent=1)
    # seed means
    summ = []
    for label, r1, r4 in PAIRS:
        for run in (r1, r4):
            rs = [r for r in out if r['run'] == run]
            s = dict(setting=label, run=run, T=rs[0]['T'])
            for key in rs[0]:
                if key not in ('setting', 'run', 'T', 'seed'):
                    s[key] = float(np.mean([r[key] for r in rs]))
            summ.append(s)
    json.dump(summ, open(os.path.join(RES, 'posthoc_topk_ablation_summary.json'), 'w'), indent=1)
    for s in summ:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == '__main__':
    main()
