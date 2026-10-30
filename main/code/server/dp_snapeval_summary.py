"""B E0: collect the sim-only snapshot evaluations (dp_rollout.py summaries, calibration seeds 200000-200049; no world
model involved) into one JSON, for the DP training-stage choice.
usage: python dp_snapeval_summary.py"""
import glob, json, os, re, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP

D = f'{BP.RES}/snapeval'
rows = []
for f in sorted(glob.glob(f'{D}/s*_ep*.pt')):
    m = re.search(r's(\d+)_ep(\d+)\.pt$', f)
    d = torch.load(f, weights_only=False)
    s = d['summary']
    eps = d['episodes']
    ro = np.array([e['r_obj'] for e in eps])
    fc = np.array([e['final_cov'] for e in eps])
    rows.append(dict(seed=int(m.group(1)), epoch=int(m.group(2)), n=s['n'], success_robj=s['success_robj'],
                     near_robj=s['near_robj'], mid_robj=s['mid_robj'], far_robj=s['far_robj'], success_dp=s['success_dp'],
                     mean_score=s['mean_score'], timeout=s['timeout'], r_obj_median=float(np.median(ro)),
                     final_cov_median=float(np.median(fc)), sec=s['sec'],
                     near_composition_block=float(np.mean([(e['r_obj_pos'] >= e['r_ang']) for e in eps if 1 <= e['r_obj'] < 2]))
                     if any(1 <= e['r_obj'] < 2 for e in eps) else None))
rows.sort(key=lambda r: (r['seed'], r['epoch']))
json.dump(rows, open(f'{BP.RES}/dp_snapeval_summary.json', 'w'), indent=1)
for r in rows:
    print(f"s{r['seed']} ep{r['epoch']:4d} succ_robj {r['success_robj']:.2f} near {r['near_robj']:.2f} mid {r['mid_robj']:.2f} "
          f"far {r['far_robj']:.2f} | succ_dp {r['success_dp']:.2f} score {r['mean_score']:.2f} timeout {r['timeout']:.2f} "
          f"| med r_obj {r['r_obj_median']:.2f} | {r['sec']:.0f}s")
