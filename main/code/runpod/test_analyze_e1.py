"""Synthetic imagine records for a smoke test of analyze_e1.py (no real data). Usage: python test_analyze_e1.py out.jsonl smalldir"""
import json
import os
import sys

import numpy as np

rng = np.random.default_rng(0)
out, small = sys.argv[1], sys.argv[2]
os.makedirs(small, exist_ok=True)
with open(out, "w") as f:
    for task in (0, 1, 2, 4, 5, 6, 8):
        for d in range(20, 50):
            for i in range(20):
                r = float(np.exp(rng.uniform(np.log(0.1), np.log(6))))
                succ = r < 1 and rng.random() > 0.02
                lr = np.log(r)
                rec = dict(job_id=f"im_t{task}_d{d:02d}_x_s{i:.4f}_k0", dep=f"cf_t{task}_d{d:02d}_x_s{i:.4f}_k0", task=task,
                           family="x", severity=0.1, r_settled=r, success_settled=bool(succ), success_ever=bool(succ), T=120,
                           a_logit_real=None if task == 0 else float(-6 * lr + rng.normal(0, 1)),
                           d_real=float(1 / (1 + np.exp(2 * lr + rng.normal(0, 1)))), v_real_k1=float(1 / (1 + np.exp(3 * lr - 1))))
                for k in (1, 2, 3, 4):
                    blur = 1 + 0.5 * k
                    rec[f"k{k}"] = dict(v_end=float(1 / (1 + np.exp(3 * lr / blur - 1 + rng.normal(0, 0.5)))), v_lift=0.5,
                                        a_logit=None if task == 0 else float(-6 * lr / blur + 1 + rng.normal(0, 1.5)),
                                        d=float(1 / (1 + np.exp(2 * lr / blur + rng.normal(0, 1)))), sec=13.0)
                f.write(json.dumps(rec) + "\n")
                feats = {"feat_real": rng.normal(size=3072).astype(np.float16)}
                feats["feat_real"][0] = -lr
                for k in (1, 2, 3, 4):
                    feats[f"feat_imag_k{k}"] = rng.normal(size=3072).astype(np.float16)
                    feats[f"feat_imag_k{k}"][0] = -lr / (1 + 0.5 * k)
                np.savez(os.path.join(small, rec["job_id"] + ".npz"), **feats)
print("synthetic", out)
