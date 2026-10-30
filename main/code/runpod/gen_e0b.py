"""E0b counterfactual set on calibration/held start states only (official demos d00-d19 of each M2 task; E1 will use
d20-d49). Split: d00-d09 severity calibration, d10-d16 judge training, d17-d19 judge held-out.
Cells per demo: nominal + CureWM 6 families x severities 0.2..1.0 (dir 0) + the task's boundary-targeted family on a
dense severity grid (dir 0; theta drawn per demo). Physics-only rollouts, renders S0 and S_lift (b_cf.py).
Resumable (one JSON + NPZ per counterfactual). Run:  $VP /workspace/code/gen_e0b.py --workers 7
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/workspace/e0b/cf")
ap.add_argument("--workers", type=int, default=7)
ap.add_argument("--tasks", type=int, nargs="*", default=[1, 2, 4, 5, 6, 8, 0])
ap.add_argument("--demos", type=int, default=20)
args = ap.parse_args()

CUREWM = ["insufficient_grip", "premature_release", "carry_slip", "contact_oscillation", "wrist_tilt", "approach_overshoot"]
GRID = {  # boundary-targeted family per task and its dense calibration grid
    8: ("place_offset", np.linspace(0.01, 0.15, 21)),
    6: ("place_offset", np.linspace(0.005, 0.10, 20)),
    1: ("place_offset", np.linspace(0.03, 0.27, 21)),
    4: ("place_offset", np.linspace(0.05, 0.45, 21)),
    2: ("place_offset", np.linspace(0.05, 0.45, 21)),
    5: ("under_actuation", np.linspace(0.05, 0.90, 18)),
    0: ("pull_truncate", np.linspace(0.12, 0.30, 19)),
}
N_DEMOS_T0 = 10  # t0 is optional: half the demos


def cells_for(task):
    cells = [("nominal", 0.0, 0)]
    if task != 0:  # CureWM families never act on the drawer task (no grasp phase), see E0
        cells += [(f, s, 0) for f in CUREWM for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
    fam, grid = GRID[task]
    cells += [(fam, float(np.round(s, 4)), 0) for s in grid]
    return cells


def job(spec):
    task, di = spec
    os.makedirs(args.out, exist_ok=True)
    from b_cf import Generator, run_cell
    gen = Generator(task)
    t0 = time.time()
    n = 0
    for fam, s, k in cells_for(task):
        run_cell(gen, di, fam, s, k, args.out)
        n += 1
    gen.close()
    return f"t{task} d{di:02d} {n} cells {time.time() - t0:.0f}s"


if __name__ == "__main__":
    specs = [(t, d) for t in args.tasks for d in range(N_DEMOS_T0 if t == 0 else args.demos)]
    t_all = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for msg in ex.map(job, specs):
            print(msg, flush=True)
    print(f"GEN_DONE {len(specs)} groups {time.time() - t_all:.0f}s", flush=True)
