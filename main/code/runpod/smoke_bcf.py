"""Smoke test of b_cf.py: a few counterfactual cells, timing, and a contact sheet of S0 / S_lift renders."""
import json
import time

import numpy as np
from PIL import Image

from b_cf import Generator, run_cell

out = "/workspace/e0b/smoke"
import os
os.makedirs(out, exist_ok=True)
rows, sheets = [], []
for task, di, cells in [(8, 1, [("nominal", 0.0), ("place_offset", 0.05), ("premature_release", 0.6)]),
                        (5, 0, [("nominal", 0.0), ("under_actuation", 0.4)]),
                        (2, 0, [("nominal", 0.0), ("place_offset", 0.25)])]:
    gen = Generator(task)
    for fam, s in cells:
        t0 = time.time()
        rec = run_cell(gen, di, fam, s, 0, out)
        rec["sec"] = time.time() - t0
        rows.append(rec)
        print(json.dumps({k: rec.get(k) for k in ["cf_id", "T", "success_ever", "r_settled", "success_settled", "r_lift",
                                                   "success_lift", "lift_changes_outcome", "sec", "skipped"]}), flush=True)
        if "skipped" not in rec:
            z = np.load(os.path.join(out, rec["cf_id"] + ".npz"))
            sheets.append(np.concatenate([z["S0_primary"], z["Slift_primary"], z["Slift_wrist"]], axis=1))
    gen.close()
Image.fromarray(np.concatenate(sheets, axis=0)).save("/workspace/results/e0b_smoke_sheet.png")
print("SMOKE_DONE")
