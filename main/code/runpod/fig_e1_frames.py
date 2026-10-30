"""Diagnostic figure for the E1 report (no new data): real vs imagined post-lift frames (k = 1) for a few E1 items of
this pod's shard, one success and one near-miss per task, with the judge (a) logits. Re-runs the k = 1 chain (last real
chunk + lift) exactly as b_runner.do_imagine (seed 195, 1-step action pass).
Run from the cosmos-policy checkout: $VP /workspace/code/fig_e1_frames.py --out /workspace/results/e1_frames.png
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

import b_judges as J
import cp_common as C
from b_cf import LIFT

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/runs/e1")
ap.add_argument("--out", default="/workspace/results/e1_frames.png")
ap.add_argument("--tasks", type=int, nargs="*", default=[1, 2, 4, 5, 6, 8])
args = ap.parse_args()
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.root, "results/imagine/*.json")))]
pick = []
for t in args.tasks:
    rt = [r for r in recs if r["task"] == t]
    s = [r for r in rt if r["success_settled"] and r["r_settled"] < 0.8]
    n = [r for r in rt if not r["success_settled"] and 1 <= r["r_settled"] < 2]
    pick += s[:1] + n[:1]
cfg = C.make_cfg("eval")
model, stats = C.load_cosmos(cfg)
tiles = []
for r in pick:
    z = np.load(os.path.join(args.root, "art", r["dep"] + ".npz"))
    a = z["actions"]
    T = len(a)
    obs = {"primary_image": z["Sk1_primary"], "wrist_image": z["Sk1_wrist"], "proprio": z["Sk1_proprio"]}
    lang = json.load(open(os.path.join(args.root, "results/cf", r["dep"] + ".json")))["language"]
    outs = C.imagine_chain(cfg, model, stats, obs, lang, [a[T - 16:T], LIFT], seed=cfg.seed, n_act_steps=1, pre_first=True)
    im_p, im_w = outs[-1]["future_image"], outs[-1]["future_wrist_image"]
    a_im = J.judge_a_cnn(r["task"], im_p, im_w)
    row = np.concatenate([z["Slift_primary"], im_p, z["Slift_wrist"], im_w], axis=1)
    img = Image.fromarray(row)
    d = ImageDraw.Draw(img)
    d.text((4, 4), f"t{r['task']} r={r['r_settled']:.2f} {'SUCC' if r['success_settled'] else 'FAIL'} REAL a={r['a_logit_real']:.1f}", fill=(255, 0, 0))
    d.text((228, 4), f"IMAG k=1 a={a_im:.1f} (job {r['k1']['a_logit']:.1f})", fill=(255, 0, 0))
    d.text((452, 4), "REAL wrist", fill=(255, 0, 0))
    d.text((676, 4), "IMAG wrist", fill=(255, 0, 0))
    tiles.append(np.asarray(img))
    print(r["job_id"], r["r_settled"], r["success_settled"], r["a_logit_real"], r["k1"]["a_logit"], a_im, flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
sheet = Image.fromarray(np.concatenate(tiles, axis=0))
sheet.resize((sheet.width * 3 // 4, sheet.height * 3 // 4)).save(args.out)
print("saved", args.out, len(tiles))
