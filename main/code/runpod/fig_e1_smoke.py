"""Small figure for the E0 report: real final frame vs imagined final frame after k = 1..4 chunks (end-aligned, as in
e1_smoke.py) for a few episodes of the E1 smoke set (one success and near-misses). Output: results/e1_smoke_frames.png
Run from the cosmos-policy checkout:  $VP /workspace/code/fig_e1_smoke.py
"""
import json

import numpy as np
from PIL import Image, ImageDraw

import cp_common as C

rows = [json.loads(l) for l in open("/workspace/results/e1_smoke.jsonl")]
keys = []
for want in ["succ", "near", "near", "far"]:
    for r in rows:
        if r["k"] != 1 or r["key"] in keys:
            continue
        ok = (want == "succ" and r["success"]) or (want == "near" and not r["success"] and 1 <= r["r_final"] < 2) or \
             (want == "far" and not r["success"] and r["r_final"] >= 2)
        if ok and r["task"] in (8, 1, 6):
            keys.append(r["key"])
            break
meta = {r["key"]: r for r in rows if r["k"] == 1}
cfg = C.make_cfg("curewm")
model, stats = C.load_cosmos(cfg)


def to_libero(a):
    a = a.copy()
    a[:, 6] = 1.0 - 2.0 * a[:, 6]
    return a


def reorder(p9):
    return np.concatenate([p9[7:9], p9[0:3], p9[3:7]])


tiles = []
for key in keys:
    t, name = key.split("/")
    for l in open(f"/workspace/data/cf_goal/{t}/index.jsonl"):
        r = json.loads(l)
        if r["name"] == name:
            lang = r["language"]
    with np.load(f"/workspace/data/cf_goal/{t}/episodes/{name}.npz") as z:
        act, fr, wr, pr = z["actions"], z["frames"], z["wrist_frames"], z["proprio"]
    T = len(act)
    row = [C.CU.prepare_images_for_model([fr[T - 1]], cfg)[0]]
    vals = []
    for k in [1, 2, 3, 4]:
        ts = T - 16 * k
        obs = {"primary_image": fr[ts - 1], "wrist_image": wr[ts - 1], "proprio": reorder(pr[ts - 1])}
        outs = C.imagine_chain(cfg, model, stats, obs, lang, [to_libero(act[ts + 16 * i: ts + 16 * (i + 1)]) for i in range(k)],
                               seed=cfg.seed, n_act_steps=1)
        row.append(outs[-1]["future_image"])
        vals.append(outs[-1]["value"])
    img = Image.fromarray(np.concatenate(row, axis=1))
    d = ImageDraw.Draw(img)
    d.text((4, 4), f"{key} r={meta[key]['r_final']:.2f} REAL", fill=(255, 0, 0))
    for k, v in enumerate(vals, 1):
        d.text((224 * k + 4, 4), f"imag k={k} V={v:.2f}", fill=(255, 0, 0))
    tiles.append(np.asarray(img))
Image.fromarray(np.concatenate(tiles, axis=0)).resize((5 * 160, len(tiles) * 160)).save("/workspace/results/e1_smoke_frames.png")
print("saved", keys)
