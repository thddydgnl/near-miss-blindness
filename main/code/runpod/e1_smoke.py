"""E1 smoke test (M2): end-of-episode imagination of k = 1..4 chunks on recorded counterfactual episodes with known
final margin r (test_margin.py), with correct obs/action alignment (obs before action t_start = frames[t_start-1],
chunks = actions[t_start : T], t_start = T - 16k), so the k-th imagined frame corresponds to the real final frame.
Per (episode, k): V(imagined final state) [judge (c), imagined], V(real final state) [judge (c), real, k = 1 only],
MSE(imagined final primary frame, real final frame in model space). Exploratory pipeline check only (E0).
Run from the cosmos-policy checkout:  $VP /workspace/code/e1_smoke.py
"""
import json
import os
import random
import time

import numpy as np

import cp_common as C

TASKS = [1, 2, 4, 5, 6, 8]
OUT = "/workspace/results/e1_smoke.jsonl"
margins = [json.loads(l) for l in open("/workspace/results/margin/cf_margins.jsonl")]
rng = random.Random(0)
sel = []
for t in TASKS:
    rs = [m for m in margins if m["task"] == t and m["T"] >= 70]
    rng.shuffle(rs)
    succ = [m for m in rs if m["success_libero_final"]][:2]
    near = [m for m in rs if not m["success_libero_final"] and 1 <= m["r_final"] < 2][:2]
    far = [m for m in rs if not m["success_libero_final"] and m["r_final"] >= 2][:1]
    sel += succ + near + far
print("episodes", len(sel), flush=True)


def to_libero(a):
    a = a.copy()
    a[:, 6] = 1.0 - 2.0 * a[:, 6]
    return a


def reorder(p9):
    return np.concatenate([p9[7:9], p9[0:3], p9[3:7]])


cfg = C.make_cfg("curewm")
model, stats = C.load_cosmos(cfg)
lang_of = {}
for t in TASKS:
    for l in open(f"/workspace/data/cf_goal/t{t}/index.jsonl"):
        r = json.loads(l)
        lang_of[f"t{t}/{r['name']}"] = r["language"]
fo = open(OUT, "w")
t_all = time.time()
for m in sel:
    tdir = f"/workspace/data/cf_goal/t{m['task']}"
    name = m["key"].split("/")[1]
    with np.load(f"{tdir}/episodes/{name}.npz") as z:
        act, fr, wr, pr = z["actions"], z["frames"], z["wrist_frames"], z["proprio"]
    T = len(act)
    real_final = C.CU.prepare_images_for_model([fr[T - 1]], cfg)[0].astype(np.float32)
    lang = lang_of[m["key"]]
    for k in [1, 2, 3, 4]:
        ts = T - 16 * k
        if ts < 1:
            continue
        obs = {"primary_image": fr[ts - 1], "wrist_image": wr[ts - 1], "proprio": reorder(pr[ts - 1])}
        chunks = [to_libero(act[ts + 16 * i: ts + 16 * (i + 1)]) for i in range(k)]
        tq = time.time()
        outs = C.imagine_chain(cfg, model, stats, obs, lang, chunks, seed=cfg.seed, n_act_steps=1)
        rec = dict(key=m["key"], task=m["task"], family=m["family"], severity=m["severity"], r_final=m["r_final"],
                   success=m["success_libero_final"], T=T, k=k, v_imag=outs[-1]["value"],
                   v_imag_per_chunk=[o["value"] for o in outs],
                   mse_imag_vs_real_final=float(np.mean((outs[-1]["future_image"].astype(np.float32) - real_final) ** 2)),
                   sec=time.time() - tq)
        if k == 1:
            fut = {"primary_image": fr[T - 1], "wrist_image": wr[T - 1], "proprio": reorder(pr[T - 1])}
            rec["v_real"] = C.value_real_future(cfg, model, stats, obs, lang, chunks[0], fut, seed=cfg.seed)
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
    print(m["key"], "r=%.2f" % m["r_final"], "done", flush=True)
print("E1_SMOKE_DONE %.0fs" % (time.time() - t_all), flush=True)
