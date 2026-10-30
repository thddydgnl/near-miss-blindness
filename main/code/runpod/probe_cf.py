"""CureWM optimism probe of the released Cosmos-Policy-LIBERO checkpoint, on counterfactuals produced by the
unmodified CureWM engine (run_cf_generate.sh). Query construction follows CureWM scripts/probe_optimism.py exactly
(t0 = first action divergence - 1, obs = recorded frames[t0], 16-step chunk = actions[t0:t0+16], CureWM cfg),
via cp_common.cf_query, which reproduces their call sequence bit-for-bit (quickstart.py check).
Additions: survivors are probed too (CureWM's "fa" probes do this; needed for AUROC / false alarm), and the value
head is also applied to the REAL future state (judge (c) on real frames, cp_common.value_real_future).
Resumable: rows already in the output JSONL are skipped.
Run from the cosmos-policy checkout:  $VP /workspace/code/probe_cf.py --max_fail 400 --max_surv 250
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np

import cp_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/data/cf_goal")
ap.add_argument("--out", default="/workspace/results/curewm_probe/probe_released_goal.jsonl")
ap.add_argument("--max_fail", type=int, default=400)
ap.add_argument("--max_surv", type=int, default=250)
ap.add_argument("--n_act_steps", type=int, default=1, help="1 gives identical outputs to 5 (checked); 5 = CureWM default")
ap.add_argument("--check_act5", type=int, default=6, help="re-run the first K rows with 5 action steps and compare")
ap.add_argument("--no_real", action="store_true")
ap.add_argument("--only_tasks", nargs="*", default=None, help="e.g. t9")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)


def reorder_proprio(p9):
    """CureWM storage (eef_pos3, eef_quat4, grip2) -> official (grip2, eef_pos3, eef_quat4)."""
    return np.concatenate([p9[7:9], p9[0:3], p9[3:7]])


def to_libero_actions(a):
    a = a.copy()
    a[:, 6] = 1.0 - 2.0 * a[:, 6]  # grip [0 closed, 1 open] -> LIBERO [+1 closed, -1 open]
    return a


def first_divergence(a, b):
    n = min(len(a), len(b))
    diff = np.any(np.abs(a[:n] - b[:n]) > 1e-6, axis=1)
    return int(np.argmax(diff)) if diff.any() else 0


rows = []
for tdir in sorted(glob.glob(os.path.join(args.data, "t[0-9]"))):
    if args.only_tasks and os.path.basename(tdir) not in args.only_tasks:
        continue
    idx = os.path.join(tdir, "index.jsonl")
    if not os.path.exists(os.path.join(tdir, "sanity_report.json")):
        print("skip unfinished", tdir)
        continue
    for line in open(idx):
        r = json.loads(line)
        r["tdir"] = tdir
        r["tid"] = os.path.basename(tdir)
        rows.append(r)
fails = [r for r in rows if r["family"] != "nominal" and r["outcome"] is False]
survs = [r for r in rows if r["family"] != "nominal" and r["outcome"] is True]
rng = random.Random(0)
rng.shuffle(fails)
rng.shuffle(survs)
todo = fails[: args.max_fail] + survs[: args.max_surv]
print(f"[probe] rows {len(rows)} fails {len(fails)} survivors {len(survs)} -> probing {len(todo)}", flush=True)

done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["key"] for l in open(args.out) if l.strip()}
cfg = C.make_cfg("curewm")
model, stats = C.load_cosmos(cfg)
nominal_cache = {}
fo = open(args.out, "a")
n_new, t_start = 0, time.time()
for i, r in enumerate(todo):
    key = f"{r['tid']}/{r['name']}"
    if key in done:
        continue
    with np.load(os.path.join(r["tdir"], "episodes", f"{r['name']}.npz")) as z:
        p_act, p_fr, p_wr, p_pr = z["actions"], z["frames"], z["wrist_frames"], z["proprio"]
    with np.load(os.path.join(r["tdir"], "episodes", f"{r['pair_of']}.npz")) as z:
        n_act, n_fr = z["actions"], z["frames"]
    t0 = max(0, first_divergence(p_act, n_act) - 1)
    t0 = min(t0, len(p_act) - 17, len(n_act) - 17)
    if t0 < 0:
        continue
    obs = {"primary_image": p_fr[t0], "wrist_image": p_wr[t0], "proprio": reorder_proprio(p_pr[t0])}
    lang = r.get("language") or r.get("task_name", "").replace("_", " ")
    tq = time.time()
    pert = C.cf_query(cfg, model, stats, obs, lang, to_libero_actions(p_act[t0:t0 + 16]), seed=cfg.seed,
                      n_act_steps=args.n_act_steps)
    nk = (r["tid"], r["pair_of"], t0)
    if nk not in nominal_cache:
        nominal_cache[nk] = C.cf_query(cfg, model, stats, obs, lang, to_libero_actions(n_act[t0:t0 + 16]),
                                       seed=cfg.seed, n_act_steps=args.n_act_steps)["value"]
    t1 = min(t0 + 16, len(p_fr) - 1, len(n_fr) - 1)
    rec = {"key": key, "task": r["task"], "name": r["name"], "family": r["family"], "severity": r["severity"],
           "outcome": r["outcome"], "t0": t0, "T": int(len(p_act)), "T_nominal": int(len(n_act)),
           "value_fail_action": pert["value"], "value_nominal_action": nominal_cache[nk]}
    imag = pert["future_image"].astype(np.float32)
    gt_fail, gt_succ = p_fr[t1].astype(np.float32), n_fr[t1].astype(np.float32)
    if imag.shape == gt_fail.shape:
        d_fail, d_succ = float(np.mean((imag - gt_fail) ** 2)), float(np.mean((imag - gt_succ) ** 2))
        rec.update(mse_to_gt_failure=d_fail, mse_to_gt_success=d_succ, visually_optimistic=bool(d_succ < d_fail))
    if not args.no_real:
        fut = {"primary_image": p_fr[t1], "wrist_image": p_wr[t1], "proprio": reorder_proprio(p_pr[t1])}
        rec["value_real_future"] = C.value_real_future(cfg, model, stats, obs, lang, to_libero_actions(p_act[t0:t0 + 16]),
                                                       fut, seed=cfg.seed)
    if n_new < args.check_act5:
        rec["value_fail_action_act5"] = C.cf_query(cfg, model, stats, obs, lang, to_libero_actions(p_act[t0:t0 + 16]),
                                                   seed=cfg.seed, n_act_steps=5)["value"]
    rec["sec"] = time.time() - tq
    fo.write(json.dumps(rec) + "\n")
    fo.flush()
    n_new += 1
    if n_new % 20 == 0 or n_new <= 3:
        print(f"[probe] {n_new} new / {i + 1} of {len(todo)}  {key} V-={rec['value_fail_action']:.3f} "
              f"V+={rec['value_nominal_action']:.3f} Vreal={rec.get('value_real_future', float('nan')):.3f} "
              f"out={r['outcome']} {(time.time() - t_start) / n_new:.2f}s/row", flush=True)
print("PROBE_DONE", n_new, "new rows", flush=True)
