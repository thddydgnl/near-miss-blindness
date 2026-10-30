"""Quick evaluation of DP checkpoints on HELD start states (E0b split: env.reset() placements with seeds
10000..10019, saved to /workspace/dp/held_init/t<task>.npy; never the LIBERO official init states 0-49 that E2 uses).
Episode loop as run_libero_eval: 10 no-op steps, then 16-step open-loop chunks from the policy, LIBERO success
terminates the episode (max 300 steps). After the episode: 15 settle steps (zero motion, gripper open) and
r = e / tau (libero_margin.task_margin). Writes one JSON line per episode.
Run (cwd = cosmos-policy checkout):  $VP /workspace/code/dp_eval.py --task 8 --episodes 20
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import cp_common as C
import libero_margin as M
from dp_model import PolicyRunner

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, required=True)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seed_base", type=int, default=10000)
ap.add_argument("--ckpt_dir", default="/workspace/dp/ckpt")
ap.add_argument("--out", default="/workspace/results/dp_eval")
ap.add_argument("--max_steps", type=int, default=300)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs("/workspace/dp/held_init", exist_ok=True)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

b = benchmark.get_benchmark_dict()["libero_goal"]()
task = b.get_task(args.task)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                         camera_heights=256, camera_widths=256)
held_path = f"/workspace/dp/held_init/t{args.task}.npy"
if os.path.exists(held_path):
    held = np.load(held_path)
else:
    held = []
    for i in range(args.episodes):
        np.random.seed(args.seed_base + i)
        env.seed(args.seed_base + i)
        env.reset()
        held.append(env.get_sim_state().copy())
    held = np.stack(held)
    np.save(held_path, held)
cfg = C.make_cfg("eval")
out_f = os.path.join(args.out, f"t{args.task}.jsonl")
done = set()
if os.path.exists(out_f):
    done = {(json.loads(l)["ckpt"], json.loads(l)["episode"]) for l in open(out_f) if l.strip()}
fo = open(out_f, "a")
for ck in sorted(glob.glob(os.path.join(args.ckpt_dir, f"t{args.task}_s*", "ckpt_*.pt")),
                 key=lambda p: int(p.split("_")[-1].split(".")[0])):
    name = os.path.relpath(ck, args.ckpt_dir)
    pol = None
    for ep in range(min(args.episodes, len(held))):
        if (name, ep) in done:
            continue
        if pol is None:
            pol = PolicyRunner(ck, seed=ep)
        t0 = time.time()
        env.reset()
        obs = env.set_init_state(held[ep])
        t, success, n_q = 0, False, 0
        while t < args.max_steps + 10:
            if t < 10:
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                t += 1
                continue
            o = C.env_obs_to_policy_obs(obs, flip=True)
            p224 = C.CU.prepare_images_for_model([o["primary_image"], o["wrist_image"]], cfg)
            chunk = pol.act(p224[0], p224[1], o["proprio"])
            n_q += 1
            for a in chunk:
                obs, _, done_flag, _ = env.step(np.asarray(a, dtype=np.float64).tolist())
                t += 1
                if done_flag:
                    success = True
                    break
                if t >= args.max_steps + 10:
                    break
            if success:
                break
        for _ in range(15):
            env.step([0, 0, 0, 0, 0, 0, -1])
        m = M.task_margin(env)
        rec = dict(task=args.task, ckpt=name, step=int(name.split("_")[-1].split(".")[0]), episode=ep, success=success,
                   steps=t, queries=n_q, r_settled=m["r"], success_settled=m["success_libero"], aux_ok=m["aux_ok"],
                   sec=time.time() - t0)
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
    print("done", name, flush=True)
env.close()
print("EVAL_DONE", args.task, flush=True)
