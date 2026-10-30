"""Closed-loop Cosmos Policy rollouts on one LIBERO task (same loop as run_libero_eval.run_episode, serial,
no best-of-N): checks the native install end to end and records timing, success, the per-chunk predicted values,
and predicted vs real next-chunk frames.
Run from the cosmos-policy checkout:  $VP /workspace/code/rollout_check.py --suite libero_goal --task 8 --episodes 5
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from libero.libero import benchmark
from cosmos_policy.experiments.robot.cosmos_utils import get_action
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
import cp_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_goal")
ap.add_argument("--task", type=int, default=8)
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--out", default="/workspace/results/rollout_check")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cfg = C.make_cfg("eval", task_suite_name=args.suite)
model, stats = C.load_cosmos(cfg)
suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
init_states = suite.get_task_init_states(args.task)
env, lang = get_libero_env(task, "cosmos", resolution=cfg.env_img_res)
print("task", args.task, lang, flush=True)

episodes = []
for ep in range(args.episodes):
    env.reset()
    obs = env.set_init_state(init_states[ep])
    t, success, q_times, sim_times, values = 0, False, [], [], []
    pred_vs_real = []
    queue, pending = [], None
    while t < args.max_steps + 10:
        if t < 10:
            obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
            t += 1
            continue
        o = C.env_obs_to_policy_obs(obs, flip=cfg.flip_images)
        if not queue:
            if pending is not None:  # compare the previous chunk's predicted future frame with the real frame now
                real = C.CU.prepare_images_for_model([o["primary_image"]], cfg)[0]
                pred_vs_real.append(float(np.abs(pending.astype(np.float32) - real.astype(np.float32)).mean()))
                if ep == 0 and len(pred_vs_real) in (2, 5):
                    Image.fromarray(np.concatenate([real, pending], axis=1)).save(
                        f"{args.out}/ep0_chunk{len(pred_vs_real)}_real_vs_pred.png")
            C.sync(); t0 = time.time()
            rd = get_action(cfg, model, stats, o, lang, seed=cfg.seed, num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True)
            C.sync(); q_times.append(time.time() - t0)
            queue = list(rd["actions"])
            values.append(float(rd["value_prediction"]))
            pending = np.asarray(rd["future_image_predictions"]["future_image"])
        a = queue.pop(0)
        t0 = time.time()
        obs, _, done, _ = env.step(np.asarray(a).tolist())
        sim_times.append(time.time() - t0)
        if done:
            success = True
            break
        t += 1
    rec = dict(episode=ep, success=success, steps=t, n_queries=len(q_times), query_s_mean=float(np.mean(q_times)),
               sim_step_s_mean=float(np.mean(sim_times)), values=values, pred_vs_real_mae=pred_vs_real)
    episodes.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("values", "pred_vs_real_mae")}),
          "values", np.round(values, 3).tolist(), flush=True)

summary = dict(suite=args.suite, task=args.task, language=lang, n=len(episodes),
               success_rate=float(np.mean([e["success"] for e in episodes])),
               query_s_mean=float(np.mean([e["query_s_mean"] for e in episodes])),
               sim_step_s_mean=float(np.mean([e["sim_step_s_mean"] for e in episodes])),
               episodes=episodes)
json.dump(summary, open(f"{args.out}/summary_{args.suite}_t{args.task}.json", "w"), indent=1)
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "episodes"}))
