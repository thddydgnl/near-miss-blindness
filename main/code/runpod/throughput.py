"""Throughput for B's M2 cost model.
  --part cosmos : Cosmos Policy query time (single counterfactual query with 5 or 1 action steps; chained imagination
                  of 1-4 chunks; value of the real future state; batched counterfactual queries B = 1..16) and VRAM.
  --part sim    : LIBERO steps/s with EGL: physics only, 2 cameras at 224 (CureWM) and 256 (Cosmos eval),
                  single process and P parallel processes.
Run from the cosmos-policy checkout:  $VP /workspace/code/throughput.py --part cosmos ; ... --part sim
"""
import argparse
import json
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["cosmos", "sim"], required=True)
ap.add_argument("--task", type=int, default=8)
ap.add_argument("--out", default="/workspace/results/throughput")
args, _ = ap.parse_known_args()
os.makedirs(args.out, exist_ok=True)


def make_env(ti, res):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv
    b = benchmark.get_benchmark_dict()["libero_goal"]()
    t = b.get_task(ti)
    f = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    if res == 0:
        env = ControlEnv(bddl_file_name=f, has_offscreen_renderer=False, use_camera_obs=False)
    else:
        env = OffScreenRenderEnv(bddl_file_name=f, camera_heights=res, camera_widths=res)
    env.seed(0)
    return env, b.get_task_init_states(ti), t.language


def sim_job(spec):
    ti, res, n_steps = spec
    env, inits, _ = make_env(ti, res)
    env.reset()
    env.set_init_state(inits[0])
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i in range(n_steps):
        a = np.concatenate([rng.normal(0, 0.3, 6), [-1.0]])
        env.step(a.tolist())
        if (i + 1) % 100 == 0:
            env.reset()
            env.set_init_state(inits[(i // 100) % len(inits)])
    dt = time.time() - t0
    env.close()
    return n_steps / dt


if __name__ != "__main__":
    pass  # spawned sim workers import this module only for sim_job
elif args.part == "sim":
    res = {}
    for resl in [0, 224, 256]:
        n = 600 if resl == 0 else 300
        res[f"single_res{resl}_steps_per_s"] = sim_job((args.task, resl, n))
        print(resl, res[f"single_res{resl}_steps_per_s"], flush=True)
    for P in [4, 7]:
        for resl in [0, 256]:
            n = 600 if resl == 0 else 300
            t0 = time.time()
            with ProcessPoolExecutor(P, mp_context=mp.get_context("spawn")) as ex:  # EGL breaks in forked children
                per = list(ex.map(sim_job, [(args.task, resl, n)] * P))
            res[f"par{P}_res{resl}_aggregate_steps_per_s"] = float(sum(per))
            res[f"par{P}_res{resl}_per_proc"] = [float(x) for x in per]
            print(P, resl, sum(per), flush=True)
    res["note"] = "reset+set_init_state every 100 steps is included; env construction excluded"
    json.dump(res, open(f"{args.out}/sim.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
else:
    import torch
    import cp_common as C
    cfg = C.make_cfg("eval")
    model, stats = C.load_cosmos(cfg)
    env, inits, lang = make_env(args.task, 256)
    env.reset()
    obs = env.set_init_state(inits[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    o = C.env_obs_to_policy_obs(obs)
    rng = np.random.default_rng(0)
    chunks = [np.concatenate([rng.normal(0, 0.2, (16, 6)), -np.ones((16, 1))], axis=1) for _ in range(4)]
    res = {}

    def timeit(fn, n=5):
        fn()  # warm-up
        ts = []
        for _ in range(n):
            C.sync(); t = time.time(); fn(); C.sync(); ts.append(time.time() - t)
        return float(np.median(ts))

    res["cf_query_act5_s"] = timeit(lambda: C.cf_query(cfg, model, stats, o, lang, chunks[0], seed=195, n_act_steps=5))
    res["cf_query_act1_s"] = timeit(lambda: C.cf_query(cfg, model, stats, o, lang, chunks[0], seed=195, n_act_steps=1))
    for k in [1, 2, 3, 4]:
        res[f"chain_{k}chunks_act1_s"] = timeit(lambda: C.imagine_chain(cfg, model, stats, o, lang, chunks[:k], seed=195, n_act_steps=1), n=3)
    res["value_real_future_s"] = timeit(lambda: C.value_real_future(cfg, model, stats, o, lang, chunks[0], o, seed=195))
    res["policy_get_action_s"] = timeit(lambda: C.CU.get_action(cfg, model, stats, o, lang, seed=195, num_denoising_steps_action=5,
                                                                generate_future_state_and_value_in_parallel=True))
    single = C.cf_query(cfg, model, stats, o, lang, chunks[0], seed=195, n_act_steps=1)["value"]
    for B in [1, 4, 8, 16]:
        items = [(o, lang, chunks[i % 4]) for i in range(B)]
        torch.cuda.reset_peak_memory_stats()
        tb = timeit(lambda: C.cf_query_batch(cfg, model, stats, items, seed=195), n=3)
        vals = [x["value"] for x in C.cf_query_batch(cfg, model, stats, items, seed=195)]
        res[f"batch{B}_s"] = tb
        res[f"batch{B}_s_per_query"] = tb / B
        res[f"batch{B}_vram_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        res[f"batch{B}_values_first4"] = vals[:4]
        print(B, tb, tb / B, res[f"batch{B}_vram_peak_gib"], flush=True)
    res["single_value_chunk0"] = single
    res["vram_max_allocated_gib_overall"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(res, open(f"{args.out}/cosmos.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
