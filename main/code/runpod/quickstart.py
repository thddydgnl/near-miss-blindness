"""Cosmos Policy README quick start (LIBERO sample observation) plus timing, VRAM, and a check that our batch
builder (cp_common.build_batch) reproduces get_action's counterfactual query exactly.
Run from the cosmos-policy checkout:  $VP /workspace/code/quickstart.py
"""
import json
import pickle
import time

import numpy as np
import torch
from PIL import Image

from cosmos_policy.experiments.robot.cosmos_utils import get_action
import cp_common as C

OUT = "/workspace/results"
cfg = C.make_cfg("eval")
t0 = time.time()
model, stats = C.load_cosmos(cfg)
t_load = time.time() - t0
print(f"model loaded in {t_load:.1f}s; VRAM allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

with open("cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl", "rb") as f:
    observation = pickle.load(f)
task = "put both the alphabet soup and the tomato sauce in the basket"
print("obs keys", {k: (np.asarray(v).shape, np.asarray(v).dtype) for k, v in observation.items()})

res = {"model_load_s": t_load}
times = []
for i in range(4):
    C.sync(); t = time.time()
    rd = get_action(cfg, model, stats, observation, task, seed=cfg.seed,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=True)
    C.sync(); times.append(time.time() - t)
print("get_action times (s)", [round(x, 3) for x in times])
print("actions[0:2]", np.asarray(rd["actions"])[:2])
print("value", rd["value_prediction"])
Image.fromarray(rd["future_image_predictions"]["future_image"]).save(f"{OUT}/quickstart_future_image.png")
Image.fromarray(rd["future_image_predictions"]["future_wrist_image"]).save(f"{OUT}/quickstart_future_wrist_image.png")
res.update(get_action_parallel_s=times, value=float(rd["value_prediction"]),
           action_chunk_shape=list(np.asarray(rd["actions"]).shape),
           first_action=np.asarray(rd["actions"])[0].tolist())

# Counterfactual query, CureWM path (get_action -> inject -> future -> value) vs our build_batch path
acts = np.asarray(rd["actions"])
cfg_ar = C.make_cfg("eval")
outs = {}
for label, n_act in [("act5", 5), ("act1", 1)]:
    tm = C.Timer()
    q = C.cf_query(cfg_ar, model, stats, observation, task, acts, seed=cfg.seed, n_act_steps=n_act, timer=tm)
    tm2 = C.Timer()
    q2 = C.cf_query(cfg_ar, model, stats, observation, task, acts, seed=cfg.seed, n_act_steps=n_act, timer=tm2)
    outs[label] = q
    res[f"cf_query_{label}"] = dict(value=q["value"], times_first=q["times"], times_second=q2["times"])
    print(label, "value", q["value"], "times", q2["times"], flush=True)
# CureWM's own call sequence on the same input (get_action with generate_future_state_and_value_in_parallel=False)
from cosmos_policy.experiments.robot.cosmos_utils import get_future_state_prediction, get_value_prediction
from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk
with torch.inference_mode():
    rd2 = get_action(cfg_ar, model, stats, observation, task, seed=cfg.seed,
                     num_denoising_steps_action=5, generate_future_state_and_value_in_parallel=False)
    db = rd2["data_batch"]
    gen = rd2["generated_latent"]
    a = torch.tensor(C.normalize_actions(acts, stats)[None], dtype=gen.dtype, device=gen.device)
    inj = replace_latent_with_action_chunk(gen.clone(), a, db["action_latent_idx"])
    fut = get_future_state_prediction(cfg_ar, model, data_batch=db, generated_latent_with_action=inj,
                                      orig_clean_latent_frames=rd2["orig_clean_latent_frames"],
                                      future_proprio_latent_idx=5, future_wrist_image_latent_idx=6,
                                      future_wrist_image2_latent_idx=-1, future_image_latent_idx=7,
                                      future_image2_latent_idx=-1, seed=cfg.seed, num_denoising_steps_future_state=1)
    val = get_value_prediction(cfg_ar, model, data_batch=db, future_state_samples_list=fut["future_state_samples_list"],
                               seed=cfg.seed, num_denoising_steps_value=1)
img_ref = np.asarray(fut["future_image_predictions"]["future_image"])
res["curewm_path_value"] = float(val["value_prediction"])
for label, q in outs.items():
    d = float(np.abs(q["future_image"].astype(np.float32) - img_ref[0].astype(np.float32) if img_ref.ndim == 4
                     else q["future_image"].astype(np.float32) - img_ref.astype(np.float32)).mean())
    res[f"{label}_vs_curewm_path"] = dict(value_diff=abs(q["value"] - res["curewm_path_value"]), mean_abs_img_diff=d)
print("curewm path value", res["curewm_path_value"], {k: v for k, v in res.items() if "vs_curewm" in k})

# Value of the real future state (judge (c) on real frames) when "future" = current observation (sanity: runs)
res["value_real_future_same_obs"] = C.value_real_future(cfg_ar, model, stats, observation, task, acts, observation, seed=cfg.seed)
res["vram_max_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
res["vram_max_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
print(json.dumps(res, indent=1, default=float))
json.dump(res, open(f"{OUT}/quickstart.json", "w"), indent=1, default=float)
