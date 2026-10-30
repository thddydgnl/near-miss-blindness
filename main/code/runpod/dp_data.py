"""Build the DP training set for one LIBERO-Goal task: all 50 official demonstrations, re-rendered exactly from the
recorded MuJoCo states (no replay drift) at 256 px, converted to the Cosmos Policy model view
(flip -> JPEG95 -> 224 -> 90% centre crop -> 224, cosmos_utils.prepare_images_for_model) and resized to 128 px.
Output: /workspace/dp/data/t<task>.npz with images (N, 2, 128, 128, 3) uint8 [agentview, wrist], proprio (N, 9),
actions (N, 7), ep_start, ep_len.
Run (one process per task, cwd = cosmos-policy checkout):  $VP /workspace/code/dp_data.py --task 8
"""
import argparse
import os
import time

import h5py
import numpy as np

import cp_common as C
from dp_model import resize_224_to_128

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, required=True)
ap.add_argument("--out", default="/workspace/dp/data")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

b = benchmark.get_benchmark_dict()["libero_goal"]()
task = b.get_task(args.task)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                         camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
cfg = C.make_cfg("eval")  # JPEG95 + test-time crop, as in the Cosmos LIBERO eval
f = os.path.join(get_libero_path("datasets"), "libero_goal", f"{task.name}_demo.hdf5")
imgs, props, acts, starts, lens = [], [], [], [], []
t0 = time.time()
with h5py.File(f, "r") as h:
    keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for k in keys:
        states = np.asarray(h["data"][k]["states"])
        a = np.asarray(h["data"][k]["actions"], dtype=np.float32)
        starts.append(len(acts))
        lens.append(len(a))
        for t in range(len(a)):
            obs = env.set_init_state(states[t])
            o = C.env_obs_to_policy_obs(obs, flip=True)
            p224 = C.CU.prepare_images_for_model([o["primary_image"], o["wrist_image"]], cfg)
            imgs.append(np.stack([resize_224_to_128(p224[0]), resize_224_to_128(p224[1])]))
            props.append(o["proprio"].astype(np.float32))
            acts.append(a[t])
env.close()
np.savez(os.path.join(args.out, f"t{args.task}.npz"), images=np.stack(imgs), proprio=np.stack(props), actions=np.stack(acts),
         ep_start=np.array(starts), ep_len=np.array(lens), language=task.language)
print(f"task {args.task} demos {len(lens)} steps {len(acts)} in {time.time() - t0:.0f}s", flush=True)
