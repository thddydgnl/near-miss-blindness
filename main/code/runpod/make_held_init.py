"""Held start states for E0b policy checks (never used by E2): env.reset() placements with seeds 10000..10019 per task,
saved as flat MuJoCo states to /workspace/dp/held_init/t<task>.npy (dp_eval.py and the E2 smoke manifest read them)."""
import os
import sys

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

os.makedirs("/workspace/dp/held_init", exist_ok=True)
b = benchmark.get_benchmark_dict()["libero_goal"]()
for task in [int(x) for x in sys.argv[1:]] or [1, 2, 4, 5, 6, 8]:
    p = f"/workspace/dp/held_init/t{task}.npy"
    if os.path.exists(p):
        continue
    t = b.get_task(task)
    env = ControlEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                     has_offscreen_renderer=False, use_camera_obs=False)
    st = []
    for i in range(20):
        np.random.seed(10000 + i)
        env.seed(10000 + i)
        env.reset()
        st.append(np.asarray(env.env.sim.get_state().flatten()).copy())
    st = np.stack(st)
    print(task, st.shape, "distinct", len({tuple(np.round(x[:40], 5)) for x in st}), flush=True)
    np.save(p, st)
    env.close()
