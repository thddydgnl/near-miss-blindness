"""Are LIBERO fixture placements (static bodies, not in the flattened MuJoCo state) random at every reset, and does
np.random.seed / env.seed control them? Prints fixture body positions across resets."""
import os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

b = benchmark.get_benchmark_dict()["libero_goal"]()
t = b.get_task(2)
env = ControlEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                 has_offscreen_renderer=False, use_camera_obs=False)
dom = env.env
names = [n for n in ["wooden_cabinet_1_main", "flat_stove_1_main", "wine_rack_1_main"]]
def fx():
    m = dom.sim.model
    out = []
    for i in range(m.nbody):
        n = m.body_id2name(i)
        if n and any(k in n for k in ("wooden_cabinet_1", "flat_stove_1", "wine_rack_1")) and n.count("_") <= 3:
            out.append((n, np.round(m.body_pos[i], 4).tolist()))
    return out[:3]
for trial in range(3):
    env.reset(); print("unseeded reset", trial, fx())
for trial in range(2):
    np.random.seed(123); env.seed(123); env.reset(); print("seed 123 reset", trial, fx())
init = b.get_task_init_states(2)[0]
np.random.seed(5); env.reset(); env.set_init_state(init); print("after set_init_state (seed 5)", fx())
