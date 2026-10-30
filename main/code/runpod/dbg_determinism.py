"""Does a LIBERO rollout depend on what the env ran before? Same counterfactual in a fresh env vs after other
rollouts in the same env, with and without clearing MuJoCo's warm-start and the controller goal."""
import numpy as np
from b_cf import Generator


def run(gen, di, fam, s):
    d, ok = gen.demo(di)
    if not ok:
        return ("replay fail",)
    a, _ = gen.perturb(d, fam, s, 0)
    r = gen.rollout(d, a)
    return (round(r["r_settled"], 6), r["success_settled"], r["success_ever"])


task, di = 2, 18
g1 = Generator(task, render=False)
print("fresh   nominal d18:", run(g1, di, "nominal", 0.0), flush=True)
g2 = Generator(task, render=False)
print("after d17 rollouts:", run(g2, 17, "place_offset", 0.2), run(g2, 17, "wrist_tilt", 0.6), flush=True)
print("reused  nominal d18:", run(g2, di, "nominal", 0.0), flush=True)
sim = g2._sim()
print("qacc_warmstart nonzero:", int(np.count_nonzero(sim.data.qacc_warmstart)), flush=True)
