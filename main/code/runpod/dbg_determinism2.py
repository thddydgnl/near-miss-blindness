"""After the fix: a counterfactual gives the same result in a fresh process-env and after other rollouts, and the render
env carries the physics env's fixture placement."""
import numpy as np
from b_cf import Generator


def run(gen, di, fam, s, render=False):
    d, ok = gen.demo(di)
    if not ok:
        return ("replay fail",)
    a, _ = gen.perturb(d, fam, s, 0)
    r = gen.rollout(d, a)
    out = (round(r["r_settled"], 6), r["success_settled"], r["success_ever"])
    if render:
        gen.render(r["states"]["Slift"])
        same = np.allclose(gen._sim().model.body_pos, gen.renv.env.sim.model.body_pos)
        out = out + (("render_sync", same),)
    return out


g1 = Generator(2)
a1 = run(g1, 18, "nominal", 0.0, render=True)
b1 = run(g1, 5, "place_offset", 0.2)
g2 = Generator(2)
run(g2, 17, "place_offset", 0.2)
run(g2, 3, "wrist_tilt", 0.6)
b2 = run(g2, 5, "place_offset", 0.2)
a2 = run(g2, 18, "nominal", 0.0, render=True)
print("d18 nominal fresh:", a1, " after others:", a2)
print("d05 place_offset fresh-ish:", b1, " after others:", b2)
