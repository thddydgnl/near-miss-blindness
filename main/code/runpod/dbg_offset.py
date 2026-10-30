import numpy as np
from curewm.backends.libero import LiberoBackend, load_libero_demos
from curewm.perturbations import Phase, annotate_phases
import libero_margin as M
for ti in [8, 1]:
    demos = load_libero_demos("libero_goal", 2, task_ids=[ti])
    be = LiberoBackend("libero_goal", render=False)
    d = demos[1]
    be.reset_to(d.init_state); nom = be.rollout(d.actions); d.contacts = nom["contacts"]; ph = annotate_phases(d)
    carry = np.where(ph == Phase.CARRY.value)[0]; t_rel = int(carry[-1]) + 1
    print("task", ti, "T", len(d.actions), "carry", carry[0], carry[-1], "t_rel", t_rel, "phases", np.bincount(ph, minlength=5).tolist())
    print("  max |a_xy| in window", np.abs(d.actions[t_rel-12:t_rel, :2]).max().round(3))
    m0 = M.task_margin(be._env)
    for s, W in [(0.0, 12), (0.06, 12), (0.12, 12), (0.12, 0), (0.24, 0)]:
        a = d.actions.copy(); win = np.arange(max(int(carry[0]), t_rel - W), t_rel) if W else carry
        a[win, 0] += s / (len(win) * 0.05)
        be.reset_to(d.init_state); res = be.rollout(a)
        ee_rel = res["proprio"][t_rel, :3]; ee_nom = nom["proprio"][t_rel, :3]
        m = M.task_margin(be._env)
        p = m["parts"][0]
        dom = be._env.env
        print("  s=%.2f W=%d ee_dx_at_release=%+.3f r=%.2f succ=%s item_xy=%s base_xy=%s" % (
            s, len(win), ee_rel[0] - ee_nom[0], m["r"], m["success_libero"],
            np.round(dom.sim.data.body_xpos[dom.obj_body_id["akita_black_bowl_1"]][:2], 3),
            np.round(dom.sim.data.body_xpos[dom.obj_body_id["plate_1"]][:2], 3) if ti == 8 else "-"))
    be.close()
