"""Feasibility sweep for E1 (M2): can a boundary-targeted perturbation of successful demos be calibrated so that the
final margin r = e/tau (libero_margin.py) spans the near-boundary range continuously?

Families (B-specific, applied like CureWM families to recorded demo actions; CureWM phase annotation reused):
  place_offset(delta): over the whole carry phase add a constant horizontal action offset sized so that the
      realized end-effector displacement at release is about `delta` metres (G_EFF calibration) in direction theta
      (theta drawn per (demo, k), common random numbers across severities as in CureWM).
  pull_truncate(s): stop the translation for the last fraction s of the carry (pull) phase (drawer task).
  under_actuation(s): scale the translation part of the actions in the manipulation phase (grasp..end for
      articulated/push tasks) by (1 - s).
Physics only (no rendering); r is measured at the end of the action sequence and after 15 settle steps
(zero motion, gripper open). LIBERO's own predicate at the settled state is recorded for agreement.
Run:  $VP /workspace/code/nearmiss_sweep.py
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/workspace/results/nearmiss_sweep")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--demos", type=int, default=4)
ap.add_argument("--tasks", type=int, nargs="*", default=None)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

G_EFF = 0.010  # realized end-effector displacement per unit of sustained extra action per step (measured: ~20% of
               # the 0.05 m OSC output scale, because delta targets are re-anchored at the current pose every step)
PLAN = {  # task id -> (family, severities). place_offset severity = target realized displacement (m)
    8: ("place_offset", [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]),
    6: ("place_offset", [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]),
    1: ("place_offset", [0.0, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.21, 0.25, 0.30]),
    4: ("place_offset", [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.35]),
    5: ("under_actuation", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]),
    0: ("pull_truncate", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]),
}
K_DIRS = 2


def run_task(ti):
    from curewm.backends.libero import LiberoBackend, load_libero_demos
    from curewm.perturbations import Phase, annotate_phases
    import libero_margin as M
    fam, sevs = PLAN[ti]
    demos = load_libero_demos("libero_goal", args.demos, task_ids=[ti])
    be = LiberoBackend("libero_goal", render=False)
    rows = []
    for di, d in enumerate(demos):
        be.reset_to(d.init_state)
        nom = be.rollout(d.actions)
        if not nom["success"]:
            continue
        d.contacts = nom["contacts"]
        ph = annotate_phases(d)
        carry = np.where(ph == Phase.CARRY.value)[0]
        grasp_on = np.where(ph >= Phase.GRASP.value)[0]
        for k in range(K_DIRS if fam == "place_offset" else 1):
            rng = np.random.default_rng(1000 * ti + 10 * di + k)
            theta = rng.uniform(0, 2 * np.pi)
            for s in sevs:
                a = d.actions.copy()
                if fam == "place_offset":  # whole carry phase, constant horizontal offset
                    if len(carry) < 2:
                        continue
                    per_step = s / (len(carry) * G_EFF)
                    a[carry, 0] += per_step * np.cos(theta)
                    a[carry, 1] += per_step * np.sin(theta)
                    a[:, :6] = np.clip(a[:, :6], -1.0, 1.0)
                elif fam == "pull_truncate":  # stop translating for the last fraction s of the manipulation window
                    # drawer demos never close the gripper (the handle is pulled with open fingers), so CureWM's
                    # phase annotation gives APPROACH everywhere: use the second half of the demo as the pull window
                    w0, w1 = (int(carry[0]), int(carry[-1]) + 1) if len(carry) >= 2 else (len(a) // 2, len(a))
                    cut = w0 + int(round((1.0 - s) * (w1 - w0)))
                    a[cut:w1, 0:3] = 0.0
                else:
                    start = int(grasp_on[0]) if len(grasp_on) else len(a) // 2
                    a[start:, 0:3] *= (1.0 - s)
                be.reset_to(d.init_state)
                res = be.rollout(a)
                m_end = M.task_margin(be._env)
                for _ in range(15):
                    be._env.step([0, 0, 0, 0, 0, 0, -1])
                m_set = M.task_margin(be._env)
                rows.append(dict(task=ti, family=fam, demo=di, k=k, theta=float(theta), severity=float(s),
                                 success_ever=bool(res["success"]), r_end=m_end["r"], r_settled=m_set["r"],
                                 success_settled_libero=m_set["success_libero"], success_settled_geom=m_set["success_geom"],
                                 aux_ok_settled=m_set["aux_ok"]))
    be.close()
    return rows


if __name__ == "__main__":
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(run_task, args.tasks or list(PLAN)):
            rows.extend(res)
    with open(os.path.join(args.out, "sweep.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {}
    for ti in (args.tasks or list(PLAN)):
        rs = [r for r in rows if r["task"] == ti]
        if not rs:
            continue
        by_s = {}
        for r in rs:
            by_s.setdefault(r["severity"], []).append(r["r_settled"])
        rset = np.array([r["r_settled"] for r in rs])
        summ[ti] = dict(family=PLAN[ti][0], n=len(rs),
                        agree_geom_libero=float(np.mean([r["success_settled_geom"] == r["success_settled_libero"] for r in rs])),
                        agree_settled_vs_ever=float(np.mean([r["success_settled_libero"] == r["success_ever"] for r in rs])),
                        r_median_by_severity={str(k): float(np.median(v)) for k, v in sorted(by_s.items())},
                        frac_in_0p8_3=float(np.mean((rset >= 0.8) & (rset < 3))),
                        bands=dict(lt1=int((rset < 1).sum()), near_1_2=int(((rset >= 1) & (rset < 2)).sum()),
                                   mid_2_3=int(((rset >= 2) & (rset < 3)).sum()), far_ge3=int((rset >= 3).sum())),
                        spearman_sev_r=float(np.corrcoef(np.argsort(np.argsort([r["severity"] for r in rs])),
                                                         np.argsort(np.argsort(rset)))[0, 1]))
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
