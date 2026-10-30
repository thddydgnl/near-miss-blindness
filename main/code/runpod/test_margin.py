"""Test the r = e / tau computation (libero_margin.py) on recorded rollouts: the CureWM counterfactual episodes
store the full MuJoCo state at every step. For each episode the final state is restored and r, aux_ok and LIBERO's
own predicate are evaluated; also the state at the first step where LIBERO reported success, if any.
Outputs /workspace/results/margin/cf_margins.jsonl and summary.json.
Run:  $VP /workspace/code/test_margin.py --tasks 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/data/cf_goal")
ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
ap.add_argument("--out", default="/workspace/results/margin")
ap.add_argument("--workers", type=int, default=5)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def run_task(ti):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    import libero_margin as M
    tdir = os.path.join(args.data, f"t{ti}")
    if not os.path.exists(os.path.join(tdir, "index.jsonl")):
        return []
    b = benchmark.get_benchmark_dict()["libero_goal"]()
    t = b.get_task(ti)
    env = ControlEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                     has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    out = []
    for line in open(os.path.join(tdir, "index.jsonl")):
        r = json.loads(line)
        with np.load(os.path.join(tdir, "episodes", f"{r['name']}.npz")) as z:
            states = z["sim_states"]
        M.set_flat_state(env, states[-1])
        m = M.task_margin(env)
        rec = dict(task=ti, key=f"t{ti}/{r['name']}", family=r["family"], severity=r["severity"],
                   outcome_ever=r["outcome"], r_final=m["r"], aux_ok=m["aux_ok"], success_geom=m["success_geom"],
                   success_libero_final=m["success_libero"], pred=m["parts"][0]["pred"], T=int(len(states)))
        p0 = m["parts"][0]
        for k in ("contact_ok", "vert_ok", "z_ok"):
            if k in p0:
                rec[k] = p0[k]
        out.append(rec)
    env.close()
    return out


if __name__ == "__main__":
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(run_task, args.tasks):
            rows.extend(res)
    with open(os.path.join(args.out, "cf_margins.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {}
    for ti in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == ti]
        pert = [r for r in rs if r["family"] != "nominal"]
        fails = [r for r in pert if not r["success_libero_final"]]
        rf = np.array([r["r_final"] for r in fails]) if fails else np.array([])
        summ[ti] = dict(
            pred=rs[0]["pred"], n=len(rs), n_perturbed=len(pert),
            agree_geom_vs_libero_final=float(np.mean([r["success_geom"] == r["success_libero_final"] for r in rs])),
            agree_final_vs_ever=float(np.mean([r["success_libero_final"] == r["outcome_ever"] for r in rs])),
            n_fail_final=len(fails),
            fail_r_bands=dict(lt1=int((rf < 1).sum()), near_1_2=int(((rf >= 1) & (rf < 2)).sum()),
                              mid_2_3=int(((rf >= 2) & (rf < 3)).sum()), far_ge3=int((rf >= 3).sum())),
            fail_r_quantiles=(np.quantile(rf, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist() if len(rf) else None),
            success_r_quantiles=(np.quantile([r["r_final"] for r in rs if r["success_libero_final"]], [0.5, 0.9, 0.99]).round(3).tolist()
                                 if any(r["success_libero_final"] for r in rs) else None),
        )
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
