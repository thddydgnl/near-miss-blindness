"""Manifests for the M2 runner (b_runner.py). Standard library only (runs on the Mac or the pod).

  e1-candidates  phase 1 of E1: cf jobs on the E1 demos (d20-d49): nominal + CureWM 6 families x 0.2..1.0 + the task's
                 boundary-targeted family at the frozen severity list and its far-miss severities, dirs 0 and 1.
                 t0 (drawer, E1 only): nominal + pull_truncate at the frozen list and far severities, dir 0 only
                 (CureWM families never act on the drawer task, E0; pull_truncate has no direction draw, so dir 1
                 would repeat dir 0 exactly)
  e1-select      phase 2 of E1: from the merged cf results, a seeded stratified selection per task: 500 counterfactuals
                 with r_settled in [0.8, 3] (11 bins of 0.2, equal quotas; a bin's unfilled quota goes to the other bins)
                 + 100 successes with r < 0.8 + 100 far-misses with r >= 3 -> cf + imagine jobs
  e2             policy_ep + imagine_ep jobs: policies x tasks x LIBERO official init states 0..49 (modes O, C)
  smoke-e1 / smoke-e2  tiny manifests on E0b start states only (held demos d17-d19, held init seeds)
Usage: python3 make_manifest.py <cmd> --out <manifest.jsonl> [options]
"""
import argparse
import json
import random

CUREWM = ["insufficient_grip", "premature_release", "carry_slip", "contact_oscillation", "wrist_tilt", "approach_overshoot"]
M2_TASKS = [1, 2, 4, 5, 6, 8]
E1_TASKS = M2_TASKS + [0]  # t0 in E1 only (coordinator decision 2026-10-21)


def cf_job(task, demo, family, s, k):
    jid = f"cf_t{task}_d{demo:02d}_{family}_s{s:.4f}_k{k}"
    return dict(job_id=jid, kind="cf", task=task, demo=demo, family=family, severity=float(s), dir=k)


def imagine_job(cf):
    return dict(job_id="im_" + cf["job_id"][3:], kind="imagine", dep=cf["job_id"])


def e1_candidates(calib, demos, tasks=E1_TASKS):
    jobs = []
    for task in tasks:
        fam, sev_list = calib[str(task)]["family"], calib[str(task)]["severities"]
        dirs = (0,) if fam == "pull_truncate" else (0, 1)  # pull_truncate is deterministic in dir
        for d in demos:
            jobs.append(cf_job(task, d, "nominal", 0.0, 0))
            if task != 0:
                jobs += [cf_job(task, d, f, s, 0) for f in CUREWM for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
            jobs += [cf_job(task, d, fam, s, k) for s in sev_list + calib[str(task)].get("far_severities", []) for k in dirs]
    seen, uniq = set(), []
    for j in jobs:  # t2/t4: the last listed severity equals the first far severity (0.45): one job, not two
        if j["job_id"] not in seen:
            seen.add(j["job_id"])
            uniq.append(j)
    return uniq


def e1_select(cf_results, per_task=500, n_success=100, n_far=100, seed=0, tasks=E1_TASKS):
    bins = [(0.8 + 0.2 * i, 1.0 + 0.2 * i) for i in range(11)]
    sel, report = [], {}
    for task in tasks:
        rs = [r for r in cf_results if r["task"] == task and not r.get("skipped")]
        # seeded order that does not depend on the order results were merged in
        order = sorted(rs, key=lambda r: random.Random(f"{seed}|{r['job_id']}").random())
        by_bin = [[r for r in order if lo <= r["r_settled"] < hi] for lo, hi in bins]
        got = [[] for _ in bins]
        while sum(map(len, got)) < per_task:  # round-robin over bins: equal quotas, spill-over when a bin runs out
            moved = False
            for i in range(len(bins)):
                if sum(map(len, got)) >= per_task:
                    break
                if len(got[i]) < len(by_bin[i]):
                    got[i].append(by_bin[i][len(got[i])])
                    moved = True
            if not moved:
                break
        succ = [r for r in order if r["success_settled"] and r["r_settled"] < 0.8][:n_success]
        far = [r for r in order if r["r_settled"] >= 3][:n_far]
        sel += [r for g in got for r in g] + succ + far
        report[task] = dict(bins=[len(g) for g in got], available=[len(b) for b in by_bin], successes=len(succ),
                            far=len(far), candidates=len(rs))
    return sel, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e1-candidates", "e1-select", "e2", "smoke-e1", "smoke-e2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=None, help="frozen severity settings JSON {task: {family, severities}}")
    ap.add_argument("--cf_results", default=None, help="merged cf results JSONL (e1-select)")
    ap.add_argument("--policies", default=None, help="JSON list of policies [{type, ckpt, name, tasks?}] (e2)")
    ap.add_argument("--H_C", default=None, help="JSON {task: chunks} for mode C (e2)")
    ap.add_argument("--demos", default="20-49", help="source demos for e1-candidates (E1 = 20-49; E0b validation 17-19)")
    ap.add_argument("--tasks", type=int, nargs="*", default=E1_TASKS, help="e1-candidates / e1-select tasks")
    args = ap.parse_args()
    jobs = []
    if args.cmd == "e1-candidates":
        calib = json.load(open(args.calib))
        lo, hi = (int(x) for x in args.demos.split("-"))
        jobs = e1_candidates(calib, range(lo, hi + 1), args.tasks)
    elif args.cmd == "e1-select":
        res = [json.loads(l) for l in open(args.cf_results) if l.strip()]
        sel, rep = e1_select(res, tasks=args.tasks)
        cfs = {r["job_id"] for r in sel}
        jobs = [dict(job_id=r["job_id"], kind="cf", task=r["task"], demo=r["demo"], family=r["family"],
                     severity=r["severity"], dir=r["dir"]) for r in sel]
        jobs += [imagine_job(j) for j in jobs]
        print(json.dumps(rep))
    elif args.cmd == "e2":
        pols = json.load(open(args.policies))
        hc = json.load(open(args.H_C))
        for p in pols:
            for task in p.get("tasks", M2_TASKS):
                for i in range(50):
                    ep = dict(job_id=f"ep_{p['name']}_t{task}_init{i:02d}", kind="policy_ep", task=task,
                              policy={k: v for k, v in p.items() if k != "tasks"},
                              start={"type": "libero_init", "index": i})
                    jobs.append(ep)
                    jobs.append(dict(job_id="ie_" + ep["job_id"][3:], kind="imagine_ep", dep=ep["job_id"],
                                     modes=["O", "C"] if p["type"] == "dp" else ["O"], H_C=hc[str(task)]))
    elif args.cmd == "smoke-e1":
        for task, d, fam, s in [(8, 17, "place_offset", 0.05), (8, 18, "nominal", 0.0), (5, 17, "under_actuation", 0.4),
                                (1, 19, "premature_release", 0.6)]:
            j = cf_job(task, d, fam, s, 0)
            jobs += [j, imagine_job(j)]
    elif args.cmd == "smoke-e2":
        pols = json.load(open(args.policies))
        for p in pols:
            for task in p.get("tasks", [8]):
                for i in range(2):
                    ep = dict(job_id=f"ep_{p['name']}_t{task}_held{i:02d}", kind="policy_ep", task=task,
                              policy={k: v for k, v in p.items() if k != "tasks"},
                              start={"type": "held_file", "path": f"/workspace/dp/held_init/t{task}.npy", "index": i})
                    jobs += [ep, dict(job_id="ie_" + ep["job_id"][3:], kind="imagine_ep", dep=ep["job_id"],
                                      modes=["O", "C"], H_C=3)]
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs -> {args.out}")
