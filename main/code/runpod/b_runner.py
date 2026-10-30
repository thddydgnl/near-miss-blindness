"""Resumable multi-pod runner for B's M2 E1/E2 (Cosmos Policy LIBERO).

Manifest: JSON lines, one job each: {"job_id", "kind", ...params}. Kinds
  cf          one counterfactual (task, demo, family, severity, dir): physics rollout + renders     [CPU worker]
  imagine     end-aligned imagination of one cf: k = 1..4 real chunks + the lift chunk, judges     [GPU process]
  policy_ep   one real policy episode (DP checkpoint, task, start state) + renders                  [CPU worker, small GPU use]
  imagine_ep  mode O (replay of the executed chunks from the first observation) and, for DP policies,
              mode C (policy acting on imagined observations for H_C chunks), + lift chunk, judges  [GPU process]
Outputs under <root>/<stage>/: results/<kind>/<job_id>.json (atomic write = done marker), art/<job_id>.npz
(per-pod artifacts: renders, states, actions; never pulled), small/<job_id>.npz (features, pulled with results).
Restart / new pod: jobs whose result JSON exists, or whose id is in --done (merged on the Mac), are skipped. An
imagine job whose cf artifact is missing (e.g. produced on an earlier pod) re-runs the cf job first; cf and
policy_ep jobs are deterministic (physics, fixed seeds), so the regenerated artifact is identical.
Parallel pods: --shard i/N keeps jobs with crc32(key) % N == i, key = dep id for GPU jobs, else job id.
Usage (cwd = cosmos-policy checkout):
  $VP b_runner.py run --manifest m.jsonl --stage e1 [--cpu 6] [--shard 0/1] [--done done_ids.txt]
  $VP b_runner.py status --manifest m.jsonl --stage e1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

ROOT = "/workspace/runs"
CPU_KINDS = ("cf", "policy_ep")
GPU_KINDS = ("imagine", "imagine_ep")
DEP = {"imagine": "cf", "imagine_ep": "policy_ep"}
RENDER_KEYS = ("S0", "Sk1", "Sk2", "Sk3", "Sk4", "ST", "Slift")


def paths(root, stage, kind, job_id):
    base = os.path.join(root, stage)
    return (os.path.join(base, "results", kind, job_id + ".json"), os.path.join(base, "art", job_id + ".npz"),
            os.path.join(base, "small", job_id + ".npz"))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    os.replace(path + ".tmp", path)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path + ".tmp.npz", **arrays)
    os.replace(path + ".tmp.npz", path)


def load_manifest(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# ============================================================== CPU worker side
_GEN, _POL, _ENV = {}, {}, {}


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def do_cf(job, root, stage, force=False, render_keys=RENDER_KEYS):
    from b_cf import Generator, run_cell
    rp, ap_, _ = paths(root, stage, "cf", job["job_id"])
    if not force and os.path.exists(rp) and os.path.exists(ap_):
        return job["job_id"]
    t = job["task"]
    key = (t, bool(render_keys))
    if key not in _GEN:  # physics-only generators (no render env) for render-free stages (E1 candidates)
        _GEN[key] = Generator(t, render=bool(render_keys))
    tmpdir = os.path.join(root, stage, "tmp_cf")
    os.makedirs(tmpdir, exist_ok=True)
    rec = run_cell(_GEN[key], job["demo"], job["family"], job["severity"], job["dir"], tmpdir, render_keys=render_keys)
    src = os.path.join(tmpdir, rec["cf_id"] + ".npz")
    if os.path.exists(src):
        os.makedirs(os.path.dirname(ap_), exist_ok=True)
        os.replace(src, ap_)
    jp = os.path.join(tmpdir, rec["cf_id"] + ".json")
    if os.path.exists(jp):
        os.remove(jp)
    rec["job_id"] = job["job_id"]
    write_json(rp, rec)
    return job["job_id"]


def _env_for(task):
    if task not in _ENV:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        b = benchmark.get_benchmark_dict()["libero_goal"]()
        tk = b.get_task(task)
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), tk.problem_folder, tk.bddl_file),
                                 camera_heights=256, camera_widths=256)
        env.seed(0)
        _ENV[task] = (env, b.get_task_init_states(task), tk.language)
    return _ENV[task]


def start_state(job):
    env, inits, _ = _env_for(job["task"])
    st = job["start"]
    if st["type"] == "libero_init":
        return np.asarray(inits[st["index"]])
    if st["type"] == "held_file":
        return np.load(st["path"])[st["index"]]
    raise ValueError(st)


def model_view(obs, cfg):
    import cp_common as C
    o = C.env_obs_to_policy_obs(obs, flip=True)
    p224 = C.CU.prepare_images_for_model([o["primary_image"], o["wrist_image"]], cfg)
    return p224[0], p224[1], o["proprio"].astype(np.float32)


def do_policy_ep(job, root, stage, force=False):
    import cp_common as C
    import libero_margin as M
    from b_cf import LIFT, SETTLE
    from dp_model import PolicyRunner
    rp, ap_, _ = paths(root, stage, "policy_ep", job["job_id"])
    if not force and os.path.exists(rp) and os.path.exists(ap_):
        return job["job_id"]
    env, _, lang = _env_for(job["task"])
    cfg = C.make_cfg("eval")
    ck = job["policy"]["ckpt"]
    if ck not in _POL:
        _POL[ck] = PolicyRunner(ck, seed=0)
    pol = _POL[ck]
    pol.gen.manual_seed(zlib.crc32(job["job_id"].encode()) & 0x7FFFFFFF)  # per-job deterministic sampling noise
    seed = zlib.crc32(("fixtures|" + job["job_id"]).encode()) & 0xFFFFFFFF
    np.random.seed(seed)  # fixtures are re-placed (+-1 cm) at every reset and are not in the MuJoCo state: seed it
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(start_state(job))
    for _ in range(10):  # no-op steps as in the LIBERO eval
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    sim = env.env.sim
    s_start = np.asarray(sim.get_state().flatten())
    p0, w0, pr0 = model_view(obs, cfg)
    chunks, success, t, t_success = [], False, 0, None
    max_steps = job.get("max_steps", 300)
    p, w, pr = p0, w0, pr0
    while t < max_steps and not success:
        last_start = (p, w, pr)
        a = pol.act(p, w, pr)
        chunks.append(a)
        for x in a:  # the whole chunk is executed even after success (chunk-aligned end state for mode O)
            obs, _, done, _ = env.step(np.asarray(x, dtype=np.float64).tolist())
            t += 1
            if done and not success:
                success, t_success = True, t
        p, w, pr = model_view(obs, cfg)
    s_end = np.asarray(sim.get_state().flatten())
    for x in SETTLE:
        env.step(x.tolist())
    ms = M.task_margin(env)
    M.set_flat_state(env, s_end)
    for x in LIFT:
        env.step(x.tolist())
    ml = M.task_margin(env)
    pl, wl, prl = model_view(env.env._get_observations(force_update=True), cfg)
    save_npz(ap_, actions=np.stack(chunks).astype(np.float32), start_primary=p0, start_wrist=w0, start_proprio=pr0,
             end_primary=p, end_wrist=w, end_proprio=pr, lift_primary=pl, lift_wrist=wl, lift_proprio=prl,
             last_start_primary=last_start[0], last_start_wrist=last_start[1], last_start_proprio=last_start[2],
             state_start=s_start, state_end=s_end)
    rec = dict(job_id=job["job_id"], task=job["task"], policy=job["policy"], start=job["start"], language=lang,
               success=success, t_success=t_success, steps=t, n_chunks=len(chunks), r_settled=ms["r"],
               success_settled=ms["success_libero"], aux_settled=ms["aux_ok"], success_lift=ml["success_libero"],
               lift_changes_outcome=bool(ml["success_libero"] != ms["success_libero"]))
    write_json(rp, rec)
    return job["job_id"]


def run_cpu_job(job, root, stage, force=False, render_keys=RENDER_KEYS):
    try:
        if job["kind"] == "cf":
            return ("ok", do_cf(job, root, stage, force, render_keys))
        if job["kind"] == "policy_ep":
            return ("ok", do_policy_ep(job, root, stage, force))
        raise ValueError(job["kind"])
    except Exception:
        return ("err", job["job_id"] + "\n" + traceback.format_exc())


# ============================================================== GPU side
class Gpu:
    def __init__(self, n_act_steps=1, vlm=False):
        import cp_common as C
        self.C = C
        self.cfg = C.make_cfg("eval")
        self.model, self.stats = C.load_cosmos(self.cfg)
        self.n_act = n_act_steps
        self.vlm = vlm
        self._pol = {}

    def chain(self, obs, lang, chunks):
        return self.C.imagine_chain(self.cfg, self.model, self.stats, obs, lang, chunks, seed=self.cfg.seed,
                                    n_act_steps=self.n_act, pre_first=True)

    def policy(self, ckpt):
        if ckpt not in self._pol:
            from dp_model import PolicyRunner
            self._pol[ckpt] = PolicyRunner(ckpt, seed=0)
        return self._pol[ckpt]


def vlm_score(gpu, start, end, lang):
    if not gpu.vlm:
        return None
    import b_judges as J
    return J.judge_d(start, end, lang)


def judge_frames(task, primary, wrist):
    """DINOv2 features (saved) and the oracle (a) logit: the fine-tuned CNN if its weights are present (E0b primary
    candidate), otherwise the logistic regression on the features (or None)."""
    import b_judges as J
    f = J.features(primary, wrist)
    a = J.judge_a_cnn(task, primary, wrist)
    return f, (a if a is not None else J.judge_a(task, f))


def do_imagine(job, gpu, root, stage):
    from b_cf import LIFT
    rp, _, sp = paths(root, stage, "imagine", job["job_id"])
    cf_rp, cf_ap, _ = paths(root, stage, "cf", job["dep"])
    cf = json.load(open(cf_rp))
    z = np.load(cf_ap)
    if "actions" not in z or "Slift_primary" not in z:
        write_json(rp, dict(job_id=job["job_id"], dep=job["dep"], skipped="no artifact"))
        return
    a = z["actions"]
    T = len(a)
    rec = dict(job_id=job["job_id"], dep=job["dep"], task=cf["task"], family=cf["family"], severity=cf["severity"],
               r_settled=cf["r_settled"], success_settled=cf["success_settled"], success_ever=cf["success_ever"], T=T)
    small = {}
    f_real, a_real = judge_frames(cf["task"], z["Slift_primary"], z["Slift_wrist"])
    small["feat_real"] = f_real
    rec["a_logit_real"] = a_real
    rec["d_real"] = vlm_score(gpu, z["S0_primary"], z["Slift_primary"], cf["language"])
    # judge (c) on the real final state: obs before the last chunk, real last chunk, real end state S_T
    if "Sk1_primary" in z and "ST_primary" in z:
        obs1 = {"primary_image": z["Sk1_primary"], "wrist_image": z["Sk1_wrist"], "proprio": z["Sk1_proprio"]}
        fut = {"primary_image": z["ST_primary"], "wrist_image": z["ST_wrist"], "proprio": z["ST_proprio"]}
        rec["v_real_k1"] = gpu.C.value_real_future(gpu.cfg, gpu.model, gpu.stats, obs1, cf["language"], a[T - 16:T], fut,
                                                   seed=gpu.cfg.seed, preprocessed=True)
    for k in (1, 2, 3, 4):
        if f"Sk{k}_primary" not in z:
            continue
        obs = {"primary_image": z[f"Sk{k}_primary"], "wrist_image": z[f"Sk{k}_wrist"], "proprio": z[f"Sk{k}_proprio"]}
        chunks = [a[T - 16 * k + 16 * i: T - 16 * k + 16 * (i + 1)] for i in range(k)] + [LIFT]
        t0 = time.time()
        outs = gpu.chain(obs, cf["language"], chunks)
        f_im, a_im = judge_frames(cf["task"], outs[-1]["future_image"], outs[-1]["future_wrist_image"])
        small[f"feat_imag_k{k}"] = f_im
        rec[f"k{k}"] = dict(v_end=outs[k - 1]["value"], v_lift=outs[-1]["value"], v_chunks=[o["value"] for o in outs],
                            a_logit=a_im, d=vlm_score(gpu, z["S0_primary"], outs[-1]["future_image"], cf["language"]),
                            sec=time.time() - t0)
    save_npz(sp, **small)
    write_json(rp, rec)


def do_imagine_ep(job, gpu, root, stage):
    from b_cf import LIFT
    rp, _, sp = paths(root, stage, "imagine_ep", job["job_id"])
    ep_rp, ep_ap, _ = paths(root, stage, "policy_ep", job["dep"])
    ep = json.load(open(ep_rp))
    z = np.load(ep_ap)
    lang, task = ep["language"], ep["task"]
    obs0 = {"primary_image": z["start_primary"], "wrist_image": z["start_wrist"], "proprio": z["start_proprio"]}
    rec = dict(job_id=job["job_id"], dep=job["dep"], task=task, policy=ep["policy"], success=ep["success"],
               r_settled=ep["r_settled"], success_settled=ep["success_settled"], n_chunks=ep["n_chunks"])
    small = {}
    f_real, a_real = judge_frames(task, z["lift_primary"], z["lift_wrist"])
    small["feat_real"], rec["a_logit_real"] = f_real, a_real
    rec["d_real"] = vlm_score(gpu, z["start_primary"], z["lift_primary"], lang)
    if "last_start_primary" in z:  # judge (c) on the real end state (value of the real final state after the last chunk)
        ls = {"primary_image": z["last_start_primary"], "wrist_image": z["last_start_wrist"], "proprio": z["last_start_proprio"]}
        fut = {"primary_image": z["end_primary"], "wrist_image": z["end_wrist"], "proprio": z["end_proprio"]}
        rec["v_real_end"] = gpu.C.value_real_future(gpu.cfg, gpu.model, gpu.stats, ls, lang, z["actions"][-1], fut,
                                                    seed=gpu.cfg.seed, preprocessed=True)
    if "O" in job.get("modes", ["O"]):
        t0 = time.time()
        outs = gpu.chain(obs0, lang, [c for c in z["actions"]] + [LIFT])
        f, al = judge_frames(task, outs[-1]["future_image"], outs[-1]["future_wrist_image"])
        small["feat_O"] = f
        rec["O"] = dict(v_end=outs[-2]["value"], v_lift=outs[-1]["value"], v_chunks=[o["value"] for o in outs], a_logit=al,
                        d=vlm_score(gpu, z["start_primary"], outs[-1]["future_image"], lang), sec=time.time() - t0)
    if "C" in job.get("modes", []) and ep["policy"].get("type") == "dp":
        t0 = time.time()
        pol = gpu.policy(ep["policy"]["ckpt"])
        obs, vals, pre = obs0, [], True
        for _ in range(job["H_C"]):
            ch = pol.act(obs["primary_image"], obs["wrist_image"], obs["proprio"])
            q = gpu.C.cf_query(gpu.cfg, gpu.model, gpu.stats, obs, lang, ch, seed=gpu.cfg.seed, n_act_steps=gpu.n_act,
                               preprocessed=pre)
            vals.append(q["value"])
            obs = {"primary_image": q["future_image"], "wrist_image": q["future_wrist_image"], "proprio": q["future_proprio"]}
        q = gpu.C.cf_query(gpu.cfg, gpu.model, gpu.stats, obs, lang, LIFT, seed=gpu.cfg.seed, n_act_steps=gpu.n_act,
                           preprocessed=True)
        f, al = judge_frames(task, q["future_image"], q["future_wrist_image"])
        small["feat_C"] = f
        rec["C"] = dict(v_end=vals[-1], v_lift=q["value"], v_chunks=vals + [q["value"]], a_logit=al, H_C=job["H_C"],
                        d=vlm_score(gpu, z["start_primary"], q["future_image"], lang), sec=time.time() - t0)
    save_npz(sp, **small)
    write_json(rp, rec)


# ============================================================== scheduler
def run(args):
    jobs = load_manifest(args.manifest)
    if args.shard:  # a GPU job goes to the shard of its CPU dependency (key = dep id), so both run on the same pod
        i, n = (int(x) for x in args.shard.split("/"))
        jobs = [j for j in jobs if zlib.crc32(j.get("dep", j["job_id"]).encode()) % n == i]
    done_ext = set(open(args.done).read().split()) if args.done and os.path.exists(args.done) else set()
    by_id = {j["job_id"]: j for j in jobs}

    def is_done(j):
        return j["job_id"] in done_ext or os.path.exists(paths(args.root, args.stage, j["kind"], j["job_id"])[0])

    pending_gpu = [j for j in jobs if j["kind"] in GPU_KINDS and not is_done(j)]
    need_dep = {j["dep"] for j in pending_gpu}
    cpu = []
    for j in jobs:
        if j["kind"] not in CPU_KINDS:
            continue
        rp, ap_, _ = paths(args.root, args.stage, j["kind"], j["job_id"])
        if not is_done(j):
            cpu.append((j, False))
        elif j["job_id"] in need_dep and not os.path.exists(ap_):
            cpu.append((j, True))  # done elsewhere but its artifact is needed here: regenerate (deterministic)
    print(f"[runner] stage {args.stage}: {len(cpu)} cpu jobs ({sum(f for _, f in cpu)} regenerations), {len(pending_gpu)} gpu jobs",
          flush=True)
    errors = []
    cpu_done = set()
    t_start = time.time()
    with ProcessPoolExecutor(args.cpu, mp_context=mp.get_context("spawn"), initializer=_worker_init) as ex:
        rk = RENDER_KEYS if args.cf_render == "all" else ()
        futs = {ex.submit(run_cpu_job, j, args.root, args.stage, f, rk): j["job_id"] for j, f in cpu}
        gpu = Gpu(vlm=args.vlm) if (pending_gpu and not args.no_gpu) else None
        queue = list(pending_gpu)
        while futs or (queue and gpu):
            if futs:
                fin, _ = wait(list(futs), timeout=0.5 if queue else None, return_when=FIRST_COMPLETED)
                for f in fin:
                    status, info = f.result()
                    jid = futs.pop(f)
                    if status == "ok":
                        cpu_done.add(jid)
                    else:
                        errors.append(info)
                        print("[runner] ERROR", info[:2000], flush=True)
            ready = [j for j in queue if os.path.exists(paths(args.root, args.stage, DEP[j["kind"]], j["dep"])[1])
                     and os.path.exists(paths(args.root, args.stage, DEP[j["kind"]], j["dep"])[0])]
            for j in (ready[:1] if futs else ready) if gpu else []:
                queue.remove(j)
                try:
                    (do_imagine if j["kind"] == "imagine" else do_imagine_ep)(j, gpu, args.root, args.stage)
                except Exception:
                    errors.append(j["job_id"] + "\n" + traceback.format_exc())
                    print("[runner] ERROR", errors[-1][:2000], flush=True)
            if not futs and queue and not ready:
                print(f"[runner] {len(queue)} gpu jobs without dependency artifacts; stopping", flush=True)
                break
    n_done = sum(os.path.exists(paths(args.root, args.stage, j["kind"], j["job_id"])[0]) for j in jobs)
    print(f"[runner] RUN_END {n_done}/{len(jobs)} jobs have results; errors {len(errors)}; {time.time() - t_start:.0f}s", flush=True)
    return 1 if errors else 0


def status(args):
    jobs = load_manifest(args.manifest)
    by = {}
    for j in jobs:
        d = os.path.exists(paths(args.root, args.stage, j["kind"], j["job_id"])[0])
        by.setdefault(j["kind"], [0, 0])
        by[j["kind"]][0] += 1
        by[j["kind"]][1] += d
    print(json.dumps({k: {"jobs": v[0], "done": v[1]} for k, v in by.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--cpu", type=int, default=6)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--done", default=None)
    ap.add_argument("--no_gpu", action="store_true")
    ap.add_argument("--vlm", action="store_true", help="also score frames with the VLM judge (d) (set B_VLM_CACHE)")
    ap.add_argument("--cf_render", choices=["all", "none"], default="all",
                    help="none: physics-only cf jobs (E1 candidate stage; r does not depend on rendering)")
    args = ap.parse_args()
    sys.exit(run(args) if args.cmd == "run" else status(args))
