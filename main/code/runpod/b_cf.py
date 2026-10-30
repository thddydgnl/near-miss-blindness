"""B counterfactual generator for M2 (E0b calibration / judge data, and the E1 job kind of b_runner.py).

One counterfactual = an official LIBERO-Goal demo, one perturbation family, one severity, one direction draw.
Families: CureWM's six (unmodified, imported from curewm.perturbations) + B's boundary-targeted ones:
  place_offset(delta)    constant horizontal offset over the whole carry phase, sized by the measured OSC gain so that
                         the realized end-effector displacement at release is ~delta metres, direction theta
  under_actuation(s)     translation actions after the first grasp (or the second half) scaled by (1 - s)
  pull_truncate(s)       translation stopped for the last fraction s of the pull window (drawer task; demos never
                         close the gripper, so the window is the second half of the demo)
Randomness: theta and CureWM's rng come from a stable CRC32 of (task, demo, family, dir) (common random numbers across
severity, like CureWM, but independent of PYTHONHASHSEED).
Rollout protocol (physics only, ControlEnv): execute the T perturbed actions (LIBERO success at any step recorded as
success_ever); from the end state S_T
  settle branch: 15 steps zero motion, gripper open -> r_settled, success_settled   (r is defined here)
  lift branch:   the scripted 16-step chunk LIFT (3 steps open, 13 steps +z) -> success_lift, r_lift, judge frames
Saved states: S_0 (initial), S_{T-16k} for k = 1..4 (start of end-aligned imagination), S_T, S_lift.
Rendering (OffScreenRenderEnv 256 px, separate env): S_0 and S_lift (third-person + wrist), converted to the Cosmos
model view (flip, JPEG95, 224, 90% crop -> 224); S_{T-16k} on request (E1 imagination).
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np

LIFT = np.array([[0, 0, 0, 0, 0, 0, -1]] * 3 + [[0, 0, 1.0, 0, 0, 0, -1]] * 13, dtype=np.float64)
SETTLE = np.array([[0, 0, 0, 0, 0, 0, -1]] * 15, dtype=np.float64)
G_EFF = 0.010
B_FAMILIES = ("place_offset", "under_actuation", "pull_truncate")
TASK_LANG = {}


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def to_libero(a_engine: np.ndarray) -> np.ndarray:
    a = np.asarray(a_engine, dtype=np.float64).copy()
    a[:, 6] = 1.0 - 2.0 * a[:, 6]
    return a


class Generator:
    """Per-process generator for one task: physics env (CureWM backend, no rendering) + render env."""

    def __init__(self, task: int, render: bool = True):
        from curewm.backends.libero import LiberoBackend
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self.task = task
        self.be = LiberoBackend("libero_goal", render=False)
        b = benchmark.get_benchmark_dict()["libero_goal"]()
        t = b.get_task(task)
        self.language = t.language
        self.renv = None
        if render:
            self.renv = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                                           camera_heights=256, camera_widths=256)
            self.renv.seed(0)
            self.renv.reset()
        self._nominal = {}
        self._cfg = None

    # ------------------------------------------------------------------ demos and families
    def seeded_reset(self, d):
        # the env must exist before seeding: its constructor draws from the RNG too (first reset would differ)
        self.be._ensure_env(int(d.init_state["task_id"]))
        seed = stable_seed("fixtures", self.task, d.meta["demo"])
        np.random.seed(seed)
        self.be._env.seed(seed)
        self.be.reset_to(d.init_state)

    def demo(self, di: int):
        from curewm.backends.libero import load_libero_demos
        if di not in self._nominal:
            d = load_libero_demos("libero_goal", di + 1, task_ids=[self.task])[di]
            self.seeded_reset(d)
            nom = self.be.rollout(d.actions)
            ok = bool(nom["success"])
            if ok:
                from curewm.perturbations import annotate_phases
                d.contacts = nom["contacts"]
                annotate_phases(d)
            self._nominal[di] = (d, ok)
        return self._nominal[di]

    def perturb(self, d, family: str, s: float, k: int) -> tuple[np.ndarray, float | None]:
        """Return perturbed engine-convention actions (grip in [0 closed, 1 open]) and theta (place_offset)."""
        from curewm.perturbations import ALL_FAMILIES, Phase
        rng = np.random.default_rng(stable_seed(self.task, d.meta["demo"], family, k))
        a = d.actions.copy()
        theta = None
        if family == "nominal":
            return a, None
        if family not in B_FAMILIES:
            fam = {f.family: f for f in ALL_FAMILIES}[family]
            return fam.apply(d, s, rng), None
        ph = d.phases
        carry = np.where(ph == Phase.CARRY.value)[0]
        if family == "place_offset":
            theta = float(rng.uniform(0, 2 * np.pi))
            if len(carry) < 2:
                raise ValueError("no carry phase")
            u = s / (len(carry) * G_EFF)
            a[carry, 0] += u * np.cos(theta)
            a[carry, 1] += u * np.sin(theta)
            a[:, :6] = np.clip(a[:, :6], -1.0, 1.0)
        elif family == "under_actuation":
            grasp_on = np.where(ph >= Phase.GRASP.value)[0]
            start = int(grasp_on[0]) if len(grasp_on) else len(a) // 2
            a[start:, 0:3] *= (1.0 - s)
        elif family == "pull_truncate":
            w0, w1 = (int(carry[0]), int(carry[-1]) + 1) if len(carry) >= 2 else (len(a) // 2, len(a))
            cut = w0 + int(round((1.0 - s) * (w1 - w0)))
            a[cut:w1, 0:3] = 0.0
        return a, theta

    # ------------------------------------------------------------------ rollout
    def _sim(self):
        return self.be._sim

    def rollout(self, d, a_engine: np.ndarray) -> dict:
        import libero_margin as M
        env = self.be._env
        a = to_libero(a_engine)
        T = len(a)
        key_t = {T - 16 * k: k for k in (1, 2, 3, 4) if T - 16 * k >= 1}
        self.seeded_reset(d)
        states = {"S0": np.asarray(self._sim().get_state().flatten())}
        success_ever = False
        for t in range(T):
            if t in key_t:
                states[f"Sk{key_t[t]}"] = np.asarray(self._sim().get_state().flatten())
            env.step(a[t])
            success_ever = success_ever or bool(env.check_success())
        sT = np.asarray(self._sim().get_state().flatten())
        states["ST"] = sT
        for x in SETTLE:
            env.step(x)
        ms = M.task_margin(env)
        M.set_flat_state(env, sT)
        for x in LIFT:
            env.step(x)
        ml = M.task_margin(env)
        states["Slift"] = np.asarray(self._sim().get_state().flatten())
        return dict(T=T, success_ever=success_ever, r_settled=ms["r"], success_settled=ms["success_libero"],
                    aux_settled=ms["aux_ok"], r_lift=ml["r"], success_lift=ml["success_libero"],
                    lift_changes_outcome=bool(ml["success_libero"] != ms["success_libero"]), pred=ms["parts"][0]["pred"],
                    actions_libero=a, states=states)

    # ------------------------------------------------------------------ rendering (Cosmos model view)
    def render(self, flat_state) -> dict:
        import cp_common as C
        if self._cfg is None:
            self._cfg = C.make_cfg("eval")
        src, dst = self._sim().model, self.renv.env.sim.model  # same task model: copy the fixture placement
        dst.body_pos[:] = src.body_pos
        dst.body_quat[:] = src.body_quat
        obs = self.renv.set_init_state(np.asarray(flat_state))
        o = C.env_obs_to_policy_obs(obs, flip=True)
        p224 = C.CU.prepare_images_for_model([o["primary_image"], o["wrist_image"]], self._cfg)
        return {"primary": p224[0], "wrist": p224[1], "proprio": o["proprio"].astype(np.float32),
                "primary_raw": o["primary_image"], "wrist_raw": o["wrist_image"]}

    def close(self):
        self.be.close()
        if self.renv is not None:
            self.renv.close()


def run_cell(gen: Generator, di: int, family: str, s: float, k: int, out_dir: str, render_keys=("S0", "Slift"),
             save_actions: bool = True) -> dict:
    """Generate one counterfactual; writes <out_dir>/<cf_id>.json and .npz atomically; returns the JSON record."""
    cf_id = f"t{gen.task}_d{di:02d}_{family}_s{s:.4f}_k{k}"
    jp = os.path.join(out_dir, cf_id + ".json")
    if os.path.exists(jp):
        return json.load(open(jp))
    d, ok = gen.demo(di)
    rec = dict(cf_id=cf_id, task=gen.task, demo=di, family=family, severity=float(s), dir=k, language=gen.language,
               nominal_replay_ok=ok)
    if not ok:
        rec["skipped"] = "nominal replay failed"
    else:
        try:
            a, theta = gen.perturb(d, family, s, k)
        except ValueError as e:
            rec["skipped"] = str(e)
            a = None
        if a is not None:
            res = gen.rollout(d, a)
            rec.update({k2: v for k2, v in res.items() if k2 not in ("actions_libero", "states")}, theta=theta)
            arrays = {f"state_{n}": v for n, v in res["states"].items()}
            if save_actions:
                arrays["actions"] = res["actions_libero"].astype(np.float32)
            if gen.renv is not None:
                for n in render_keys:
                    if n in res["states"]:
                        im = gen.render(res["states"][n])
                        arrays[f"{n}_primary"] = im["primary"]
                        arrays[f"{n}_wrist"] = im["wrist"]
                        arrays[f"{n}_proprio"] = im["proprio"]
            tmp = os.path.join(out_dir, f"{cf_id}.tmp{os.getpid()}.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, os.path.join(out_dir, cf_id + ".npz"))
    tmpj = jp + f".tmp{os.getpid()}"
    with open(tmpj, "w") as f:
        json.dump(rec, f)
    os.replace(tmpj, jp)
    return rec
