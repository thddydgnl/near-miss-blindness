"""Continuous success margin r = e / tau for LIBERO goal predicates, from simulator state (B design 4.1).

r < 1 exactly when the geometric part of the goal predicate holds. The non-geometric conditions of the same
predicate (contact, vertical band / ordering) are returned as aux_ok, so success_geom = (r < 1) and aux_ok
reproduces LIBERO's own predicate (checked against env._check_success() in test_margin.py).
Definitions (thresholds are LIBERO's own):
  On(item, object)  [ObjectState.check_ontop]   e = |xy_item - xy_base|, tau = 0.03 m;       aux: z_base <= z_item, contact
  On(item, site)    [SiteObject.under]          d = R_site (p_item - p_site); r = max(|d_x|/s_x, |d_y|/s_y)
                                                (e/tau per horizontal axis, L-inf);           aux: s_z-0.005 < d_z < s_z+0.10, contact(parent, item)
  In(item, site)    [SiteObject.in_box]         per-axis distance to the box face, normalized by the half-extent
                                                (z face lowered by 0.01 m as in LIBERO), L-inf over x, y, z
  Open(site/object) [is_open, e.g. WoodenCabinet q < -0.14]   e = |q - q_full|, tau = |q* - q_full|, q_full = the joint
                                                limit on the open side (drawer: q_full = -0.16, q* = -0.14, tau = 0.02 m)
  TurnOn(object)    [FlatStove q >= 0.5]        same form with q_full = 2.1 (upper limit): an untouched knob already has
                                                r = 1.31, so this predicate has no usable far side (excluded from M2 tasks)
For conjunctions r = max over predicates. Episode-level r is taken at the final (settled) state.
"""
from __future__ import annotations

import numpy as np

OBJ_ON_XY_TOL = 0.03


def _dom(env):
    """OffScreenRenderEnv / ControlEnv -> the BDDL problem env."""
    return env.env if hasattr(env, "env") and hasattr(env.env, "parsed_problem") else env


def _joint_q(sim, j):
    return float(sim.data.qpos[sim.model.get_joint_qpos_addr(j)]), np.asarray(sim.model.jnt_range[sim.model.joint_name2id(j)], float)


def predicate_margin(dom, state) -> dict:
    sim = dom.sim
    name = state[0].lower()
    if name in ("on", "in"):
        item, tgt = state[1], state[2]
        ts, it = dom.object_states_dict[tgt], dom.object_states_dict[item]
        ipos = np.array(sim.data.body_xpos[dom.obj_body_id[item]])
        if ts.object_state_type == "site":
            site = dom.object_sites_dict[tgt]
            spos = np.array(sim.data.get_site_xpos(tgt))
            smat = np.array(sim.data.get_site_xmat(tgt))
            size = np.asarray(site.size, float)
            if name == "on":
                d = smat @ (ipos - spos)  # same expression as SiteObject.under
                ratios = np.abs(d[:2]) / size[:2]
                r = float(np.max(ratios))
                vert_ok = bool(size[2] - 0.005 < d[2] < size[2] + 0.10)
                parent = dom.get_object(ts.parent_name)
                contact_ok = True if parent is None else bool(dom.check_contact(parent, dom.get_object(item)))
                return dict(pred=f"on_site:{tgt}", r=r, e=float(np.abs(d[:2])[np.argmax(ratios)]),
                            tau=float(size[:2][np.argmax(ratios)]), aux_ok=vert_ok and contact_ok,
                            vert_ok=vert_ok, contact_ok=contact_ok, d=d.tolist())
            total = np.abs(smat @ size)  # same as SiteObject.in_box
            ub, lb = spos + total, spos - total
            lb[2] -= 0.01
            ratios = np.where(ipos >= spos, (ipos - spos) / np.maximum(ub - spos, 1e-9),
                              (spos - ipos) / np.maximum(spos - lb, 1e-9))
            r = float(np.max(ratios))
            return dict(pred=f"in_site:{tgt}", r=r, e=None, tau=None, aux_ok=bool(ts.check_contact(it)),
                        ratios=ratios.tolist())
        bpos = np.array(sim.data.body_xpos[dom.obj_body_id[tgt]])
        e = float(np.linalg.norm(ipos[:2] - bpos[:2]))
        z_ok = bool(bpos[2] <= ipos[2])
        contact_ok = bool(ts.check_contact(it))
        if name == "on":
            return dict(pred=f"on_obj:{tgt}", r=e / OBJ_ON_XY_TOL, e=e, tau=OBJ_ON_XY_TOL, aux_ok=z_ok and contact_ok,
                        z_ok=z_ok, contact_ok=contact_ok)
        raise NotImplementedError(f"In(object, object) not used in the selected tasks: {state}")
    if name in ("open", "close", "turnon", "turnoff"):
        obj = state[1]
        st = dom.object_states_dict[obj]
        if st.object_state_type == "site":
            joints, art = dom.object_sites_dict[obj].joints, dom.get_object(st.parent_name)
        else:
            joints, art = dom.get_object(obj).joints, dom.get_object(obj)
        props = art.object_properties["articulation"]
        rs = []
        for j in joints:
            q, (lo, hi) = _joint_q(sim, j)
            if name in ("open", "close"):
                open_lo = bool(art.is_open(lo))  # which joint limit is the open side
                q_full = lo if open_lo else hi
                if name == "open":
                    q_star = max(props["default_open_ranges"]) if open_lo else min(props["default_open_ranges"])
                else:
                    q_full = hi if open_lo else lo
                    q_star = min(props["default_close_ranges"]) if open_lo else max(props["default_close_ranges"])
            else:
                q_full = hi if name == "turnon" else lo
                q_star = min(props["default_turnon_ranges"]) if name == "turnon" else max(props["default_turnoff_ranges"])
            tau = abs(q_star - q_full)
            rs.append(dict(joint=j, q=q, q_full=q_full, q_star=q_star, r=abs(q - q_full) / tau if tau > 0 else np.inf,
                           side_ok=bool((q - q_star) * (q_full - q_star) > 0)))
        # Open/TurnOn: any joint suffices; Close/TurnOff: all joints must hold
        pick = min(rs, key=lambda x: x["r"]) if name in ("open", "turnon") else max(rs, key=lambda x: x["r"])
        r = pick["r"] if pick["side_ok"] else max(pick["r"], 1.0)
        return dict(pred=f"{name}:{obj}", r=float(r), e=abs(pick["q"] - pick["q_full"]),
                    tau=abs(pick["q_star"] - pick["q_full"]), aux_ok=True, joints=rs)
    raise NotImplementedError(state)


def task_margin(env) -> dict:
    """Margin of the full goal (conjunction): r = max over predicates; also returns LIBERO's own success flag."""
    dom = _dom(env)
    parts = [predicate_margin(dom, s) for s in dom.parsed_problem["goal_state"]]
    r = max(p["r"] for p in parts)
    aux = all(p["aux_ok"] for p in parts)
    return dict(r=float(r), aux_ok=bool(aux), success_geom=bool(r < 1.0 and aux),
                success_libero=bool(dom._check_success()), parts=parts)


def set_flat_state(env, flat_state):
    dom = _dom(env)
    dom.sim.set_state_from_flattened(np.asarray(flat_state))
    dom.sim.forward()
