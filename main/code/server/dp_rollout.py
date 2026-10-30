"""B E0: roll out Diffusion Policy snapshots (EMA weights) in the DINO-WM PushT env (the env the DINO-WM checkpoint was
trained on; physics verified identical to the diffusion_policy env, dp_conventions.py C3).

Protocol (DP-native, as diffusion_policy PushTKeypointsRunner): init state from the episode seed with DP's formula
(RandomState(seed): agent xy ~ U{50..449}, block xy ~ U{100..399}, angle ~ 2*pi*N(0,1) - pi; zero velocity),
n_obs_steps 2 (history padded with the first obs), n_action_steps 8, DDPM 100 steps, stop when coverage > 0.95
(success) or after 300 control steps. Low-dim obs = 9 block keypoints (DP's own keypoint map) + agent xy.

Action conversion (the key integration step): DP outputs absolute PD targets A_t (512 px world coords). The DINO-WM env
takes relative actions a_t in env units with PD target = agent_pos_t + 100 * a_t, so a_t = (A_t - agent_pos_t) / 100
reproduces A_t exactly. This is also the convention of the DINO-WM training data (rel_actions = abs_actions - agent_pos,
C4) and its action normalisation stats are exactly the statistics of the standard demos (C6). The recorded a_t are what
the DINO-WM imagination replays (mode O).

Per episode saved: seed, init state (7), a_rel (T,2), A (T,2), states (T+1,7) (float64, before/after each step),
coverage (T,), T_end, success_dp (coverage > 0.95 reached), final r components vs the fixed goal pose (256,256,pi/4).
usage: python dp_rollout.py --ckpt <ema_epXXXX.pt> --seeds 100000:100100 --out <file.pt> [--batch 100]"""
import argparse, os, sys, time, json
import numpy as np
import torch
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r21lib as L
import r12lib as R
sys.path.insert(0, BP.DP_REPO)

GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])
MAX_STEPS, N_OBS, N_ACT, COV_THR = 300, 2, 8, 0.95


def load_policy(ckpt, device):
    import hydra
    from train_dp import load_cfg
    cfg = load_cfg()
    pol = hydra.utils.instantiate(cfg.policy)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['ema_model']
    pol.load_state_dict(sd)
    pol.to(device).eval()
    return pol, cfg


_KPM = {}


def kp_manager():
    if 'm' not in _KPM:
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
        _KPM['m'] = PymunkKeypointManager(**PushTKeypointsEnv.genenerate_keypoint_manager_params())
    return _KPM['m']


def dp_init_state(seed):
    rs = np.random.RandomState(seed=seed)
    return np.array([rs.randint(50, 450), rs.randint(50, 450), rs.randint(100, 400), rs.randint(100, 400),
                     rs.randn() * 2 * np.pi - np.pi, 0.0, 0.0], dtype=np.float64)


def state7(env):
    return np.array(tuple(env.agent.position) + tuple(env.block.position) + (env.block.angle % (2 * np.pi),)
                    + tuple(env.agent.velocity), dtype=np.float64)


def obs20(env, kpm):
    tf = kpm.get_tf_img((env.block.position[0], env.block.position[1], env.block.angle))
    kps = tf(kpm.local_keypoint_map['block'])
    return np.concatenate([kps.reshape(-1), np.array(env.agent.position)]).astype(np.float32)


def step_rel(env, a):
    """One control step of the DINO-WM env with relative action a (env units): identical to PushTEnv.step physics."""
    from pymunk.vec2d import Vec2d
    tgt = env.agent.position + np.asarray(a, np.float64) * env.action_scale
    dt = 1.0 / env.sim_hz
    for _ in range(env.sim_hz // env.control_hz):
        acc = env.k_p * (tgt - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
        env.agent.velocity += acc * dt
        env.space.step(dt)


def final_parts(st):
    """r components vs the fixed goal pose (object pose only; the standard task has no agent goal)."""
    block = float(np.linalg.norm(st[2:4] - GOAL_POSE[:2]))
    ang = abs(st[4] - GOAL_POSE[2]) % (2 * np.pi)
    ang = float(min(ang, 2 * np.pi - ang))
    return dict(block=block, ang=ang, r_obj=max(block / R.OBJ_POS_THR, ang / R.ANG_THR), r_obj_pos=block / R.OBJ_POS_THR,
                r_ang=ang / R.ANG_THR)


@torch.no_grad()
def rollout(policy, seeds, device, batch=100, torch_seed=0, log=print):
    L.setup_paths()
    kpm = kp_manager()
    out = []
    for b0 in range(0, len(seeds), batch):
        sb = seeds[b0:b0 + batch]
        envs = [L.make_env('pusht') for _ in sb]
        torch.manual_seed(torch_seed + b0)
        recs = []
        from env.pusht.pusht_env import pymunk_to_shapely
        for e, s in zip(envs, sb):
            st0 = dp_init_state(int(s))
            R.sim_reset(e, 'T', st0)
            gb = e._get_goal_pose_body(e.goal_pose)
            gg = pymunk_to_shapely(gb, e.block.shapes)
            recs.append(dict(seed=int(s), init=state7(e), states=[state7(e)], A=[], a_rel=[], cov=[], done=False,
                             obs=[obs20(e, kpm)], gg=gg, ga=gg.area))
        t0 = time.time()
        while True:
            act_idx = [i for i, r in enumerate(recs) if not r['done']]
            if not act_idx:
                break
            ob = np.stack([np.stack([recs[i]['obs'][max(0, len(recs[i]['obs']) - N_OBS + k)] if len(recs[i]['obs']) >= N_OBS
                                     else recs[i]['obs'][0] for k in range(N_OBS)]) for i in act_idx])
            res = policy.predict_action({'obs': torch.as_tensor(ob, device=device)})
            acts = res['action'].float().cpu().numpy().astype(np.float64)          # (n, 8, 2) absolute targets
            for j, i in enumerate(act_idx):
                r, e = recs[i], envs[i]
                for A in acts[j]:
                    a = (A - np.array(e.agent.position)) / e.action_scale
                    step_rel(e, a)
                    r['A'].append(A)
                    r['a_rel'].append(a)
                    r['states'].append(state7(e))
                    c = float(r['gg'].intersection(pymunk_to_shapely(e.block, e.block.shapes)).area / r['ga'])
                    r['cov'].append(c)
                    r['obs'].append(obs20(e, kpm))
                    if c > COV_THR or len(r['A']) >= MAX_STEPS:
                        r['done'] = True
                        break
        for r in recs:
            T = len(r['A'])
            fin = r['states'][-1]
            fp = final_parts(fin)
            out.append(dict(seed=r['seed'], init=r['init'], T_end=T, a_rel=np.array(r['a_rel']), A=np.array(r['A']),
                            states=np.array(r['states']), cov=np.array(r['cov']),
                            success_dp=bool(max(r['cov']) > COV_THR), max_cov=float(max(r['cov'])),
                            final_cov=float(r['cov'][-1]), score=float(np.clip(max(r['cov']) / COV_THR, 0, 1)), **fp))
        log(f'  batch {b0}-{b0 + len(sb)}: {time.time() - t0:.1f}s, success_dp {np.mean([o["success_dp"] for o in out[-len(sb):]]):.2f}')
    return out


def summarize(eps):
    r = np.array([e['r_obj'] for e in eps])
    T = np.array([e['T_end'] for e in eps])
    return dict(n=len(eps), success_dp=float(np.mean([e['success_dp'] for e in eps])),
                mean_score=float(np.mean([e['score'] for e in eps])),
                success_robj=float((r < 1).mean()), near_robj=float(((r >= 1) & (r < 2)).mean()),
                mid_robj=float(((r >= 2) & (r < 3)).mean()), far_robj=float((r >= 3).mean()),
                r_obj_q=np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                T_end_q=np.quantile(T, [0.1, 0.5, 0.9]).tolist(), timeout=float((T >= MAX_STEPS).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--seeds', required=True, help='a:b range')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=100)
    args = ap.parse_args()
    if os.path.exists(args.out):
        print('exists', args.out)
        return
    a, b = [int(x) for x in args.seeds.split(':')]
    device = torch.device('cuda:0')
    t0 = time.time()
    pol, _ = load_policy(args.ckpt, device)
    eps = rollout(pol, list(range(a, b)), device, batch=args.batch)
    summ = summarize(eps)
    summ.update(ckpt=args.ckpt, seeds=args.seeds, sec=time.time() - t0)
    torch.save(dict(episodes=eps, summary=summ), args.out + '.tmp')
    os.replace(args.out + '.tmp', args.out)
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
