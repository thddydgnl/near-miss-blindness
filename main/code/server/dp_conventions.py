"""B E0: action / observation conventions between the DINO-WM PushT env (what the DINO-WM checkpoint was trained on),
the diffusion_policy PushT env, the standard PushT demos (pusht_cchi_v7_replay.zarr) and the DINO-WM dataset
(pusht_noise). Writes results/<stage>/dp_conventions.json. Sim + numpy only (no GPU).

C1  standard demos: episodes / steps / keys
C2  DP low-dim observation: keypoints recomputed with DP's own keypoint manager from the demo block pose == zarr keypoints
C3  physics: DP env stepping absolute targets A_t  vs  DINO-WM env stepping a_t = (A_t - agent_pos_t) / 100
    (relative=True, action_scale=100 -> PD target = agent_pos_t + 100 a_t = A_t), same init state, demo action sequences
C4  DINO-WM dataset convention: rel_actions vs abs_actions - agent_pos (states[..., :2]); rel action scale
C5  round trip: the recorded relative actions replayed with the fast replay used for imagination (r12lib.sim_run)
    reproduce the recorded final state
C6  action distribution: DP demo relative displacements vs DINO-WM training rel_actions (env units)
usage: python dp_conventions.py"""
import os, sys, json, time
import numpy as np
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import r21lib as L

sys.path.insert(0, BP.DP_REPO)


def load_demos():
    import zarr
    root = zarr.open(BP.DP_DATA, 'r')
    d = {k: root['data'][k][:] for k in ('state', 'action', 'keypoint', 'n_contacts')}
    ends = root['meta']['episode_ends'][:]
    return d, ends


def dp_kp_manager():
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
    kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    return PymunkKeypointManager(**kw), kw


def kp_from_pose(kpm, pose):
    """DP global block keypoints (9,2) from block pose (x, y, angle) -- same as PymunkKeypointManager.get_keypoints_global."""
    tf = kpm.get_tf_img(pose)
    return tf(kpm.local_keypoint_map['block'])


def dp_env_run(state0, acts, legacy):
    from diffusion_policy.env.pusht.pusht_env import PushTEnv as DPEnv
    env = DPEnv(legacy=legacy, render_action=False)
    env.seed(0)
    env.reset_to_state = np.asarray(state0, np.float64)
    env.reset()
    sts = [np.array(tuple(env.agent.position) + tuple(env.block.position) + (env.block.angle % (2 * np.pi),))]
    for a in acts:
        env.step(np.asarray(a, np.float64))
        sts.append(np.array(tuple(env.agent.position) + tuple(env.block.position) + (env.block.angle % (2 * np.pi),)))
    return np.stack(sts)


def dinowm_env_run(env, state0, acts, legacy):
    """DINO-WM env, relative actions computed on line from the absolute targets. Returns states (T+1, 7) and rel acts."""
    from pymunk.vec2d import Vec2d
    env.legacy = legacy
    s7 = np.concatenate([np.asarray(state0, np.float64)[:5], [0.0, 0.0]])
    R.sim_reset(env, 'T', s7)
    dt = 1.0 / env.sim_hz
    n_sub = env.sim_hz // env.control_hz
    sts = [np.array(tuple(env.agent.position) + tuple(env.block.position) + (env.block.angle % (2 * np.pi),)
                    + tuple(env.agent.velocity))]
    rel = []
    for A in acts:
        a = (np.asarray(A, np.float64) - np.array(env.agent.position)) / env.action_scale   # DINO-WM env units
        rel.append(a)
        tgt = np.array(a) * env.action_scale
        tgt = env.agent.position + tgt                                                   # = A (relative=True)
        for _ in range(n_sub):
            acc = env.k_p * (tgt - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acc * dt
            env.space.step(dt)
        sts.append(np.array(tuple(env.agent.position) + tuple(env.block.position) + (env.block.angle % (2 * np.pi),)
                            + tuple(env.agent.velocity)))
    env.legacy = False
    return np.stack(sts), np.stack(rel)


def angdiff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def main():
    t0 = time.time()
    out = {}
    d, ends = load_demos()
    starts = np.concatenate([[0], ends[:-1]])
    out['C1'] = dict(n_episodes=int(len(ends)), n_steps=int(ends[-1]), keys=sorted(d.keys()),
                     shapes={k: list(v.shape) for k, v in d.items()},
                     ep_len=dict(min=int((ends - starts).min()), median=float(np.median(ends - starts)),
                                 max=int((ends - starts).max())))
    print('C1', out['C1'], flush=True)
    # C2 keypoints
    kpm, kw = dp_kp_manager()
    idx = np.random.RandomState(0).choice(len(d['state']), 2000, replace=False)
    kps = np.stack([kp_from_pose(kpm, d['state'][i, 2:5]) for i in idx])
    zk = d['keypoint'][idx]
    out['C2'] = dict(n=int(len(idx)), max_abs_diff_px=float(np.abs(kps - zk).max()),
                     mean_abs_diff_px=float(np.abs(kps - zk).mean()),
                     local_keypoints_block=kw['local_keypoint_map']['block'].tolist())
    print('C2 keypoints recomputed vs zarr: max |diff| %.2e px' % out['C2']['max_abs_diff_px'], flush=True)
    # C3 physics equivalence on demo episodes (absolute demo actions)
    env = R.get_env()
    rows = []
    for ep in range(0, len(ends), 10):          # 21 demo episodes
        s, e = starts[ep], ends[ep]
        st0, acts = d['state'][s], d['action'][s:e]
        row = dict(ep=int(ep), T=int(e - s))
        for leg in (False, True):
            a = dp_env_run(st0, acts, leg)
            b, rel = dinowm_env_run(env, st0, acts, leg)
            row[f'legacy{int(leg)}_max_state_diff'] = float(np.abs(a[:, :4] - b[:, :4]).max())
            row[f'legacy{int(leg)}_max_angle_diff'] = float(angdiff(a[:, 4], b[:, 4]).max())
            # demo states vs re-simulation (fidelity of the recorded demo states; zarr state[t] is the obs before action t)
            row[f'legacy{int(leg)}_vs_zarr_max_pos'] = float(np.abs(a[:-1, :4] - d['state'][s:e, :4]).max())
            row[f'legacy{int(leg)}_vs_zarr_final_pos'] = float(np.abs(a[-2, :4] - d['state'][e - 1, :4]).max())
        # C5: replay of the recorded relative actions with the fast replay path used for imagination
        b, rel = dinowm_env_run(env, st0, acts, False)
        s7 = np.concatenate([st0[:5], [0.0, 0.0]])
        fin, _ = R.sim_run(env, 'T', s7, rel, render=False)
        row['C5_replay_final_diff'] = float(np.abs(fin[:4] - b[-1, :4]).max())
        rows.append(row)
        print('C3/C5', row, flush=True)
    out['C3_C5'] = dict(rows=rows,
                        max_state_diff_legacy0=max(r['legacy0_max_state_diff'] for r in rows),
                        max_state_diff_legacy1=max(r['legacy1_max_state_diff'] for r in rows),
                        max_C5_replay_diff=max(r['C5_replay_final_diff'] for r in rows))
    # C4 DINO-WM dataset convention
    import torch
    root = BP.DS_DIR + '/pusht_noise'
    C4 = {}
    for split in ('train', 'val'):
        st = torch.load(f'{root}/{split}/states.pth').numpy()
        ra = torch.load(f'{root}/{split}/rel_actions.pth').numpy()
        aa = torch.load(f'{root}/{split}/abs_actions.pth').numpy()
        import pickle
        lens = pickle.load(open(f'{root}/{split}/seq_lengths.pkl', 'rb'))
        n = min(len(lens), 300)
        e1, e2, e3, cnt = [], [], [], 0
        relall = []
        for i in range(n):
            T = lens[i]
            e1.append(np.abs(ra[i, :T] - (aa[i, :T] - st[i, :T, :2])).max())                       # target - agent pos
            prev = np.concatenate([st[i, :1, :2], aa[i, :T - 1]], 0)
            e2.append(np.abs(ra[i, :T] - (aa[i, :T] - prev)).max())                               # target - previous target
            relall.append(ra[i, :T])
            cnt += T
        relall = np.concatenate(relall) / 100.0
        C4[split] = dict(n_traj=int(len(lens)), n_checked=int(n), steps_checked=int(cnt),
                         max_err_rel_eq_abs_minus_agentpos=float(np.max(e1)),
                         max_err_rel_eq_abs_minus_prevtarget=float(np.max(e2)),
                         rel_env_units_mean=relall.mean(0).tolist(), rel_env_units_std=relall.std(0).tolist(),
                         rel_env_units_absq=np.quantile(np.abs(relall), [0.5, 0.9, 0.99]).tolist())
        print('C4', split, C4[split], flush=True)
    out['C4'] = C4
    # C6 DP demo relative displacements (target - agent pos) in DINO-WM env units
    reld = (d['action'] - d['state'][:, :2]) / 100.0
    out['C6'] = dict(dp_rel_env_units_mean=reld.mean(0).tolist(), dp_rel_env_units_std=reld.std(0).tolist(),
                     dp_rel_env_units_absq=np.quantile(np.abs(reld), [0.5, 0.9, 0.99]).tolist(),
                     dinowm_action_mean=[-0.0087, 0.0068], dinowm_action_std=[0.2019, 0.2002])
    print('C6', out['C6'], flush=True)
    out['time_s'] = time.time() - t0
    L.jdump(out, f'{BP.RES}/dp_conventions.json')
    print('CONVENTIONS_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
