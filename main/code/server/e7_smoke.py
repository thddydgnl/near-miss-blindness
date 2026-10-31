"""B E7 smoke test on the pod (before the chain; a few seconds of GPU): PointMaze env + dataset loader, replay exactness
against the recorded dataset states (20 trajectories x 25 steps, dataset actions denormalised as the pipeline does),
render shape / time, episode sampling from TrajSubsets (base-index mapping), one encode + imagine, margin_parts.
usage: python e7_smoke.py   (writes results/<stage>/pm/smoke_e7.json; prints SMOKE_E7_DONE)"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E


def main():
    t0 = time.time()
    assert BP.TASK == 'pointmaze', BP.TASK
    out = dict(task=BP.TASK, place=BP.PLACE, torch=torch.__version__, cuda=torch.cuda.is_available(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    env = R.get_env()
    out['env'] = dict(cls=type(env).__name__, frame_skip=int(env.frame_skip), action_space=str(env.action_space))
    D = R.load_traj_dsets()
    dt, dv = D['train'], D['valid']
    out['split'] = dict(train=len(dt), valid=len(dv), seq_len=int(dt.get_seq_length(0)), valid_base_head=[int(i) for i in dv.indices[:8]])
    # replay exactness vs the recorded states (raw dataset tensors of the base dataset)
    base = dv.dataset
    S, A = base.states, base.actions
    amean, astd = base.action_mean.numpy(), base.action_std.numpy()
    errs = []
    for li in range(0, len(dv), max(1, len(dv) // 20)):
        bi = dv.indices[li]
        acts = A[bi, :25].numpy() * astd + amean
        st, _ = R.sim_run(env, None, S[bi, 0].numpy().astype(np.float64), acts, render=False)
        errs.append(float(np.linalg.norm(st[:2] - S[bi, 25].numpy()[:2])))
    out['replay_xy_err'] = dict(n=len(errs), max=max(errs), median=float(np.median(errs)))
    t1 = time.time()
    st, img = R.sim_run(env, None, S[dv.indices[0], 0].numpy().astype(np.float64), A[dv.indices[0], :25].numpy() * astd + amean, render=True)
    out['render'] = dict(shape=list(np.asarray(img).shape), dtype=str(np.asarray(img).dtype), ms=1000 * (time.time() - t1), mean=float(np.mean(img)))
    # episode sampling + margins
    eps = E.sample_episodes_from(dv, list(range(len(dv))), 4, 5, seed=1)
    out['episode'] = dict(traj=[e['traj'] for e in eps], traj_local=[e['traj_local'] for e in eps], offset=[e['offset'] for e in eps],
                          expert_shape=list(eps[0]['expert'].shape), init_state=eps[0]['init_state'].tolist())
    gst, gimg = R.sim_run(env, None, eps[0]['init_state'], eps[0]['expert'], render=True)
    out['margin_expert'] = R.margin_parts(gst, gst)
    st2, _ = R.sim_run(env, None, eps[0]['init_state'], eps[0]['expert'] + 0.3, render=False)
    out['margin_perturbed_0p3'] = R.margin_parts(gst, st2)
    # world model: encode + imagine
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    s0, img0, p0 = R.start_obs(env, None, eps[0]['init_state'])
    zs, ps = R.encode_frames(m, pre, img0[None], p0[None])
    zg, _ = R.encode_frames(m, pre, gimg[None], R.proprio_of(gst)[None])
    zi, pi = R.imagine(m, pre, img0[None], p0[None], eps[0]['expert'][None])
    out['wm'] = dict(z_shape=list(zs.shape), p_shape=list(ps.shape), zi_shape=list(zi.shape), num_hist=int(T_['cfg'].num_hist),
                     frameskip=int(T_['cfg'].frameskip), d_vis_imag_goal=float(R.d_vis(zi, zg)[0]), d_vis_start_goal=float(R.d_vis(zs, zg)[0]))
    out['time_s'] = time.time() - t0
    R.jdump(out, f'{BP.RES}/smoke_{E.SUF}.json')
    print(json.dumps(out, indent=1, default=str))
    ok = out['replay_xy_err']['max'] < 1e-4 and list(zs.shape) == [1, 196, 384] and out['render']['shape'] == [224, 224, 3]
    print('SMOKE_E7_DONE' if ok else 'SMOKE_E7_FAILED', f'{time.time() - t0:.0f}s')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
