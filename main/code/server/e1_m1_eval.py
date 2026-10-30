"""B E1 M1 step 3: failure-distance curve sets (DESIGN_main v1 section 0.2 / 6.1).
One cell = (set, H, definition). set: 'val' (main; all 21 val trajectories, world-model unseen) or 'trainsplit'
(auxiliary; train-split trajectories T, disjoint from the judge trajectories J and the calibration trajectories C).
Per cell 3,600 counterfactual targets (3,000 stratified uniform on [0.8, 3], 300 on [0.2, 0.8), 300 on [3, 6]); pattern
family offset / iid half each; per target the severity is found by bisection (14 steps, bounds from the calibration on
C) so that the target definition's failure distance hits r*. Targets unreachable on an episode are re-queued to new
episodes (at most 30 attempts). Six targets per episode; each episode also carries its expert sequence (r = 0).
Real: simulator replay + final-frame render + DINOv2 encoding. Imagined: open-loop DINO-WM from the 1-frame start
observation with the same actions (R2-12 / DINO-WM planning convention). Frozen judges: a_full, a_obj ensembles (K = 5),
information-ceiling ensembles ai_full, ai_obj (imagined-feature judges), (b) d_vis with frozen thresholds.
Writes results/<stage>/m1/eval_<set>_H<H>_<def>.jsonl (+ .done). usage: python e1_m1_eval.py <set> <H> <def>"""
import os, sys, time, json, zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E

PER_EP = 6
BATCH_EP = 100
MAX_TRIES = 30


def main():
    st, H, key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    assert st in ('val', 'trainsplit') and H in E.HS and key in E.DEFS
    out_path = f'{BP.RES}/eval_{st}_H{H}_{key}.jsonl'
    if os.path.exists(out_path + '.done'):
        print('done already', out_path)
        return
    if os.path.exists(out_path):
        os.remove(out_path)
    t0 = time.time()
    cell_seed = zlib.crc32(f'{st}_{H}_{key}'.encode()) % 100000
    split = json.load(open(f'{BP.RES}/split_e1.json'))
    cal = json.load(open(f'{BP.RES}/calib_e1.json'))
    s_max = cal['s_max'][str(H)][key]
    J = torch.load(f'{BP.FEAT}/judges_e1.pt', weights_only=False)
    tau = J['tau']
    nets = {name: E.load_ensemble(J['sds'][name]) for name in ('a_full', 'a_obj', 'ai_full', 'ai_obj')}
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    if st == 'val':
        dset = T_['dset']
        trajs = list(range(len(dset)))
    else:
        dset = T_['dset_train']
        trajs = split['T']
    rng = np.random.RandomState(cell_seed)
    rstars, fams = E.stratified_targets(rng)
    queue = [dict(tid=i, rstar=float(rstars[i]), fam=str(fams[i]), tries=0) for i in range(len(rstars))]
    dropped, n_ep, n_unreach, batch = [], 0, 0, 0
    pool = R.pool()
    stats = dict(calls=[])
    with open(out_path, 'a') as fh:
        while queue:
            eps = E.sample_episodes_from(dset, trajs, BATCH_EP, H, seed=cell_seed * 1000 + batch)
            jobs, assigned = [], []
            for i, ep in enumerate(eps):
                if not queue:
                    break
                tg, queue = queue[:PER_EP], queue[PER_EP:]
                for t in tg:
                    t['tries'] += 1
                assigned.append(tg)
                jobs.append((ep, [(t['tid'], t['rstar'], t['fam']) for t in tg], key, s_max,
                             (cell_seed * 100003 + batch * 1000 + i) % 4294967295, True))
            outs = pool.map(E.grid_episode_job, jobs, chunksize=2)
            for tg, o in zip(assigned, outs):
                un = {u['tid'] for u in o['unreachable']}
                n_unreach += len(un)
                for t in tg:
                    if t['tid'] in un:
                        (dropped if t['tries'] >= MAX_TRIES else queue).append(t)
            # GPU: encode + imagine (expert reference + every reached counterfactual)
            recs, simg, spro, acts, fimg, fpro, gidx = [], [], [], [], [], [], []
            gimg = np.stack([o['goal_img'] for o in outs]); gpro = np.stack([o['goal_proprio'] for o in outs])
            for j, (ep, o) in enumerate(zip(eps[:len(outs)], outs)):
                items = [('expert', None, 0.0, 'expert', 0.0, 0, o['expert'])] + \
                        [('cf', d['tid'], d['rstar'], d['fam'], d['s'], d['calls'], d) for d in o['done']]
                for kind, tid, rstar, fam, s, calls, d in items:
                    mp = d['margins']
                    recs.append(dict(set=st, H=H, defk=key, kind=kind, tid=tid, rstar=rstar, fam=fam, s=s, calls=calls,
                                     ep_uid=n_ep + j, traj=int(ep['traj']), offset=int(ep['offset']),
                                     start_id=f"{int(ep['traj'])}_{int(ep['offset'])}",
                                     bin=R.r_bin(mp['r']), bin_obj=R.r_bin(mp['r_obj']), success=bool(mp['r'] < 1),
                                     success_obj=bool(mp['r_obj'] < 1), **mp))
                    simg.append(o['start_img']); spro.append(o['start_proprio']); acts.append(d['acts'])
                    fimg.append(d['final_img']); fpro.append(d['final_proprio']); gidx.append(j)
                    if kind == 'cf':
                        stats['calls'].append(calls)
            n_ep += len(outs)
            if not recs:
                batch += 1
                continue
            zg, _ = R.encode_frames(m, pre, gimg, gpro)
            zg_r = zg[torch.as_tensor(gidx)]
            zr, _ = R.encode_frames(m, pre, np.stack(fimg), np.stack(fpro))
            zi, _ = R.imagine(m, pre, np.stack(simg), np.stack(spro), np.stack(acts))
            f = {}
            for name in ('a_full', 'a_obj'):
                Pr, er = E.ens_prob(nets[name], zr, zg_r)
                Pi, ei = E.ens_prob(nets[name], zi, zg_r)
                f[f'p{name[1:]}_real'], f[f'p{name[1:]}_imag'] = er, ei
                f[f'p{name[1:]}_real_m'], f[f'p{name[1:]}_imag_m'] = Pr.T, Pi.T
            for name in ('ai_full', 'ai_obj'):
                _, ei = E.ens_prob(nets[name], zi, zg_r)
                f[f'p{name[1:]}_imag'] = ei
            f['dv_real'] = R.d_vis(zr, zg_r).numpy()
            f['dv_imag'] = R.d_vis(zi, zg_r).numpy()
            f['dv_imag_vs_real'] = R.d_vis(zi, zr).numpy()
            for k, rec in enumerate(recs):
                for kk, vv in f.items():
                    rec[kk] = [round(float(x), 5) for x in vv[k]] if np.ndim(vv[k]) else round(float(vv[k]), 6)
                fh.write(json.dumps(rec) + '\n')
            fh.flush()
            n_cf = sum(1 for r in recs if r['kind'] == 'cf')
            print(f'{st} H{H} {key}: batch {batch} episodes {n_ep} +{n_cf} cf, queue {len(queue)}, dropped {len(dropped)}, '
                  f'{time.time() - t0:.0f}s', flush=True)
            batch += 1
    pool.close()
    summ = dict(set=st, H=H, defk=key, n_targets=len(rstars), n_dropped=len(dropped), n_episodes=n_ep,
                n_unreachable_attempts=n_unreach, reach_rate_per_attempt=float(len(rstars) - len(dropped)) /
                max(1, len(rstars) - len(dropped) + n_unreach), s_max=s_max, tau=tau,
                mean_sim_calls=float(np.mean(stats['calls'])) if stats['calls'] else None, time_s=time.time() - t0)
    R.jdump(summ, out_path.replace('.jsonl', '_meta.json'))
    open(out_path + '.done', 'w').write(json.dumps(summ))
    print('EVAL_CELL_DONE', json.dumps(summ))


if __name__ == '__main__':
    main()
