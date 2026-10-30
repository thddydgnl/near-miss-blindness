"""Judge-training set (PushT train split; disjoint from all evaluation episodes of the main set, which use the val split).
Per episode: the expert segment + 7 perturbed sequences drawn from a broad mixture (iid / offset / scale / gross,
magnitude = log-uniform multiplier in [0.3, 3] of the family's 50%-success magnitude from the sim calibration).
Stores real final-frame features, imagined final features (open-loop DINO-WM), goal and start features, labels, margins.
usage: python clfset12.py"""
import sys, time, json
import numpy as np
import torch
sys.path.insert(0, '/workspace/code')
import r12lib as R

N_EP = {5: 1400, 10: 600}
SEED = {5: 202, 10: 203}
K = 7
CHUNK = 200


def mags(H):
    P = json.load(open(f'{R.RES}/policies_H{H}.json'))
    from calib12 import interp_param, GRIDS
    key, grid = GRIDS['scale']
    k50 = interp_param(grid, [P['grid'][f'scale_{x}']['success'] for x in grid], 0.5)
    return dict(sigma=P['chosen']['iid_L']['p']['sigma'], beta=P['chosen']['near_L']['p']['beta_max'], k=k50,
                sb=P['sig_big'])


def make_seqs(ep, M, rng):
    seqs, fams = [('expert', ep['expert'].copy())], ['expert']
    for k in range(K):
        fam = rng.choice(['iid', 'offset', 'scale', 'gross'], p=[0.35, 0.35, 0.15, 0.15])
        mult = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        if fam == 'iid':
            p = {'sigma': M['sigma'] * mult}
        elif fam == 'offset':
            p = {'beta_max': M['beta'] * mult}
        elif fam == 'scale':
            p = {'k_max': M['k'] * mult}
        else:
            p = {'q': 1.0, 'sig_big': M['sb'] * rng.uniform(0.5, 1.5), 'sig_small': 1.0}
        a, _ = R.make_actions(ep['expert'], fam, p, rng)
        seqs.append((fam, a))
        fams.append(fam)
    return seqs, fams


def main():
    t0 = time.time()
    T = R.load_model()
    m, pre = T['model'], T['pre']
    dt = T['dset_train']
    pool = R.pool()
    store = {k: [] for k in ['z_real', 'p_real', 'z_imag', 'p_imag', 'z_goal', 'p_goal', 'z_start', 'p_start']}
    meta = dict(ep=[], H=[], r=[], fam=[], agent=[], block=[], ang=[])
    ep_info = []
    e_off = 0
    for H in (5, 10):
        M = mags(H)
        eps = R.sample_episodes(dt, N_EP[H], H, seed=SEED[H])
        rng = np.random.RandomState(SEED[H] + 7)
        for c0 in range(0, len(eps), CHUNK):
            ch = eps[c0:c0 + CHUNK]
            jobs, famss = [], []
            for ep in ch:
                seqs, fams = make_seqs(ep, M, rng)
                jobs.append((ep, seqs, True))
                famss.append(fams)
            outs = pool.map(R.episode_job, jobs, chunksize=2)
            nS = K + 1
            simg = np.stack([o['start_img'] for o in outs])
            spro = np.stack([o['start_proprio'] for o in outs])
            zs, ps = R.encode_frames(m, pre, simg, spro)
            zg, pg = R.encode_frames(m, pre, np.stack([o['goal_img'] for o in outs]),
                                     np.stack([o['goal_proprio'] for o in outs]))
            fimg = np.concatenate([o['final_imgs'] for o in outs])
            fpro = np.concatenate([o['final_proprios'] for o in outs])
            zr, pr = R.encode_frames(m, pre, fimg, fpro)
            acts = np.stack([s[1] for j in jobs for s in j[1]])
            zi, pi = R.imagine(m, pre, np.repeat(simg, nS, 0), np.repeat(spro, nS, 0), acts)
            for k, v in (('z_real', zr), ('p_real', pr), ('z_imag', zi), ('p_imag', pi), ('z_goal', zg),
                         ('p_goal', pg), ('z_start', zs), ('p_start', ps)):
                store[k].append(v)
            for i, (ep, o) in enumerate(zip(ch, outs)):
                ep_info.append(dict(traj=ep['traj'], offset=ep['offset'], H=H))
                for k in range(nS):
                    mp = o['margins'][k]
                    meta['ep'].append(e_off + i)
                    meta['H'].append(H)
                    meta['r'].append(mp['r'])
                    meta['fam'].append(famss[i][k])
                    meta['agent'].append(mp['agent'])
                    meta['block'].append(mp['block'])
                    meta['ang'].append(mp['ang'])
            e_off += len(ch)
            print(f'H{H} chunk {c0} done: {e_off} episodes, {len(meta["r"])} seqs, {time.time()-t0:.0f}s', flush=True)
    pool.close()
    out = {k: torch.cat(v) for k, v in store.items()}
    for k, v in meta.items():
        out[k] = torch.as_tensor(np.array(v)) if k != 'fam' else v
    rng = np.random.RandomState(0)
    is_val = rng.uniform(size=e_off) < 0.2           # held-out 20% of episodes
    out['ep_is_val'] = torch.as_tensor(is_val)
    torch.save(out, f'{R.FEAT}/clf.pt')
    r = np.array(meta['r'])
    summ = dict(n_episodes=e_off, n_seqs=len(r), success_rate=float((r < 1).mean()),
                bins={b: float(np.mean([R.r_bin(x) == b for x in r])) for b in ['success', 'near', 'mid', 'far']},
                train_trajs_used=sorted(set(int(e['traj']) for e in ep_info)), n_val_episodes=int(is_val.sum()),
                mags={H: mags(H) for H in (5, 10)}, time_s=time.time() - t0)
    R.jdump(summ, f'{R.RES}/clfset_meta.json')
    print('CLFSET_DONE', {k: v for k, v in summ.items() if k != 'train_trajs_used'})


if __name__ == '__main__':
    main()
