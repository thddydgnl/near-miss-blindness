"""B E1 M1 step 2a: judge-training set on the judge trajectories J of the train split (disjoint from calibration C,
auxiliary evaluation T and val). R2-12 clfset protocol extended to H in {3,5,7,10}: per H 500 episodes x (expert + 7
perturbations from the iid/offset/scale/gross mixture, magnitude = log-uniform multiplier in [0.3, 3] of the family's
50%-success magnitude from the E1 calibration on C). Stores real final-frame features (judges a_full, a_obj), imagined
final features at that H (information-ceiling judges), goal / start features, labels and margins of both definitions.
20% of episodes are held out. usage: python e1_judgeset.py"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E

N_EP = 500
K = 7
CHUNK = 125


def make_seqs(ep, M, rng):
    seqs, fams = [('expert', ep['expert'].copy())], ['expert']
    for _ in range(K):
        fam = rng.choice(['iid', 'offset', 'scale', 'gross'], p=[0.35, 0.35, 0.15, 0.15])
        mult = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        if fam == 'iid':
            p = {'sigma': M['sigma'] * mult}
        elif fam == 'offset':
            p = {'beta_max': M['beta'] * mult}
        elif fam == 'scale':
            p = {'k_max': M['k'] * mult}
        else:
            p = {'q': 1.0, 'sig_big': M['sb'] * rng.uniform(0.5, 1.5), 'sig_small': R.SIG_SMALL}
        a, _ = R.make_actions(ep['expert'], fam, p, rng)
        seqs.append((fam, a))
        fams.append(fam)
    return seqs, fams


def main():
    t0 = time.time()
    split = json.load(open(f'{BP.RES}/split_{E.SUF}.json'))
    cal = json.load(open(f'{BP.RES}/calib_{E.SUF}.json'))
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    dt = T_['dset_train']
    pool = R.pool()
    store = {k: [] for k in ['z_real', 'p_real', 'z_imag', 'p_imag', 'z_goal', 'p_goal', 'z_start', 'p_start']}
    meta = {k: [] for k in ['ep', 'H', 'r', 'r_obj', 'agent', 'block', 'ang', 'r_pos', 'r_ang', 'r_obj_pos', 'fam']}
    ep_info, e_off = [], 0
    for H in E.HS:
        M = cal['mags'][str(H)]
        eps = E.sample_episodes_from(dt, split['J'], N_EP, H, seed=E.SEED_BASE + 8000 + H)
        rng = np.random.RandomState(E.SEED_BASE + 8100 + H)
        for c0 in range(0, len(eps), CHUNK):
            ch = eps[c0:c0 + CHUNK]
            jobs, famss = [], []
            for ep in ch:
                seqs, fams = make_seqs(ep, M, rng)
                jobs.append((ep, seqs, True))
                famss.append(fams)
            outs = pool.map(R.episode_job, jobs, chunksize=2)
            nS = K + 1
            simg = np.stack([o['start_img'] for o in outs]); spro = np.stack([o['start_proprio'] for o in outs])
            zs, ps = R.encode_frames(m, pre, simg, spro)
            zg, pg = R.encode_frames(m, pre, np.stack([o['goal_img'] for o in outs]), np.stack([o['goal_proprio'] for o in outs]))
            zr, pr = R.encode_frames(m, pre, np.concatenate([o['final_imgs'] for o in outs]),
                                     np.concatenate([o['final_proprios'] for o in outs]))
            acts = np.stack([s[1] for j in jobs for s in j[1]])
            zi, pi = R.imagine(m, pre, np.repeat(simg, nS, 0), np.repeat(spro, nS, 0), acts)
            for k, v in (('z_real', zr), ('p_real', pr), ('z_imag', zi), ('p_imag', pi), ('z_goal', zg), ('p_goal', pg),
                         ('z_start', zs), ('p_start', ps)):
                store[k].append(v)
            for i, (ep, o) in enumerate(zip(ch, outs)):
                ep_info.append(dict(traj=ep['traj'], offset=ep['offset'], H=H))
                for k in range(nS):
                    mp = o['margins'][k]
                    meta['ep'].append(e_off + i); meta['H'].append(H); meta['fam'].append(famss[i][k])
                    for kk in ('r', 'r_obj', 'agent', 'block', 'ang', 'r_pos', 'r_ang', 'r_obj_pos'):
                        meta[kk].append(mp[kk])
            e_off += len(ch)
            print(f'H{H} {c0 + len(ch)}/{N_EP}: {e_off} episodes, {len(meta["r"])} seqs, {time.time() - t0:.0f}s', flush=True)
    pool.close()
    out = {k: torch.cat(v) for k, v in store.items()}
    for k, v in meta.items():
        out[k] = torch.as_tensor(np.array(v)) if k != 'fam' else v
    is_val = np.random.RandomState(0).uniform(size=e_off) < 0.2
    out['ep_is_val'] = torch.as_tensor(is_val)
    out['ep_info'] = ep_info
    torch.save(out, f'{BP.FEAT}/judgeset_{E.SUF}.pt')
    r, ro = np.array(meta['r']), np.array(meta['r_obj'])
    summ = dict(n_episodes=e_off, n_seqs=len(r), n_val_episodes=int(is_val.sum()),
                bins={b: float(np.mean([R.r_bin(x) == b for x in r])) for b in ('success', 'near', 'mid', 'far')},
                bins_obj={b: float(np.mean([R.r_bin(x) == b for x in ro])) for b in ('success', 'near', 'mid', 'far')},
                trajs_used=sorted(set(int(e['traj']) for e in ep_info)), time_s=time.time() - t0)
    R.jdump(summ, f'{BP.RES}/judgeset_{E.SUF}_meta.json')
    print('JUDGESET_E1_DONE', {k: v for k, v in summ.items() if k != 'trajs_used'})


if __name__ == '__main__':
    main()
