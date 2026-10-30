"""B E2 M1 control set: the R2-12 main set (val split, H = 5, 400 episodes, seed 303; expert + 8 perturbation policies with
the parameters of policies_H5.json, reproduced bit-identically in E0) judged with the E1 judges: a_full and a_obj
ensembles (K = 5), the information-ceiling ensembles ai_full / ai_obj, and (b) d_vis with the frozen thresholds.
Reported separately from the DP policy law (DESIGN_main v1 section 0.2, v2.2 section 7). Episodes and actions are
generated exactly as in main12.py (same seeds); the achieved margins are checked against the E0 reproduction file.
Writes results/E2/m1/ctrl_main.jsonl. usage: python e2_ctrl.py"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E

S = dict(split='valid', H=5, n=400, seed=303)
POLICIES = ['far_L', 'mixA_L', 'mixB_L', 'near_L', 'iid_L', 'far_H', 'near_H', 'iid_H']
CHUNK = 100
E0_RES = os.path.expanduser('~/pmx/B/results/E0/m1')
E1_FEAT = os.path.expanduser('~/pmx/B/feat/E1/m1')


def main():
    t0 = time.time()
    T = R.load_model()
    m, pre = T['model'], T['pre']
    dset = T['dset']
    P = json.load(open(f'{E0_RES}/policies_H5.json'))['chosen']
    J = torch.load(f'{E1_FEAT}/judges_e1.pt', weights_only=False)
    nets = {name: E.load_ensemble(J['sds'][name]) for name in ('a_full', 'a_obj', 'ai_full', 'ai_obj')}
    tau = J['tau']
    eps = R.sample_episodes(dset, S['n'], S['H'], S['seed'])
    out_path = f'{BP.RES}/ctrl_main.jsonl'
    pool = R.pool()
    nS = 1 + len(POLICIES)
    with open(out_path + '.tmp', 'w') as fh:
        for c0 in range(0, S['n'], CHUNK):
            ch = eps[c0:c0 + CHUNK]
            jobs, infos = [], []
            for i, ep in enumerate(ch):
                rng = np.random.RandomState(S['seed'] * 10007 + c0 + i)
                seqs, inf = [('expert', ep['expert'].copy())], [{}]
                for pn in POLICIES:
                    a, info = R.make_actions(ep['expert'], P[pn]['fam'], P[pn]['p'], rng)
                    seqs.append((pn, a))
                    inf.append(info)
                jobs.append((ep, seqs, True))
                infos.append(inf)
            outs = pool.map(R.episode_job, jobs, chunksize=2)
            simg = np.stack([o['start_img'] for o in outs])
            spro = np.stack([o['start_proprio'] for o in outs])
            zg, _ = R.encode_frames(m, pre, np.stack([o['goal_img'] for o in outs]), np.stack([o['goal_proprio'] for o in outs]))
            zr, _ = R.encode_frames(m, pre, np.concatenate([o['final_imgs'] for o in outs]),
                                    np.concatenate([o['final_proprios'] for o in outs]))
            acts = np.stack([s[1] for j in jobs for s in j[1]])
            zi, _ = R.imagine(m, pre, np.repeat(simg, nS, 0), np.repeat(spro, nS, 0), acts)
            zg_r = zg.repeat_interleave(nS, 0)
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
            for i, (ep, o) in enumerate(zip(ch, outs)):
                for k in range(nS):
                    j = i * nS + k
                    mp = o['margins'][k]
                    rec = dict(set='main', ep=c0 + i, traj=int(ep['traj']), offset=int(ep['offset']), H=S['H'],
                               policy=jobs[i][1][k][0], bin=R.r_bin(mp['r']), success=bool(mp['r'] < 1),
                               bin_obj=R.r_bin(mp['r_obj']), success_obj=bool(mp['r_obj'] < 1), **mp,
                               **{kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else float(vv)) for kk, vv in infos[i][k].items()})
                    for kk, vv in f.items():
                        rec[kk] = [round(float(x), 5) for x in vv[j]] if np.ndim(vv[j]) else round(float(vv[j]), 6)
                    fh.write(json.dumps(rec) + '\n')
            print(f'ctrl main: {c0 + len(ch)}/{S["n"]} episodes ({time.time() - t0:.0f}s)', flush=True)
    pool.close()
    os.replace(out_path + '.tmp', out_path)
    # identity check against the E0 reproduction of the R2-12 main set (same episodes / actions -> same margins)
    chk = None
    e0 = f'{E0_RES}/traj_main.jsonl'
    if os.path.exists(e0):
        a = {(x['ep'], x['policy']): x['r'] for x in map(json.loads, open(e0))}
        b = {(x['ep'], x['policy']): x['r'] for x in map(json.loads, open(out_path))}
        common = sorted(set(a) & set(b))
        chk = dict(n_common=len(common), max_abs_diff_r=float(max(abs(a[k] - b[k]) for k in common)) if common else None)
    summ = dict(n_records=S['n'] * nS, tau=tau, e0_identity=chk, time_s=time.time() - t0)
    R.jdump(summ, f'{BP.RES}/ctrl_main_meta.json')
    print('E2_CTRL_DONE', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
