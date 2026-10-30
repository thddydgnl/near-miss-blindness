"""Evaluation sets: episodes x (expert + 8 policies). Real: fast sim + final-frame render + DINOv2 encoding.
Imagined: open-loop DINO-WM rollout from the same 1-frame start observation with the same actions. Frozen judges.
Writes one JSON line per trajectory to results/traj_<set>.jsonl as it goes (chunks of 100 episodes).
usage: python main12.py <set: main|H10|trainsplit> [n_episodes]"""
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import r12lib as R

SETS = {'main': dict(split='valid', H=5, n=400, seed=303),
        'H10': dict(split='valid', H=10, n=300, seed=304),
        'trainsplit': dict(split='train', H=5, n=400, seed=305),
        'smoke': dict(split='valid', H=5, n=8, seed=999)}
POLICIES = ['far_L', 'mixA_L', 'mixB_L', 'near_L', 'iid_L', 'far_H', 'near_H', 'iid_H']
CHUNK = 100


def load_judge(sd, variant):
    d, l = (128, 2) if variant == 'd128_l2' else (256, 4)
    net = R.PairJudge(d=d, layers=l).cuda()
    net.load_state_dict(sd)
    net.eval()
    return net


def main():
    name = sys.argv[1]
    S = SETS[name]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else S['n']
    tag = name if len(sys.argv) <= 2 else f'{name}_n{n}'
    t0 = time.time()
    T = R.load_model()
    m, pre = T['model'], T['pre']
    dset = T['dset'] if S['split'] == 'valid' else T['dset_train']
    P = json.load(open(f'{R.RES}/policies_H{S["H"]}.json'))['chosen']
    J = torch.load(f'{R.FEAT}/judges.pt')
    net_a, net_ai = load_judge(J['a'], J['a_variant']), load_judge(J['a_imag'], J['a_imag_variant'])
    exclude = None
    if S['split'] == 'train':
        exclude = json.load(open(f'{R.RES}/clfset_meta.json'))['train_trajs_used']
    eps = R.sample_episodes(dset, n, S['H'], S['seed'], exclude=exclude)
    out_path = f'{R.RES}/traj_{tag}.jsonl'
    if os.path.exists(out_path):
        os.remove(out_path)
    pool = R.pool()
    nS = 1 + len(POLICIES)
    for c0 in range(0, n, CHUNK):
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
        zs, ps = R.encode_frames(m, pre, simg, spro)
        zg, pg = R.encode_frames(m, pre, np.stack([o['goal_img'] for o in outs]), np.stack([o['goal_proprio'] for o in outs]))
        zr, pr = R.encode_frames(m, pre, np.concatenate([o['final_imgs'] for o in outs]),
                                 np.concatenate([o['final_proprios'] for o in outs]))
        acts = np.stack([s[1] for j in jobs for s in j[1]])
        zi, pi = R.imagine(m, pre, np.repeat(simg, nS, 0), np.repeat(spro, nS, 0), acts)
        zg_r, pg_r = zg.repeat_interleave(nS, 0), pg.repeat_interleave(nS, 0)
        zs_r, ps_r = zs.repeat_interleave(nS, 0), ps.repeat_interleave(nS, 0)
        zi_exp = zi[0::nS].repeat_interleave(nS, 0)
        f = dict(
            pa_real=R.judge_prob(net_a, zr, zg_r), pa_imag=R.judge_prob(net_a, zi, zg_r),
            pa_null=R.judge_prob(net_a, zs_r, zg_r),
            pai_real=R.judge_prob(net_ai, zr, zg_r), pai_imag=R.judge_prob(net_ai, zi, zg_r),
            dv_real=R.d_vis(zr, zg_r).numpy(), dv_imag=R.d_vis(zi, zg_r).numpy(), dv_null=R.d_vis(zs_r, zg_r).numpy(),
            df_real=R.d_full(zr, zg_r, pr, pg_r).numpy(), df_imag=R.d_full(zi, zg_r, pi, pg_r).numpy(),
            df_null=R.d_full(zs_r, zg_r, ps_r, pg_r).numpy(),
            dv_imag_vs_expimag=R.d_vis(zi, zi_exp).numpy(), dv_imag_vs_real=R.d_vis(zi, zr).numpy())
        with open(out_path, 'a') as fh:
            for i, (ep, o) in enumerate(zip(ch, outs)):
                for k in range(nS):
                    j = i * nS + k
                    mp = o['margins'][k]
                    rec = dict(set=name, ep=c0 + i, traj=ep['traj'], offset=ep['offset'], H=S['H'],
                               policy=jobs[i][1][k][0], bin=R.r_bin(mp['r']), success=bool(mp['r'] < 1),
                               bin_obj=R.r_bin(mp['r_obj']), success_obj=bool(mp['r_obj'] < 1), **mp,
                               **{kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else float(vv))
                                  for kk, vv in infos[i][k].items()},
                               **{kk: float(vv[j]) for kk, vv in f.items()})
                    fh.write(json.dumps(rec) + '\n')
        torch.save(dict(zr=zr, zi=zi, zg=zg, zs=zs, pr=pr, pi=pi, pg=pg, ps=ps),
                   f'{R.FEAT}/eval_{tag}_{c0}.pt')
        print(f'{tag}: {c0 + len(ch)}/{n} episodes, {time.time() - t0:.0f}s', flush=True)
    pool.close()
    print('MAIN_DONE', tag, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
