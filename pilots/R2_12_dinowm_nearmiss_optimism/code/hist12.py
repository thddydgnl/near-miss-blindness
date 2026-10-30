"""Post-hoc (사후) check: does the training-time context (num_hist = 3 frames) change the near-miss blur?
Same main-set episodes (val, H5, seed 303) with offset >= 10 and the identical action sequences as main12.
Imagination with (i) the 1-frame start (DINO-WM planning convention, as in main12) and (ii) 3 history frames
(dataset states at offset-10, -5, 0 rendered by the env + the dataset actions in between). Real outcomes identical.
usage: python hist12.py"""
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, '/workspace/code')
import r12lib as R
from main12 import SETS, POLICIES, load_judge

CHUNK = 100


@torch.no_grad()
def imagine_hist(m, pre, himgs, hpros, hacts, acts, bs=128):
    """himgs (N,3,H,W,3), hpros (N,3,4) raw, hacts (N,10,2) env-unit history actions (2 chunks),
    acts (N,H*FS,2) future env-unit actions. Returns final visual (N,196,384) fp16, proprio-emb (N,10)."""
    vs, ps = [], []
    for i in range(0, len(acts), bs):
        o = pre.transform_obs({'visual': np.asarray(himgs[i:i + bs]), 'proprio': np.asarray(hpros[i:i + bs])})
        o = {k: v.float().cuda() for k, v in o.items()}
        a = torch.cat([R.to_model_actions(pre, hacts[i:i + bs]), R.to_model_actions(pre, acts[i:i + bs])], 1).cuda()
        z_obses, _ = m.rollout(obs_0=o, act=a)
        vs.append(z_obses['visual'][:, -1].half().cpu())
        ps.append(z_obses['proprio'][:, -1].float().cpu())
    return torch.cat(vs), torch.cat(ps)


def main():
    t0 = time.time()
    S = SETS['main']
    T = R.load_model()
    m, pre = T['model'], T['pre']
    dset = T['dset']
    P = json.load(open(f'{R.RES}/policies_H5.json'))['chosen']
    J = torch.load(f'{R.FEAT}/judges.pt')
    net_a, net_ai = load_judge(J['a'], J['a_variant']), load_judge(J['a_imag'], J['a_imag_variant'])
    eps = R.sample_episodes(dset, S['n'], S['H'], S['seed'])
    amean, astd = dset.action_mean.numpy(), dset.action_std.numpy()
    env = R.get_env()
    keep = [e for e, ep in enumerate(eps) if ep['offset'] >= 10]
    print(f'{len(keep)} of {len(eps)} main episodes have offset >= 10', flush=True)
    out_path = f'{R.RES}/traj_hist.jsonl'
    if os.path.exists(out_path):
        os.remove(out_path)
    pool = R.pool()
    nS = 1 + len(POLICIES)
    for c0 in range(0, len(keep), CHUNK):
        idx = keep[c0:c0 + CHUNK]
        jobs, himgs, hpros, hacts = [], [], [], []
        for e in idx:
            ep = eps[e]
            rng = np.random.RandomState(S['seed'] * 10007 + e)     # identical to main12 (c0 + i == e there)
            seqs = [('expert', ep['expert'].copy())]
            for pn in POLICIES:
                a, _ = R.make_actions(ep['expert'], P[pn]['fam'], P[pn]['p'], rng)
                seqs.append((pn, a))
            jobs.append((ep, seqs, True))
            hi, hp = [], []
            for dt in (10, 5, 0):
                st = dset.states[ep['traj'], ep['offset'] - dt].numpy().astype(np.float64)
                _, img, pro = R.start_obs(env, ep['shape'], st)
                hi.append(img)
                hp.append(pro)
            himgs.append(np.stack(hi))
            hpros.append(np.stack(hp))
            hacts.append(dset.actions[ep['traj'], ep['offset'] - 10:ep['offset']].numpy() * astd + amean)
        outs = pool.map(R.episode_job, jobs, chunksize=2)
        simg = np.stack([o['start_img'] for o in outs])
        spro = np.stack([o['start_proprio'] for o in outs])
        assert np.abs(simg.astype(int) - np.stack(himgs)[:, 2].astype(int)).max() == 0
        zg, pg = R.encode_frames(m, pre, np.stack([o['goal_img'] for o in outs]), np.stack([o['goal_proprio'] for o in outs]))
        zr, pr = R.encode_frames(m, pre, np.concatenate([o['final_imgs'] for o in outs]),
                                 np.concatenate([o['final_proprios'] for o in outs]))
        acts = np.stack([s[1] for j in jobs for s in j[1]])
        z1, p1 = R.imagine(m, pre, np.repeat(simg, nS, 0), np.repeat(spro, nS, 0), acts)
        z3, p3 = imagine_hist(m, pre, np.repeat(np.stack(himgs), nS, 0), np.repeat(np.stack(hpros), nS, 0),
                              np.repeat(np.stack(hacts), nS, 0), acts)
        zg_r, pg_r = zg.repeat_interleave(nS, 0), pg.repeat_interleave(nS, 0)
        f = dict(pa_real=R.judge_prob(net_a, zr, zg_r), pa_imag1=R.judge_prob(net_a, z1, zg_r),
                 pa_imag3=R.judge_prob(net_a, z3, zg_r), pai_imag1=R.judge_prob(net_ai, z1, zg_r),
                 pai_imag3=R.judge_prob(net_ai, z3, zg_r),
                 dv_real=R.d_vis(zr, zg_r).numpy(), dv_imag1=R.d_vis(z1, zg_r).numpy(), dv_imag3=R.d_vis(z3, zg_r).numpy(),
                 df_real=R.d_full(zr, zg_r, pr, pg_r).numpy(), df_imag1=R.d_full(z1, zg_r, p1, pg_r).numpy(),
                 df_imag3=R.d_full(z3, zg_r, p3, pg_r).numpy(),
                 dv_err1=R.d_vis(z1, zr).numpy(), dv_err3=R.d_vis(z3, zr).numpy())
        with open(out_path, 'a') as fh:
            for i, (e, o) in enumerate(zip(idx, outs)):
                for k in range(nS):
                    j = i * nS + k
                    mp = o['margins'][k]
                    rec = dict(set='hist', ep=e, traj=eps[e]['traj'], offset=eps[e]['offset'], policy=jobs[i][1][k][0],
                               bin=R.r_bin(mp['r']), success=bool(mp['r'] < 1), r=mp['r'],
                               **{kk: float(vv[j]) for kk, vv in f.items()})
                    fh.write(json.dumps(rec) + '\n')
        print(f'hist: {c0 + len(idx)}/{len(keep)} episodes, {time.time() - t0:.0f}s', flush=True)
    pool.close()
    # summary
    recs = [json.loads(l) for l in open(out_path)]
    jd = json.load(open(f'{R.RES}/judges.json'))
    tb, tb2 = jd['b']['tau'], jd['b2']['tau']
    g = lambda k: np.array([x[k] for x in recs])
    pol, bins, succ = g('policy'), g('bin'), g('success').astype(bool)
    inpol = np.isin(pol, POLICIES)
    from scipy.stats import spearmanr, mannwhitneyu, kendalltau
    summ = {}
    for ctx in ('1', '3'):
        F = {'a': g(f'pa_imag{ctx}') > 0.5, 'b': g(f'dv_imag{ctx}') < tb, 'b2': g(f'df_imag{ctx}') < tb2,
             'ai': g(f'pai_imag{ctx}') > 0.5}
        d = {}
        for j, fl in F.items():
            pb = {b: float(fl[inpol & (bins == b)].mean()) for b in ['success', 'near', 'mid', 'far']}
            s = np.array([succ[pol == p].mean() for p in POLICIES])
            sh = np.array([fl[pol == p].mean() for p in POLICIES])
            n = np.array([(bins[pol == p] == 'near').mean() for p in POLICIES])
            sc = g(f'pa_imag{ctx}') if j == 'a' else (g(f'pai_imag{ctx}') if j == 'ai' else -g(f'dv_imag{ctx}'))
            u = mannwhitneyu(sc[inpol & (bins == 'success')], sc[inpol & (bins == 'near')], alternative='two-sided').statistic
            d[j] = dict(p=pb, p_expert=float(fl[pol == 'expert'].mean()), spearman_near_over=float(spearmanr(n, sh - s).correlation),
                        kendall=float(kendalltau(sh, s).correlation), over=(sh - s).round(3).tolist(),
                        auc_success_vs_near=float(u / ((inpol & (bins == 'success')).sum() * (inpol & (bins == 'near')).sum())))
        d['median_imag_err_dvis'] = float(np.median(g(f'dv_err{ctx}')[inpol]))
        summ[f'ctx{ctx}'] = d
    summ['real'] = {'a': {b: float((g('pa_real') > 0.5)[inpol & (bins == b)].mean()) for b in ['success', 'near', 'mid', 'far']},
                    'b': {b: float((g('dv_real') < tb)[inpol & (bins == b)].mean()) for b in ['success', 'near', 'mid', 'far']}}
    summ['n_episodes'] = len(keep)
    R.jdump(summ, f'{R.RES}/hist_summary.json')
    print(json.dumps(summ, indent=1))
    print('HIST_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
