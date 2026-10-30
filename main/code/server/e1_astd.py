"""B E1 M1 step 2c: the standard-goal oracle judge a_std (K = 5 seed ensemble) used for the Diffusion-Policy episodes in
E2 (DESIGN_main v1 section 0.2: r_obj vs the fixed goal pose (256, 256, pi/4); judge = a_std ensemble). Fixed now,
before any evaluation set is imagined.
Data: judge-training seeds 300000-300599 (never the E2 evaluation seeds 100000-): every one of the 24 DP checkpoints
(seed 42/43/44 x epoch 20,25,30,40,50,80,220,1000) rolled out on its own 25 initial states; per episode 20 real frames
(half uniform in time, half at times with r_obj(t) in [0.5, 3]); goal input = one canonical goal frame (block at the goal
pose, agent at (256, 400)). 20% of episodes held out. usage: python e1_astd.py"""
import os, sys, time, json, glob
import numpy as np
import torch
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import dp_rollout as DPR
from astd_judge_e0 import frames_job
from dp_imagine import goal_state, CANON_AGENT

SEEDS_DP = (42, 43, 44)
STAGES = (20, 25, 30, 40, 50, 80, 220, 1000)
N_PER = 25
K = 20
ENS = (0, 1, 2, 3, 4)
BINS = ('success', 'near', 'mid', 'far')


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def main():
    t0 = time.time()
    RO = f'{BP.RES}/astd_rollouts'
    os.makedirs(RO, exist_ok=True)
    device = torch.device('cuda:0')
    k = 0
    for s in SEEDS_DP:
        for e in STAGES:
            o = f'{RO}/s{s}_ep{e:04d}.pt'
            seeds = list(range(300000 + N_PER * k, 300000 + N_PER * (k + 1)))
            k += 1
            if os.path.exists(o):
                continue
            pol, _ = DPR.load_policy(os.path.expanduser(f'~/pmx/B/ckpt/dp/s{s}/ema_ep{e:04d}.pt'), device)
            eps = DPR.rollout(pol, seeds, device, batch=N_PER, torch_seed=0, log=lambda *a, **kw: None)
            summ = DPR.summarize(eps)
            torch.save(dict(episodes=eps, summary=summ, seeds=seeds), o + '.tmp')
            os.replace(o + '.tmp', o)
            print(f's{s} ep{e}: success_robj {summ["success_robj"]:.2f} near {summ["near_robj"]:.2f} ({time.time() - t0:.0f}s)', flush=True)
            del pol
    eps = []
    for f in sorted(glob.glob(f'{RO}/s*_ep*.pt')):
        eps += torch.load(f, weights_only=False)['episodes']
    rng = np.random.RandomState(0)
    jobs = []
    for e in eps:
        T = e['T_end']
        ro_t = np.array([DPR.final_parts(e['states'][t])['r_obj'] for t in range(T + 1)])
        uni = rng.choice(T + 1, K // 2, replace=False)
        cand = np.where((ro_t >= 0.5) & (ro_t <= 3.0))[0]
        bnd = rng.choice(cand, min(K // 2, len(cand)), replace=False) if len(cand) else np.array([], int)
        jobs.append((e, np.unique(np.concatenate([uni, bnd, [T]])).tolist()))
    pool = R.pool()
    fr = pool.map(frames_job, jobs, chunksize=4)
    pool.close()
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    imgs = np.stack([x['img'] for f in fr for x in f]); pros = np.stack([x['pro'] for f in fr for x in f])
    ro = np.array([x['r_obj'] for f in fr for x in f])
    epi = np.array([i for i, f in enumerate(fr) for _ in f])
    z, _ = R.encode_frames(m, pre, imgs, pros)
    env = R.get_env()
    R.sim_reset(env, 'T', goal_state(CANON_AGENT))
    zgC, _ = R.encode_frames(m, pre, env._render_frame('rgb_array')[None], np.array([[*CANON_AGENT, 0, 0]], np.float32))
    zg = zgC.expand(len(z), -1, -1)
    is_val = np.random.RandomState(1).uniform(size=len(fr)) < 0.2
    va, tr = np.where(is_val[epi])[0], np.where(~is_val[epi])[0]
    y = (ro < 1).astype(np.float32)
    lr = np.log(np.maximum(ro, 1e-3))
    bv = np.array([R.r_bin(x) for x in ro[va]])
    sds, members, probs = [], [], []
    for sd in ENS:
        net, hist = R.train_judge(z[tr], zg[tr], y[tr], lr[tr], z[va], zg[va], y[va], epochs=30, seed=sd, log=lambda *a, **kw: None)
        pv = R.judge_prob(net, z[va], zg[va])
        probs.append(pv)
        sds.append({k: v.detach().cpu() for k, v in net.state_dict().items()})
        members.append(dict(seed=sd, acc=float(((pv > 0.5) == (y[va] > 0.5)).mean()), bacc=R.balanced_acc(pv > 0.5, y[va]),
                            auc_sn=auc(pv[bv == 'success'], pv[bv == 'near'])))
        print(f'a_std seed {sd}: held-out acc {members[-1]["acc"]:.4f}', flush=True)
    P = np.stack(probs)
    pe = P.mean(0)
    ens = dict(acc=float(((pe > 0.5) == (y[va] > 0.5)).mean()), bacc=R.balanced_acc(pe > 0.5, y[va]),
               per_bin={b: (float((pe[bv == b] > 0.5).mean()) if (bv == b).any() else None) for b in BINS},
               auc_sn=auc(pe[bv == 'success'], pe[bv == 'near']), auc_sf=auc(pe[bv == 'success'], pe[bv == 'far']),
               member_prob_sd_mean=float(P.std(0, ddof=1).mean()))
    ens['passes_0p95'] = bool(ens['acc'] >= 0.95)
    out = dict(n_episodes=len(fr), n_frames=int(len(z)), n_train=int(len(tr)), n_val=int(len(va)),
               bins={b: float(np.mean([R.r_bin(x) == b for x in ro])) for b in BINS},
               rollout_summaries={os.path.basename(f): torch.load(f, weights_only=False)['summary'] for f in sorted(glob.glob(f'{RO}/s*_ep*.pt'))},
               members=members, ensemble=ens, time_s=time.time() - t0)
    torch.save(dict(sds=sds, zgC=zgC, seeds=ENS), f'{BP.FEAT}/astd_e1.pt')
    R.jdump(out, f'{BP.RES}/astd_e1.json')
    print('A_STD ENSEMBLE', json.dumps(ens))
    print('ASTD_E1_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
