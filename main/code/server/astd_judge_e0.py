"""B E0 feasibility (descriptive; not a hypothesis test): an oracle discriminator for the STANDARD PushT goal used by the
Diffusion-Policy episodes (fixed goal pose (256,256,pi/4), object-pose-only success r_obj < 1).

1. Judge-training frames: E0 judge-training rollouts (seeds 300000-, several DP stages of seed 42). Per episode K frames:
   half at uniform times, half at times with r_obj(t) in [0.5, 3] (boundary enrichment); r_obj(t) from recorded states.
2. Real frames rendered by exact replay, encoded with the DINO-WM encoder (DINOv2 patch features).
3. Goal input: one canonical goal frame (block at the goal pose, agent at (256, 400)); the target region is drawn in every
   frame anyway. Same PairJudge / training rule as R2-12 (a); 20% of episodes held out.
4. Applied to E0 demo windows (dp_imagine.py output; other seeds, other seed of DP): P(judged success | r_obj band) and
   AUROC success-vs-near / success-vs-far on real vs imagined final frames, per horizon.
usage: python astd_judge_e0.py <imagine_out.pt>"""
import sys, os, json, time, glob
import numpy as np
import torch
from scipy.stats import mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
from dp_rollout import step_rel, state7, final_parts
from dp_imagine import goal_state, CANON_AGENT

K = 20
BINS = ['success', 'near', 'mid', 'far']


def frames_job(args):
    ep, times = args
    torch.set_num_threads(1)
    env = R.get_env()
    R.sim_reset(env, 'T', ep['init'])
    out, t = [], 0
    for tt in sorted(times):
        while t < tt:
            step_rel(env, ep['a_rel'][t])
            t += 1
        st = state7(env)
        out.append(dict(t=tt, img=env._render_frame('rgb_array'), pro=np.concatenate([st[:2], st[5:]]).astype(np.float32),
                        **final_parts(st)))
    return out


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def main():
    t0 = time.time()
    rng = np.random.RandomState(0)
    eps = []
    for f in sorted(glob.glob(f'{BP.RES}/rollouts/jt_*.pt')):
        eps += torch.load(f, weights_only=False)['episodes']
    jobs = []
    for e in eps:
        T = e['T_end']
        ro_t = np.array([final_parts(e['states'][t])['r_obj'] for t in range(T + 1)])
        uni = rng.choice(T + 1, K // 2, replace=False)
        cand = np.where((ro_t >= 0.5) & (ro_t <= 3.0))[0]
        bnd = rng.choice(cand, min(K // 2, len(cand)), replace=False) if len(cand) else np.array([], int)
        times = np.unique(np.concatenate([uni, bnd, [T]]))
        jobs.append((e, times.tolist()))
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
    net, hist = R.train_judge(z[tr], zg[tr], y[tr], lr[tr], z[va], zg[va], y[va], epochs=30, seed=0, log=lambda *a, **k: None)
    pv = R.judge_prob(net, z[va], zg[va])
    bv = np.array([R.r_bin(x) for x in ro[va]])
    out = dict(n_episodes=len(fr), n_frames=int(len(z)), frac_success=float(y.mean()),
               bins={b: float(np.mean([R.r_bin(x) == b for x in ro])) for b in BINS},
               heldout_acc=float(((pv > 0.5) == (y[va] > 0.5)).mean()), heldout_bacc=R.balanced_acc(pv > 0.5, y[va]),
               heldout_per_bin={b: float((pv[bv == b] > 0.5).mean()) if (bv == b).any() else None for b in BINS},
               heldout_auc_sn=auc(pv[bv == 'success'], pv[bv == 'near']))
    print('A_STD', json.dumps(out), flush=True)
    # demo windows: real vs imagined
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        Dm = torch.load(sys.argv[1], weights_only=False)
        recs = Dm['recs']
        zgw = zgC.expand(len(recs), -1, -1)
        pr, pi = R.judge_prob(net, Dm['zr'], zgw), R.judge_prob(net, Dm['zi'], zgw)
        demo = {}
        for H in Dm['Hs']:
            ix = np.array([k for k, r in enumerate(recs) if r['H'] == H])
            b = np.array([R.r_bin(recs[k]['r_obj']) for k in ix])
            prh, pih = pr[ix], pi[ix]
            demo[f'H{H}'] = dict(n={bb: int((b == bb).sum()) for bb in BINS},
                                 p_real={bb: (float((prh[b == bb] > 0.5).mean()) if (b == bb).any() else None) for bb in BINS},
                                 p_imag={bb: (float((pih[b == bb] > 0.5).mean()) if (b == bb).any() else None) for bb in BINS},
                                 auc_sn_real=auc(prh[b == 'success'], prh[b == 'near']), auc_sn_imag=auc(pih[b == 'success'], pih[b == 'near']),
                                 auc_sf_real=auc(prh[b == 'success'], prh[b == 'far']), auc_sf_imag=auc(pih[b == 'success'], pih[b == 'far']),
                                 med_dv_imag_vs_real={bb: (float(np.median([recs[k]['dv_imag_vs_real'] for k, x in zip(ix, b) if x == bb]))
                                                           if (b == bb).any() else None) for bb in BINS})
            print(f'H{H}', json.dumps(demo[f'H{H}']), flush=True)
        out['demo'] = demo
    out['time_s'] = time.time() - t0
    R.jdump(out, f'{BP.RES}/astd_judge_e0.json')
    print('ASTD_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
