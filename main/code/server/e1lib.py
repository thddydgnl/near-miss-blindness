"""B E1 (M1) shared helpers: train-split partition, episode sampling from a trajectory subset, the counterfactual severity
grid (scalar severity x fixed pattern, bisection to a target failure distance), and CPU worker jobs.
Frozen by DESIGN_main v1 section 0.2: H in {3,5,7,10} x definition in {r, r_obj}; per (H, def) 3,000 targets stratified
uniform on [0.8, 3] + 300 in [0.2, 0.8) + 300 in [3, 6]; bisection 14 steps; patterns offset / iid half each;
calibration sweep on the train split (trajectories not used by the judges)."""
import os, sys, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R

HS = (3, 5, 7, 10)
DEFS = ('r', 'r_obj')
FAMS = ('offset', 'iid')
N_BISECT = 14
SPLIT_SEED = 20261021
N_J, N_C, N_T = 2000, 1000, 3000          # judge / calibration / trainsplit-auxiliary trajectories (disjoint)


def make_split(dset_train, min_len=52):
    """Disjoint train-split trajectory sets J (judges), C (severity calibration), T (auxiliary evaluation)."""
    lens = np.array([dset_train.get_seq_length(i) for i in range(len(dset_train))])
    ok = np.where(lens >= min_len)[0]
    perm = np.random.RandomState(SPLIT_SEED).permutation(ok)
    return dict(J=sorted(perm[:N_J].tolist()), C=sorted(perm[N_J:N_J + N_C].tolist()),
                T=sorted(perm[N_J + N_C:N_J + N_C + N_T].tolist()), n_eligible=int(len(ok)), min_len=min_len,
                seed=SPLIT_SEED)


def sample_episodes_from(dset, trajs, n, H, seed):
    """n episodes (trajectory uniform from trajs, offset uniform with offset <= len - H*FS - 1), as R.sample_episodes."""
    rng = np.random.RandomState(seed)
    T = H * R.FS
    trajs = np.asarray([t for t in trajs if dset.get_seq_length(int(t)) - T - 1 >= 0])
    amean, astd = dset.action_mean.numpy(), dset.action_std.numpy()
    eps = []
    for _ in range(n):
        bi = int(trajs[rng.randint(len(trajs))])
        L = dset.get_seq_length(bi)
        off = int(rng.randint(0, L - T))
        acts = dset.actions[bi, off:off + T].numpy() * astd + amean
        eps.append(dict(traj=bi, offset=off, shape=dset.shapes[bi], H=H,
                        init_state=dset.states[bi, off].numpy().astype(np.float64), expert=acts.astype(np.float64)))
    return eps


def stratified_targets(rng):
    """3,600 targets: 3,000 stratified uniform on [0.8, 3], 300 on [0.2, 0.8), 300 on [3, 6]; shuffled."""
    def strat(lo, hi, n):
        return lo + (hi - lo) * (np.arange(n) + rng.uniform(size=n)) / n
    t = np.concatenate([strat(0.8, 3.0, 3000), strat(0.2, 0.8, 300), strat(3.0, 6.0, 300)])
    fam = np.array(['offset'] * 1800 + ['iid'] * 1800)
    rng.shuffle(fam)
    perm = rng.permutation(len(t))
    return t[perm], fam


def pattern(fam, T, rng):
    if fam == 'offset':
        u = rng.normal(size=2)
        return np.repeat((u / (np.linalg.norm(u) + 1e-12))[None], T, 0)
    return rng.normal(size=(T, 2))


def counterfactual(env, ep, gst, fam, rstar, key, s_max, rng, n_bisect=N_BISECT):
    """Bisection on the scalar severity s (world px) of a fixed pattern so that margin[key](s) ~ rstar.
    Returns dict(acts, s, reachable, rmax, calls). If r(s_max) < rstar the target is unreachable on this episode."""
    pat = pattern(fam, len(ep['expert']), rng)

    def f(s):
        st, _ = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'] + s * pat / 100.0, render=False)
        return R.margin_parts(gst, st)[key]
    rmax = f(s_max)
    calls = 1
    if rmax < rstar:
        return dict(acts=None, s=None, reachable=False, rmax=float(rmax), calls=calls)
    lo, hi, best = 0.0, s_max, None
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        rm = f(mid)
        calls += 1
        if best is None or abs(rm - rstar) < abs(best[0] - rstar):
            best = (rm, mid)
        if rm < rstar:
            lo = mid
        else:
            hi = mid
    s = best[1]
    return dict(acts=ep['expert'] + s * pat / 100.0, s=float(s), reachable=True, rmax=float(rmax), calls=calls)


def grid_episode_job(args):
    """CPU worker: one episode with several targets. args = (ep, targets [(tid, rstar, fam)], key, s_max dict, seed, render).
    Returns start/goal renders, per-target counterfactual results (final state, render, margins) and unreachable ids."""
    ep, targets, key, s_max, seed, render = args
    torch.set_num_threads(1)
    env = R.get_env()
    rng = np.random.RandomState(seed)
    gst, gimg = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=render)
    out = dict(goal_state=gst, goal_img=gimg, goal_proprio=np.concatenate([gst[:2], gst[5:]]).astype(np.float32),
               done=[], unreachable=[])
    if render:
        s0, img0, p0 = R.start_obs(env, ep['shape'], ep['init_state'])
        out['start_img'], out['start_proprio'] = img0, p0
        # expert reference (r = 0)
        out['expert'] = dict(final_state=gst, final_img=gimg, final_proprio=out['goal_proprio'],
                             margins=R.margin_parts(gst, gst), acts=ep['expert'])
    for tid, rstar, fam in targets:
        cf = counterfactual(env, ep, gst, fam, rstar, key, s_max[fam], rng)
        if not cf['reachable']:
            out['unreachable'].append(dict(tid=tid, rstar=rstar, fam=fam, rmax=cf['rmax']))
            continue
        st, img = R.sim_run(env, ep['shape'], ep['init_state'], cf['acts'], render=render)
        out['done'].append(dict(tid=tid, rstar=rstar, fam=fam, s=cf['s'], calls=cf['calls'], final_state=st, final_img=img,
                                final_proprio=np.concatenate([st[:2], st[5:]]).astype(np.float32),
                                margins=R.margin_parts(gst, st), acts=cf['acts']))
    return out


def load_ensemble(sds, d=128, layers=2):
    nets = []
    for sd in sds:
        net = R.PairJudge(d=d, layers=layers).cuda()
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    return nets


@torch.no_grad()
def ens_prob(nets, z, zg):
    """member probabilities (K, N) and ensemble mean (N,)."""
    P = np.stack([R.judge_prob(n, z, zg) for n in nets])
    return P, P.mean(0)


def jdump(obj, path):
    R.jdump(obj, path)
