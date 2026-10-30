"""B E4a / E5 (M1): fine-tuning data for the public DINO-WM PushT predictor (DESIGN_main v3.1 section 9).
Pool: 3,000 DINO-WM training-unit windows (4 frames at frameskip 5, 20 actions; seed 20261101) on the train-split
trajectories F (not J, C, T of E1). Per window: expert actions + 7 perturbations from the E1 judge-set mixture (C-calibrated
H3 magnitudes, multiplier log-U[0.3, 3]; seed 20261102). Label: original-definition r at step 15 vs the expert state at
step 15 (success = goal reached, r < 1).
Selections (seed 20261103): S97 = 9,700 successes + 300 failures, S50 = 5,000 successes (subset of S97's) + 5,000
failures (superset of S97's).
E5 counterfactuals (seed 20261104; bisection 14 steps on a pattern fixed per window, E1 grid machinery, offset / iid half
each, bounds = E1 calibration for H3): 300 windows with a near-boundary failure r* ~ U[1, 2) and its pair success
r* ~ U[0.8, 1) on the same pattern ((ii) / (iii)); 300 other windows with a full-range failure r* ~ U[1, 6] ((iv)).
Counterfactual windows are drawn from the start windows of S97 successes; actions 16-20 are the expert's (not used by the loss).
Stored (feat/E4/m1): start features per window, features of frames 1-3 per sequence (fp16), normalised proprio / actions,
labels, block / agent patch masks of frames 1-3, selections. usage: python e4_m1_ftdata.py"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
import e4lib as E4
from e1_judgeset import make_seqs

N_WIN = 3000
N_S97_S, N_S97_F, N_S50 = 9700, 300, 5000
N_CF = 300


def label_job(args):
    """CPU worker: simulate the 8 variants of one window (no rendering) -> states at steps 0, 5, 10, 15."""
    ep, seqs = args
    torch.set_num_threads(1)
    env = R.get_env()
    return [E4.run_frames(env, ep['shape'], ep['init_state'], a, render=False)[0] for a in seqs]


def bisect_pattern(env, ep, gst, pat, rstar, s_max, n_bisect=E.N_BISECT):
    """as e1lib.counterfactual (key 'r') but with a given pattern, on the first 15 actions."""
    exp15 = ep['expert'][:E4.LABEL_STEP]

    def f(s):
        st, _ = R.sim_run(env, ep['shape'], ep['init_state'], exp15 + s * pat / 100.0, render=False)
        return R.margin_parts(gst, st)['r']
    if f(s_max) < rstar:
        return None
    lo, hi, best = 0.0, s_max, None
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        rm = f(mid)
        if best is None or abs(rm - rstar) < abs(best[0] - rstar):
            best = (rm, mid)
        if rm < rstar:
            lo = mid
        else:
            hi = mid
    return best[1]


def cf_job(args):
    """CPU worker: one window, list of targets [(tag, rstar)], one pattern (family fam) shared by the targets.
    Returns per target the 20-action sequence (or None if unreachable)."""
    ep, fam, targets, s_max, seed = args
    torch.set_num_threads(1)
    env = R.get_env()
    rng = np.random.RandomState(seed)
    pat = E.pattern(fam, E4.LABEL_STEP, rng)
    gst, _ = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'][:E4.LABEL_STEP], render=False)
    out = []
    for tag, rstar in targets:
        s = bisect_pattern(env, ep, gst, pat, rstar, s_max[fam])
        if s is None:
            out.append(None)
            continue
        acts = ep['expert'].copy()
        acts[:E4.LABEL_STEP] = acts[:E4.LABEL_STEP] + s * pat / 100.0
        out.append(dict(tag=tag, rstar=rstar, s=float(s), fam=fam, acts=acts))
    return out


def render_job(args):
    ep, seqs = args
    torch.set_num_threads(1)
    env = R.get_env()
    out = []
    for a in seqs:
        st, imgs, masks = E4.run_frames(env, ep['shape'], ep['init_state'], a, render=True)
        out.append(dict(states=st, imgs=imgs, masks=masks))
    s0, img0, _ = R.start_obs(env, ep['shape'], ep['init_state'])
    return dict(start_img=img0, start_state=s0, seqs=out)


def main():
    t0 = time.time()
    out_dir = BP.FEAT
    os.makedirs(out_dir, exist_ok=True)
    cal = json.load(open(f'{E4.E1_RES}/calib_e1.json'))
    M = cal['mags']['3']
    s_max = cal['s_max']['3']['r']
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    dt = T_['dset_train']
    F_trajs = E4.f_split(dt)
    wins = E4.sample_windows(dt, F_trajs, N_WIN, seed=20261101)
    rng = np.random.RandomState(20261102)
    seqs_all, fams_all = [], []
    for ep in wins:
        s, f = make_seqs(ep, M, rng)
        seqs_all.append([a for _, a in s])
        fams_all.append(f)
    pool = R.pool()
    states = pool.map(label_job, [(ep, s) for ep, s in zip(wins, seqs_all)], chunksize=8)
    print(f'labels: {N_WIN} windows x 8 variants simulated ({time.time() - t0:.0f}s)', flush=True)
    rows = []                                   # (window, variant, r, r_obj)
    for w, sts in enumerate(states):
        g = sts[0][-1]                          # expert state at step 15
        for k, st in enumerate(sts):
            mp = R.margin_parts(g, st[-1])
            rows.append((w, k, mp['r'], mp['r_obj']))
    rows = np.array(rows)
    succ = rows[:, 2] < 1
    rs = np.random.RandomState(20261103)
    i_s = np.where(succ)[0]
    i_f = np.where(~succ)[0]
    s97_s = rs.choice(i_s, N_S97_S, replace=False)
    s50_f = rs.choice(i_f, N_S50, replace=False)
    s97_f = rs.choice(s50_f, N_S97_F, replace=False)
    s50_s = rs.choice(s97_s, N_S50, replace=False)
    base_sel = np.unique(np.concatenate([s97_s, s50_f]))
    print(f'pool: {len(rows)} sequences, success fraction {succ.mean():.3f}; rendering {len(base_sel)} base sequences', flush=True)
    # E5 counterfactual windows: start windows of S97 successes, two disjoint sets of N_CF
    rc = np.random.RandomState(20261104)
    cand = rc.permutation(np.unique(rows[s97_s, 0].astype(int)))
    got = {'near': [], 'full': []}
    ci = 0
    for kind in ('near', 'full'):
        while len(got[kind]) < N_CF:
            assert ci < len(cand), 'ran out of candidate windows'
            batch = []
            while len(batch) < 64 and ci < len(cand):
                w = int(cand[ci]); ci += 1
                fam = 'offset' if rc.uniform() < 0.5 else 'iid'
                if kind == 'near':
                    tg = [('cf_near', float(rc.uniform(1.0, 2.0))), ('cf_pair', float(rc.uniform(0.8, 1.0)))]
                else:
                    tg = [('cf_full', float(rc.uniform(1.0, 6.0)))]
                batch.append((w, fam, tg, int(rc.randint(0, 2 ** 31 - 1))))
            res = pool.map(cf_job, [(wins[w], fam, tg, s_max, sd) for w, fam, tg, sd in batch], chunksize=2)
            for (w, fam, tg, sd), o in zip(batch, res):
                if any(x is None for x in o) or len(got[kind]) >= N_CF:
                    continue
                got[kind].append((w, o))
    assert len(got['near']) == N_CF and len(got['full']) == N_CF, {k: len(v) for k, v in got.items()}
    print(f'counterfactual windows: near {len(got["near"])}, full {len(got["full"])} (tried {ci}) ({time.time() - t0:.0f}s)', flush=True)
    # sequences to render: base selections + counterfactuals; grouped by window
    seq_list = []                      # (window, acts, kind, variant, rstar, s, fam)
    for i in base_sel:
        w, k = int(rows[i, 0]), int(rows[i, 1])
        seq_list.append((w, seqs_all[w][k], 'expert' if k == 0 else 'mix_' + fams_all[w][k], int(i), None, None, None))
    for kind in ('near', 'full'):
        for w, o in got[kind]:
            for x in o:
                seq_list.append((w, x['acts'], x['tag'], -1, x['rstar'], x['s'], x['fam']))
    by_win = {}
    for j, s in enumerate(seq_list):
        by_win.setdefault(s[0], []).append(j)
    win_ids = sorted(by_win)
    res = pool.map(render_job, [(wins[w], [seq_list[j][1] for j in by_win[w]]) for w in win_ids], chunksize=4)
    pool.close()
    print(f'rendered {len(seq_list)} sequences in {len(win_ids)} windows ({time.time() - t0:.0f}s)', flush=True)
    n_seq = len(seq_list)
    wpos = {w: k for k, w in enumerate(win_ids)}
    start_img = np.stack([r_['start_img'] for r_ in res])
    start_st = np.stack([r_['start_state'] for r_ in res])
    zs, _ = R.encode_frames(m, pre, start_img, E4.proprio_of(start_st))
    vis = np.zeros((n_seq, 3, 196, 384), np.float16)
    pro = np.zeros((n_seq, 4, 4), np.float32)
    act = np.zeros((n_seq, 4, 10), np.float32)
    bmask = np.zeros((n_seq, 3, 196), bool)
    amask = np.zeros((n_seq, 3, 196), bool)
    lab = np.zeros((n_seq, 2))
    win_of = np.zeros(n_seq, int)
    for w, r_ in zip(win_ids, res):
        imgs = np.stack([im for s in r_['seqs'] for im in s['imgs']])
        prs = np.stack([E4.proprio_of(s['states'][1:]) for s in r_['seqs']]).reshape(-1, 4)
        z, _ = R.encode_frames(m, pre, imgs, prs)
        z = z.numpy().reshape(len(r_['seqs']), 3, 196, 384)
        for q, (j, s) in enumerate(zip(by_win[w], r_['seqs'])):
            vis[j] = z[q]
            pro[j] = pre.normalize_proprios(torch.as_tensor(E4.proprio_of(s['states']))).numpy()
            a = torch.as_tensor(np.asarray(seq_list[j][1]), dtype=torch.float32)
            act[j] = pre.normalize_actions(a).reshape(4, 10).numpy()
            bmask[j] = np.stack([mm[0] for mm in s['masks']])
            amask[j] = np.stack([mm[1] for mm in s['masks']])
            win_of[j] = wpos[w]
    # labels vs the expert state at step 15 of each window (recomputed from the expert's simulation)
    env = R.get_env()
    gst = {}
    for j, s in enumerate(seq_list):
        w = s[0]
        if w not in gst:
            gst[w] = E4.run_frames(env, wins[w]['shape'], wins[w]['init_state'], wins[w]['expert'], render=False)[0][-1]
    finals = {}
    for w, r_ in zip(win_ids, res):
        for j, s in zip(by_win[w], r_['seqs']):
            finals[j] = s['states'][-1]
    for j, s in enumerate(seq_list):
        mp = R.margin_parts(gst[s[0]], finals[j])
        lab[j] = (mp['r'], mp['r_obj'])
    kinds = np.array([s[2] for s in seq_list])
    pool_idx = np.array([s[3] for s in seq_list])
    pos = {int(p): j for j, p in enumerate(pool_idx) if p >= 0}
    sel = dict(S97_s=np.array([pos[int(i)] for i in s97_s]), S97_f=np.array([pos[int(i)] for i in s97_f]),
               S50_s=np.array([pos[int(i)] for i in s50_s]), S50_f=np.array([pos[int(i)] for i in s50_f]),
               cf_near=np.where(kinds == 'cf_near')[0], cf_pair=np.where(kinds == 'cf_pair')[0], cf_full=np.where(kinds == 'cf_full')[0])
    # sanity: pool labels reproduce, counterfactual targets reached
    chk = dict(max_abs_r_vs_pool=float(max(abs(lab[pos[int(i)], 0] - rows[int(i), 2]) for i in base_sel)),
               S97_f_all_fail=bool((lab[sel['S97_f'], 0] >= 1).all()), S97_s_all_succ=bool((lab[sel['S97_s'], 0] < 1).all()),
               cf_near_r=[float(np.min(lab[sel['cf_near'], 0])), float(np.max(lab[sel['cf_near'], 0]))],
               cf_pair_r=[float(np.min(lab[sel['cf_pair'], 0])), float(np.max(lab[sel['cf_pair'], 0]))],
               cf_full_r=[float(np.min(lab[sel['cf_full'], 0])), float(np.max(lab[sel['cf_full'], 0]))])
    np.save(f'{out_dir}/ft_vis.npy', vis)
    np.save(f'{out_dir}/ft_start_vis.npy', zs.numpy())
    np.savez(f'{out_dir}/ft_meta.npz', pro=pro, act=act, bmask=bmask, amask=amask, lab=lab, win_of=win_of, kinds=kinds,
             pool_idx=pool_idx, win_traj=np.array([wins[w]['traj'] for w in win_ids]), win_off=np.array([wins[w]['offset'] for w in win_ids]),
             **{f'sel_{k}': v for k, v in sel.items()})
    summ = dict(n_windows_pool=N_WIN, n_pool_sequences=int(len(rows)), pool_success_frac=float(succ.mean()),
                pool_band_frac={b: float(np.mean([R.r_bin(x) == b for x in rows[:, 2]])) for b in ('success', 'near', 'mid', 'far')},
                n_rendered=int(n_seq), n_windows_rendered=len(win_ids), sizes={k: int(len(v)) for k, v in sel.items()},
                S97_f_bands={b: float(np.mean([R.r_bin(x) == b for x in lab[sel['S97_f'], 0]])) for b in ('near', 'mid', 'far')},
                S50_f_bands={b: float(np.mean([R.r_bin(x) == b for x in lab[sel['S50_f'], 0]])) for b in ('near', 'mid', 'far')},
                S97_s_expert_frac=float(np.mean(kinds[sel['S97_s']] == 'expert')), checks=chk, time_s=time.time() - t0)
    R.jdump(summ, f'{BP.RES}/ft_data_meta.json')
    print('E4_FTDATA_DONE', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
