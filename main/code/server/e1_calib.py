"""B E1 M1 step 1 (simulator only, no world model): train-split partition and calibration sweeps on the calibration
trajectories C (disjoint from the judge trajectories J, the auxiliary evaluation trajectories T and from val).
 (a) severity sweep: per H and pattern family, r and r_obj at a severity grid -> bisection upper bound S_MAX(H, family)
     = smallest grid severity whose median r (original definition, and separately r_obj) reaches 6.5 (cap 200 px)
 (b) bisection check with those bounds: reachability and |r - r*| per (H, definition, family)
 (c) judge-set perturbation magnitudes (R2-12 clfset mixture): iid sigma50, offset beta50, scale k50 (50% success under
     the original definition) and gross sigma_big (smallest grid value with P(r >= 3) >= 0.9)
Writes results/E1/m1/split_e1.json and calib_e1.json. usage: python e1_calib.py"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
from calib12 import GRIDS, interp_param

S_GRID = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200]
N_SWEEP, N_CHECK, N_MAG = 300, 400, 400


def sweep_job(args):
    ep, seed = args
    env = R.get_env()
    rng = np.random.RandomState(seed)
    gst, _ = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=False)
    out = {}
    for fam in E.FAMS:
        pat = E.pattern(fam, len(ep['expert']), rng)
        rr, ro = [], []
        for s in S_GRID:
            st, _ = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'] + s * pat / 100.0, render=False)
            mp = R.margin_parts(gst, st)
            rr.append(mp['r']); ro.append(mp['r_obj'])
        out[fam] = dict(r=rr, r_obj=ro)
    return out


def check_job(args):
    ep, targets, s_max, seed = args
    env = R.get_env()
    rng = np.random.RandomState(seed)
    gst, _ = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=False)
    out = []
    for key, fam, rstar in targets:
        cf = E.counterfactual(env, ep, gst, fam, rstar, key, s_max[key][fam], rng)
        if not cf['reachable']:
            out.append(dict(key=key, fam=fam, rstar=rstar, ok=False, calls=cf['calls']))
            continue
        st, _ = R.sim_run(env, ep['shape'], ep['init_state'], cf['acts'], render=False)
        mp = R.margin_parts(gst, st)
        out.append(dict(key=key, fam=fam, rstar=rstar, ok=True, r_ach=mp[key], s=cf['s'], calls=cf['calls'],
                        dom=('angle' if mp['r_ang'] > (mp['r_pos'] if key == 'r' else mp['r_obj_pos']) else
                             ('agent' if (key == 'r' and mp['agent'] > mp['block']) else 'block'))))
    return out


def mag_job(args):
    ep, fam_params, seed = args
    env = R.get_env()
    rng = np.random.RandomState(seed)
    gst, _ = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=False)
    r = []
    for name, fam, p in fam_params:
        a, _ = R.make_actions(ep['expert'], fam, p, rng)
        st, _ = R.sim_run(env, ep['shape'], ep['init_state'], a, render=False)
        r.append(R.margin_parts(gst, st)['r'])
    return r


def main():
    t0 = time.time()
    D = R.load_traj_dsets()
    dt = D['train']
    split = E.make_split(dt)
    E.jdump(split, f'{BP.RES}/split_e1.json')
    print('split', {k: (len(v) if isinstance(v, list) else v) for k, v in split.items()}, flush=True)
    pool = R.pool()
    out = dict(s_grid=S_GRID, n_sweep=N_SWEEP, n_check=N_CHECK, n_mag=N_MAG, sweep={}, s_max={}, check={}, mags={})
    for H in E.HS:
        # (a) severity sweep
        eps = E.sample_episodes_from(dt, split['C'], N_SWEEP, H, seed=5000 + H)
        res = pool.map(sweep_job, [(ep, 7000 + 100 * H + i) for i, ep in enumerate(eps)], chunksize=4)
        sm = {}
        out['sweep'][H] = {}
        for key in E.DEFS:
            sm[key] = {}
            for fam in E.FAMS:
                M = np.array([x[fam][key] for x in res])            # (n_ep, n_s)
                med = np.median(M, 0)
                idx = np.where(med >= 6.5)[0]
                sm[key][fam] = float(S_GRID[idx[0]]) if len(idx) else float(S_GRID[-1])
                out['sweep'][H][f'{key}/{fam}'] = dict(median=med.tolist(), q25=np.quantile(M, 0.25, 0).tolist(),
                                                       q75=np.quantile(M, 0.75, 0).tolist(),
                                                       frac_max_ge6=float((M.max(1) >= 6).mean()),
                                                       frac_max_ge3=float((M.max(1) >= 3).mean()),
                                                       frac_max_ge1=float((M.max(1) >= 1).mean()))
        out['s_max'][H] = sm
        print(f'H{H} S_MAX', sm, flush=True)
        # (b) bisection check
        eps = E.sample_episodes_from(dt, split['C'], N_CHECK, H, seed=5100 + H)
        rng = np.random.RandomState(5200 + H)
        jobs = []
        for i, ep in enumerate(eps):
            tg = [(key, fam, float(rng.uniform(0.8, 3.0))) for key in E.DEFS for fam in E.FAMS]
            jobs.append((ep, tg, sm, 7500 + 100 * H + i))
        res = [x for o in pool.map(check_job, jobs, chunksize=4) for x in o]
        out['check'][H] = {}
        for key in E.DEFS:
            for fam in E.FAMS:
                rr = [x for x in res if x['key'] == key and x['fam'] == fam]
                ok = [x for x in rr if x['ok']]
                err = np.array([abs(x['r_ach'] - x['rstar']) for x in ok]) if ok else np.array([np.nan])
                nm = [x for x in ok if 1 <= x['r_ach'] < 2]
                out['check'][H][f'{key}/{fam}'] = dict(
                    n=len(rr), reachable=len(ok) / max(1, len(rr)), abs_err_median=float(np.nanmedian(err)),
                    within_0p1=float(np.nanmean(err < 0.1)), mean_calls=float(np.mean([x['calls'] for x in rr])),
                    near_dominant={d: float(np.mean([x['dom'] == d for x in nm])) if nm else None for d in ('agent', 'block', 'angle')})
                print(f"H{H} {key}/{fam}: reachable {out['check'][H][f'{key}/{fam}']['reachable']:.2f} "
                      f"|r-r*| med {out['check'][H][f'{key}/{fam}']['abs_err_median']:.4f}", flush=True)
        # (c) judge-set magnitudes (original definition, as R2-12 calib12)
        eps = E.sample_episodes_from(dt, split['C'], N_MAG, H, seed=5300 + H)
        fp = []
        for fam, (key, grid) in GRIDS.items():
            for x in grid:
                if fam == 'grossbig':
                    fp.append((f'{fam}_{x}', 'gross', {'q': 1.0, 'sig_big': x, 'sig_small': 1.0}))
                else:
                    fp.append((f'{fam}_{x}', fam, {key: x}))
        r = np.array(pool.map(mag_job, [(ep, fp, 7900 + 100 * H + i) for i, ep in enumerate(eps)], chunksize=4))
        succ = {name: float((r[:, j] < 1).mean()) for j, (name, _, _) in enumerate(fp)}
        far = {name: float((r[:, j] >= 3).mean()) for j, (name, _, _) in enumerate(fp)}
        def x50(fam):
            key, grid = GRIDS[fam]
            return interp_param(grid, [succ[f'{fam}_{x}'] for x in grid], 0.5)
        sb = next((x for x in GRIDS['grossbig'][1] if far[f'grossbig_{x}'] >= 0.9), GRIDS['grossbig'][1][-1])
        out['mags'][H] = dict(sigma=x50('iid'), beta=x50('offset'), k=x50('scale'), sb=sb,
                              grid_success=succ, grid_far=far)
        print(f"H{H} mags sigma {out['mags'][H]['sigma']} beta {out['mags'][H]['beta']} k {out['mags'][H]['k']} sb {sb}", flush=True)
    pool.close()
    out['time_s'] = time.time() - t0
    E.jdump(out, f'{BP.RES}/calib_e1.json')
    print('CALIB_E1_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
