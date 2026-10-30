"""B E0 (sim only, no world model): feasibility of the E1 counterfactual severity grid concentrated near the boundary
(design 6.0: severity calibrated so that r is ~uniform on [0.8, 3]; >= 1000 near-miss trajectories per model).

Counterfactual = successful expert segment (DINO-WM dset-goal episodes, as R2-12) + severity s x a fixed perturbation
pattern (per episode and draw). Families: offset (s * u, u a random unit vector, constant over the segment) and iid
(s * eps_t, eps_t ~ N(0, I) per control step); s in world px. For a target r* the severity is found per counterfactual
by bisection on s (the pattern is fixed, so only the scale changes), for each of the two failure-distance definitions
(r: original evaluator incl. agent position; r_obj: object pose only). Reports reachability, |r - r*|, sim calls and
the component make-up of the achieved near-misses. Calibration episodes only (val split, seed 4040; never used for
judging or evaluation).
usage: python cf_grid_pilot.py [H=5] [n_ep=200]"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r12lib as R
import bpaths as BP

S_MAX = {'offset': 40.0, 'iid': 60.0}
N_BISECT = 14


def pattern(fam, T, rng):
    if fam == 'offset':
        u = rng.normal(size=2)
        return np.repeat((u / (np.linalg.norm(u) + 1e-12))[None], T, 0)
    return rng.normal(size=(T, 2))


def r_of(env, ep, gst, acts, key):
    st, _ = R.sim_run(env, ep['shape'], ep['init_state'], acts, render=False)
    return R.margin_parts(gst, st)[key], st


def job(args):
    ep, targets, seed = args
    env = R.get_env()
    gst, _ = R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=False)
    rng = np.random.RandomState(seed)
    out = []
    for key, fam, rstar in targets:
        pat = pattern(fam, len(ep['expert']), rng)
        f = lambda s: r_of(env, ep, gst, ep['expert'] + s * pat / 100.0, key)
        rmax, _ = f(S_MAX[fam])
        calls = 1
        if rmax < rstar:
            out.append(dict(key=key, fam=fam, rstar=rstar, ok=False, rmax=rmax, calls=calls))
            continue
        lo, hi = 0.0, S_MAX[fam]
        best = None
        for _ in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            rm, st = f(mid)
            calls += 1
            if best is None or abs(rm - rstar) < abs(best[0] - rstar):
                best = (rm, mid, st)
            if rm < rstar:
                lo = mid
            else:
                hi = mid
        mp = R.margin_parts(gst, best[2])
        out.append(dict(key=key, fam=fam, rstar=rstar, ok=True, r_ach=best[0], s=best[1], calls=calls, **mp))
    return out


def main():
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    t0 = time.time()
    D = R.load_traj_dsets()
    eps = R.sample_episodes(D['valid'], n_ep, H, seed=4040 + H)
    rng = np.random.RandomState(4041)
    jobs = []
    for i, ep in enumerate(eps):
        tg = [(key, fam, float(rng.uniform(0.8, 3.0))) for key in ('r', 'r_obj') for fam in ('offset', 'iid')]
        jobs.append((ep, tg, 9000 + i))
    pool = R.pool()
    res = [x for o in pool.map(job, jobs, chunksize=4) for x in o]
    pool.close()
    summ = {}
    for key in ('r', 'r_obj'):
        for fam in ('offset', 'iid'):
            rr = [x for x in res if x['key'] == key and x['fam'] == fam]
            ok = [x for x in rr if x['ok']]
            err = np.array([abs(x['r_ach'] - x['rstar']) for x in ok])
            ach = np.array([x['r_ach'] for x in ok])
            nm = [x for x in ok if 1 <= x['r_ach'] < 2]
            comp = {}
            if key == 'r' and nm:
                c = np.array(['angle' if x['r_ang'] > x['r_pos'] else ('agent_pos' if x['agent'] > x['block'] else 'block_pos') for x in nm])
                comp = {k: float((c == k).mean()) for k in ('agent_pos', 'block_pos', 'angle')}
            if key == 'r_obj' and nm:
                c = np.array(['angle' if x['r_ang'] > x['r_obj_pos'] else 'block_pos' for x in nm])
                comp = {k: float((c == k).mean()) for k in ('block_pos', 'angle')}
            summ[f'{key}/{fam}'] = dict(n=len(rr), reachable=len(ok) / max(1, len(rr)),
                                        abs_err_median=float(np.median(err)) if len(err) else None,
                                        abs_err_q90=float(np.quantile(err, 0.9)) if len(err) else None,
                                        within_0p1=float((err < 0.1).mean()) if len(err) else None,
                                        achieved_hist_0p8_3=np.histogram(ach, bins=[0.8, 1.0, 1.5, 2.0, 2.5, 3.0])[0].tolist() if len(ach) else None,
                                        near_share=len(nm) / max(1, len(ok)), near_composition=comp,
                                        mean_calls=float(np.mean([x['calls'] for x in rr])),
                                        s_median=float(np.median([x['s'] for x in ok])) if ok else None)
            print(f'{key}/{fam}', json.dumps(summ[f"{key}/{fam}"]), flush=True)
    out = dict(H=H, n_ep=n_ep, s_max=S_MAX, n_bisect=N_BISECT, summary=summ, time_s=time.time() - t0)
    R.jdump(out, f'{BP.RES}/cf_grid_pilot_H{H}.json')
    print('CF_GRID_PILOT_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
