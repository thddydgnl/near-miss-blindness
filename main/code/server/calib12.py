"""Sim-only policy calibration (no world model involved). For each perturbation family, sweep the magnitude on
calibration episodes (val split, seed 101), pick the magnitude that hits the target true-success levels by linear
interpolation, then verify on fresh episodes (val split, seed 102). Writes results/policies_H{H}.json.
usage: python calib12.py <H>"""
import sys, time, json
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import r12lib as R

GRIDS = {
    'iid': ('sigma', [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]),
    'offset': ('beta_max', [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15, 20]),
    'scale': ('k_max', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]),
    'grossbig': ('sig_big', [20, 30, 45, 60, 80, 100, 130]),   # q = 1 (always corrupted)
    'clean': ('sig_small', [0.5, 1, 2]),
}
TARGETS = {'H': 0.75, 'L': 0.50}
N_CAL, N_VER = 400, 400


def seqs_for(ep, fam_params, seed):
    rng = np.random.RandomState(seed)
    out = []
    for name, fam, p in fam_params:
        a, _ = R.make_actions(ep['expert'], fam, p, rng)
        out.append((name, a))
    return out


def run(eps, fam_params, base_seed, pool):
    jobs = [(ep, seqs_for(ep, fam_params, base_seed * 100000 + i), False) for i, ep in enumerate(eps)]
    res = pool.map(R.episode_job, jobs, chunksize=4)
    r = np.array([[mp['r'] for mp in o['margins']] for o in res])     # (n_ep, n_pol)
    return r


def summ(rcol):
    rcol = np.asarray(rcol)
    return dict(success=float((rcol < 1).mean()), near=float(((rcol >= 1) & (rcol < 2)).mean()),
                mid=float(((rcol >= 2) & (rcol < 3)).mean()), far=float((rcol >= 3).mean()),
                r_q=[float(x) for x in np.quantile(rcol, [0.1, 0.25, 0.5, 0.75, 0.9])])


def interp_param(xs, succ, target):
    """success decreases with magnitude; linear interpolation between the grid points bracketing the target."""
    xs, succ = np.asarray(xs, float), np.asarray(succ, float)
    for i in range(len(xs) - 1):
        if (succ[i] - target) * (succ[i + 1] - target) <= 0 and succ[i] != succ[i + 1]:
            return float(xs[i] + (target - succ[i]) * (xs[i + 1] - xs[i]) / (succ[i + 1] - succ[i]))
    return None


def main():
    H = int(sys.argv[1])
    t0 = time.time()
    D = R.load_traj_dsets()
    dv = D['valid']
    eps_c = R.sample_episodes(dv, N_CAL, H, seed=101 + 1000 * H)
    eps_v = R.sample_episodes(dv, N_VER, H, seed=102 + 1000 * H)
    fam_params = []
    for fam, (key, grid) in GRIDS.items():
        for x in grid:
            if fam == 'grossbig':
                fam_params.append((f'{fam}_{x}', 'gross', {'q': 1.0, 'sig_big': x, 'sig_small': 1.0}))
            else:
                fam_params.append((f'{fam}_{x}', fam, {key: x}))
    pool = R.pool()
    r = run(eps_c, fam_params, 101, pool)
    grid_out = {}
    for j, (name, fam, p) in enumerate(fam_params):
        grid_out[name] = dict(fam=fam, p=p, **summ(r[:, j]))
        print(f'{name:16s} succ {grid_out[name]["success"]:.3f} near {grid_out[name]["near"]:.3f} '
              f'mid {grid_out[name]["mid"]:.3f} far {grid_out[name]["far"]:.3f} r_med {grid_out[name]["r_q"][2]:.2f}',
              flush=True)
    # choose (pre-registered policy set, PLAN.md section 3)
    def x_for(fam, tgt):
        key, grid = GRIDS[fam]
        return interp_param(grid, [grid_out[f'{fam}_{x}']['success'] for x in grid], tgt)
    sb = None
    for x in GRIDS['grossbig'][1]:
        if grid_out[f'grossbig_{x}']['far'] >= 0.9:
            sb = x
            break
    s_clean = grid_out['clean_1']['success']
    b50, b60, b75 = x_for('offset', 0.50), x_for('offset', 0.60), x_for('offset', 0.75)
    chosen = {
        'far_L': dict(fam='mix', p={'q': round(1 - 0.50 / s_clean, 4), 'sig_big': sb, 'beta_max': 0.0, 'sig_small': 1.0}, target=0.50),
        'mixA_L': dict(fam='mix', p={'q': round(1 / 3, 4), 'sig_big': sb, 'beta_max': round(b75, 4), 'sig_small': 0.0}, target=0.50),
        'mixB_L': dict(fam='mix', p={'q': round(1 / 6, 4), 'sig_big': sb, 'beta_max': round(b60, 4), 'sig_small': 0.0}, target=0.50),
        'near_L': dict(fam='offset', p={'beta_max': round(b50, 4)}, target=0.50),
        'iid_L': dict(fam='iid', p={'sigma': round(x_for('iid', 0.50), 4)}, target=0.50),
        'far_H': dict(fam='mix', p={'q': round(1 - 0.75 / s_clean, 4), 'sig_big': sb, 'beta_max': 0.0, 'sig_small': 1.0}, target=0.75),
        'near_H': dict(fam='offset', p={'beta_max': round(b75, 4)}, target=0.75),
        'iid_H': dict(fam='iid', p={'sigma': round(x_for('iid', 0.75), 4)}, target=0.75),
    }
    # verify on fresh episodes
    names = sorted(chosen)
    fp = [(n, chosen[n]['fam'], chosen[n]['p']) for n in names]
    rv = run(eps_v, fp, 102, pool)
    for j, n in enumerate(names):
        chosen[n]['verify'] = summ(rv[:, j])
        print(f'VERIFY {n:10s} {chosen[n]["p"]} succ {chosen[n]["verify"]["success"]:.3f} '
              f'near {chosen[n]["verify"]["near"]:.3f} mid {chosen[n]["verify"]["mid"]:.3f} far {chosen[n]["verify"]["far"]:.3f}',
              flush=True)
    pool.close()
    R.jdump(dict(H=H, n_cal=N_CAL, n_ver=N_VER, grid=grid_out, chosen=chosen, clean_success=s_clean, sig_big=sb,
                 time_s=time.time() - t0), f'{R.RES}/policies_H{H}.json')
    print('CALIB_DONE', H, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
