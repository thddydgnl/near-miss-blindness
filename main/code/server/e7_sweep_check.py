"""B E7 post-hoc check (CPU, no GPU): recompute the calibration severity sweep of e1_calib.py with the corrected unit
(s / ACT_DIV; the E7 run recorded the sweep table with a leftover PushT '/ 100.0'). Same episodes, seeds and S_MAX rule
(smallest grid value whose median r >= 6.5, else the grid end). Writes <B_RES>/calib_e7_sweep_check.json.
usage: python e7_sweep_check.py"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
import e1_calib as C


def main():
    t0 = time.time()
    D = R.load_traj_dsets()
    dt = D['train']
    split = E.make_split(dt)
    pool = R.pool()
    out = dict(note='post-hoc CPU recomputation of the e1_calib.py severity sweep with the corrected unit (s / ACT_DIV); '
                    'same episodes and seeds (SEED_BASE + 5000 + H; workers SEED_BASE + 7000 + 100H + i); S_MAX rule: '
                    'smallest grid value whose median r >= 6.5, else the grid end', s_grid=C.S_GRID, sweep={}, s_max={})
    for H in E.HS:
        eps = E.sample_episodes_from(dt, split['C'], C.N_SWEEP, H, seed=E.SEED_BASE + 5000 + H)
        res = pool.map(C.sweep_job, [(ep, E.SEED_BASE + 7000 + 100 * H + i) for i, ep in enumerate(eps)], chunksize=4)
        out['sweep'][H], out['s_max'][H] = {}, {}
        for fam in E.FAMS:
            M = np.array([x[fam]['r'] for x in res])
            med = np.median(M, 0)
            idx = np.where(med >= 6.5)[0]
            smax = float(C.S_GRID[idx[0]]) if len(idx) else float(C.S_GRID[-1])
            out['s_max'][H][fam] = smax
            out['sweep'][H][fam] = dict(median=[round(float(x), 3) for x in med], q75=[round(float(x), 3) for x in np.quantile(M, 0.75, 0)],
                                        frac_max_ge6=float((M.max(1) >= 6).mean()), frac_max_ge3=float((M.max(1) >= 3).mean()))
            print(f"H{H} {fam}: median r per S {[round(float(x), 2) for x in med]} | max>=6 {out['sweep'][H][fam]['frac_max_ge6']:.2f} "
                  f"max>=3 {out['sweep'][H][fam]['frac_max_ge3']:.2f} -> S_MAX {smax}", flush=True)
    pool.close()
    out['time_s'] = time.time() - t0
    R.jdump(out, f'{BP.RES}/calib_e7_sweep_check.json')
    print('SWEEP_CHECK_DONE', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
