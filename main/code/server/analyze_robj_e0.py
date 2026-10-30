"""B E0 diagnostic (descriptive, not a hypothesis test): the R2-12 evaluation sets under the two PushT failure-distance
definitions (r: evaluator incl. agent position; r_obj: object pose only), with per-component make-up.
Judges (a), (b) were trained / thresholded on the original labels, so P(judged success | r_obj bin) is descriptive only.
usage: python analyze_robj_e0.py [sets...]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
from analyze12 import load, flags, POL

BINS = ['success', 'near', 'mid', 'far']


def main():
    sets = sys.argv[1:] or ['main', 'H10', 'trainsplit']
    J = json.load(open(f'{BP.RES}/judges.json'))
    out = {}
    for s in sets:
        recs = load(s)
        if recs is None:
            continue
        F = flags(recs, J)
        g = lambda k: np.array([x[k] for x in recs])
        pol, b, bo = g('policy'), g('bin'), g('bin_obj')
        inpol = np.isin(pol, POL)
        ct = {bb: {cc: int(((b == bb) & (bo == cc) & inpol).sum()) for cc in BINS} for bb in BINS}
        judged = {}
        for j in ('a', 'b'):
            judged[j] = {f'{v}': {cc: (float(F[f'{j}_{v}'][inpol & (bo == cc)].mean()) if (inpol & (bo == cc)).any() else None)
                                  for cc in BINS} for v in ('imag', 'real')}
        rp, ra, ag, bl, rop = g('r_pos'), g('r_ang'), g('agent'), g('block'), g('r_obj_pos')
        comp_orig = np.where(ra > rp, 'angle', np.where(ag ** 2 > bl ** 2, 'agent_pos', 'block_pos'))
        comp_obj = np.where(ra > rop, 'angle', 'block_pos')
        nm, nmo = inpol & (b == 'near'), inpol & (bo == 'near')
        per_pol = {p: dict(success=float(g('success')[pol == p].mean()), success_obj=float(g('success_obj')[pol == p].mean()),
                           near=float((b[pol == p] == 'near').mean()), near_obj=float((bo[pol == p] == 'near').mean()),
                           far_obj=float((bo[pol == p] == 'far').mean())) for p in POL}
        out[s] = dict(n=int(inpol.sum()), crosstab_orig_rows_obj_cols=ct,
                      p_judged_success_by_obj_bin=judged,
                      near_orig_composition={c: int((nm & (comp_orig == c)).sum()) for c in ('agent_pos', 'block_pos', 'angle')},
                      near_obj_composition={c: int((nmo & (comp_obj == c)).sum()) for c in ('block_pos', 'angle')},
                      per_policy=per_pol)
        print(s, json.dumps({k: v for k, v in out[s].items() if k != 'per_policy'}))
        for p, d in per_pol.items():
            print(f'  {p:7s} ' + ' '.join(f'{k} {v:.3f}' for k, v in d.items()))
    json.dump(out, open(f'{BP.RES}/robj_e0.json', 'w'), indent=1)
    print('ROBJ_DONE')


if __name__ == '__main__':
    main()
