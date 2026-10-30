"""B E3 M1, judge (d) rule set for the standard goal (DP episodes; DESIGN_main v2.2 section 7): the E1 a_std
judge-training frames are regenerated exactly (same rollouts astd_rollouts/*.pt, same frame-time draws RandomState(0),
same held-out episode mask RandomState(1); counts checked against astd_e1.json). Selected rows: all held-out frames
(criterion set) and 2,000 training frames drawn with seed 20261022 (threshold set). The frames and the canonical goal
frame are encoded with DINOv2 and decoded by the checkpoint's VQ-VAE decoder (uint8 224 px).
Writes feat/E3/m1/rules_std.npz and results/E3/m1/rules_std_meta.json. usage: python e3_rules_std.py"""
import os, sys, time, json, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import declib as D
import dp_rollout as DPR
from astd_judge_e0 import frames_job

E1_RES = os.path.expanduser('~/pmx/B/results/E1/m1')
E1_FEAT = os.path.expanduser('~/pmx/B/feat/E1/m1')
K = 20
N_TRAIN_ROWS = 2000


def main():
    t0 = time.time()
    eps = []
    for f in sorted(glob.glob(f'{E1_RES}/astd_rollouts/s*_ep*.pt')):
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
    flat = [x for f in fr for x in f]
    ro = np.array([x['r_obj'] for x in flat])
    epi = np.array([i for i, f in enumerate(fr) for _ in f])
    is_val = np.random.RandomState(1).uniform(size=len(fr)) < 0.2
    va, tr = np.where(is_val[epi])[0], np.where(~is_val[epi])[0]
    ref = json.load(open(f'{E1_RES}/astd_e1.json'))
    ident = dict(n_frames=int(len(ro)) == ref['n_frames'], n_val=int(len(va)) == ref['n_val'], n_train=int(len(tr)) == ref['n_train'],
                 bins_equal={b: float(np.mean([R.r_bin(x) == b for x in ro])) for b in ('success', 'near', 'mid', 'far')} == ref['bins'])
    print('identity vs E1 a_std frames:', json.dumps(ident), flush=True)
    assert all(ident.values()), 'regenerated a_std frames differ from E1'
    pick = np.sort(np.random.RandomState(20261022).choice(tr, N_TRAIN_ROWS, replace=False))
    rows = np.concatenate([va, pick])
    split = np.array(['held'] * len(va) + ['train'] * len(pick))
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    z, _ = R.encode_frames(m, pre, np.stack([flat[i]['img'] for i in rows]), np.stack([flat[i]['pro'] for i in rows]))
    A = torch.load(f'{E1_FEAT}/astd_e1.pt', weights_only=False)
    goal_dec = D.decode(A['zgC'])[0]
    final_dec = D.decode(z, bs=32)
    os.makedirs(BP.FEAT, exist_ok=True)
    np.savez_compressed(f'{BP.FEAT}/rules_std.npz', goal=goal_dec, final=final_dec, r_obj=ro[rows], ep=epi[rows],
                        t=np.array([flat[i]['t'] for i in rows]), split=split)
    summ = dict(identity=ident, n_rows=int(len(rows)), n_held=int(len(va)), n_train=int(len(pick)),
                held_success_frac=float((ro[va] < 1).mean()), time_s=time.time() - t0)
    R.jdump(summ, f'{BP.RES}/rules_std_meta.json')
    print('E3_RULES_STD_DONE', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
