"""B E3 M1, judge (d) rule set for dset goals (DESIGN_main v2.1 section 6, v2.2 section 7): the E1 judge-training set on the
judge trajectories J is regenerated exactly (same episode sampling, seeds and perturbation mixture as e1_judgeset.py;
checked against judgeset_e1_meta.json). Selected rows: every sequence of the held-out episodes (20% of episodes, the same
mask as E1; criterion set) and 2,000 sequences drawn with seed 20261022 from the training episodes (threshold set).
For each selected row the goal frame and the real final frame are encoded with DINOv2 and decoded by the checkpoint's
VQ-VAE decoder (uint8 224 px), the images shown to the VLM.
Writes feat/E3/m1/rules_dset.npz and results/E3/m1/rules_dset_meta.json. usage: python e3_rules_dset.py"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
import declib as D
from e1_judgeset import make_seqs, N_EP, K, CHUNK

E1_RES = os.path.expanduser('~/pmx/B/results/E1/m1')
N_TRAIN_ROWS = 2000


def main():
    t0 = time.time()
    split = json.load(open(f'{E1_RES}/split_e1.json'))
    cal = json.load(open(f'{E1_RES}/calib_e1.json'))
    ref = json.load(open(f'{E1_RES}/judgeset_e1_meta.json'))
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    dt = T_['dset_train']
    # 1. episodes and sequences exactly as e1_judgeset.py (rng consumed in the same order)
    all_eps, all_seqs, all_fams, all_H = [], [], [], []
    for H in E.HS:
        M = cal['mags'][str(H)]
        eps = E.sample_episodes_from(dt, split['J'], N_EP, H, seed=8000 + H)
        rng = np.random.RandomState(8100 + H)
        for c0 in range(0, len(eps), CHUNK):
            for ep in eps[c0:c0 + CHUNK]:
                seqs, fams = make_seqs(ep, M, rng)
                all_eps.append(ep); all_seqs.append(seqs); all_fams.append(fams); all_H.append(H)
    n_ep = len(all_eps)
    is_val = np.random.RandomState(0).uniform(size=n_ep) < 0.2
    nS = K + 1
    train_rows = [(e, k) for e in np.where(~is_val)[0] for k in range(nS)]
    pick = np.random.RandomState(20261022).choice(len(train_rows), N_TRAIN_ROWS, replace=False)
    sel = {(int(e), k): 'held' for e in np.where(is_val)[0] for k in range(nS)}
    for i in pick:
        sel[(int(train_rows[i][0]), int(train_rows[i][1]))] = 'train'
    ep_sel = {e for e, _ in sel}
    # 2. simulate every sequence (margins of the whole set for the identity check); render only selected episodes
    pool = R.pool()
    jobs = [(all_eps[e], all_seqs[e], e in ep_sel) for e in range(n_ep)]
    outs = pool.map(R.episode_job, jobs, chunksize=4)
    pool.close()
    r_all = np.array([o['margins'][k]['r'] for o in outs for k in range(nS)])
    ro_all = np.array([o['margins'][k]['r_obj'] for o in outs for k in range(nS)])
    bins = {b: float(np.mean([R.r_bin(x) == b for x in r_all])) for b in ('success', 'near', 'mid', 'far')}
    bins_obj = {b: float(np.mean([R.r_bin(x) == b for x in ro_all])) for b in ('success', 'near', 'mid', 'far')}
    ident = dict(bins_equal=bins == ref['bins'], bins_obj_equal=bins_obj == ref['bins_obj'],
                 n_val_episodes_equal=int(is_val.sum()) == ref['n_val_episodes'], bins=bins, bins_ref=ref['bins'])
    print('identity vs E1 judge set:', json.dumps(ident), flush=True)
    assert ident['bins_equal'] and ident['bins_obj_equal'] and ident['n_val_episodes_equal'], 'regenerated judge set differs from E1'
    # 3. selected rows: goal + real final frames -> DINOv2 -> VQ-VAE decoder
    rows = sorted(sel)
    gimg = np.stack([outs[e]['goal_img'] for e, _ in rows]); gpro = np.stack([outs[e]['goal_proprio'] for e, _ in rows])
    fimg = np.stack([outs[e]['final_imgs'][k] for e, k in rows]); fpro = np.stack([outs[e]['final_proprios'][k] for e, k in rows])
    zg, _ = R.encode_frames(m, pre, gimg, gpro)
    zf, _ = R.encode_frames(m, pre, fimg, fpro)
    goal_dec, final_dec = D.decode(zg, bs=32), D.decode(zf, bs=32)
    mp = [outs[e]['margins'][k] for e, k in rows]
    os.makedirs(BP.FEAT, exist_ok=True)
    np.savez_compressed(f'{BP.FEAT}/rules_dset.npz', goal=goal_dec, final=final_dec,
                        r=np.array([x['r'] for x in mp]), r_obj=np.array([x['r_obj'] for x in mp]),
                        H=np.array([all_H[e] for e, _ in rows]), fam=np.array([all_fams[e][k] for e, k in rows]),
                        ep=np.array([e for e, _ in rows]), seq=np.array([k for _, k in rows]),
                        split=np.array([sel[x] for x in rows]))
    sp = np.array([sel[x] for x in rows])
    fam = np.array([all_fams[e][k] for e, k in rows])
    summ = dict(identity=ident, n_rows=len(rows), n_held=int((sp == 'held').sum()), n_train=int((sp == 'train').sum()),
                n_held_expert=int(((sp == 'held') & (fam == 'expert')).sum()),
                held_success_frac_r=float(np.mean([x['r'] < 1 for x, s in zip(mp, sp) if s == 'held'])),
                held_success_frac_robj=float(np.mean([x['r_obj'] < 1 for x, s in zip(mp, sp) if s == 'held'])),
                time_s=time.time() - t0)
    R.jdump(summ, f'{BP.RES}/rules_dset_meta.json')
    print('E3_RULES_DSET_DONE', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
