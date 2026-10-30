"""B E3 M1: regenerate one E1 curve cell exactly (same code path and seeds as e1_m1_eval.py) and decode its frames for
judge (d). E1 stored judge outputs only, so the cell is re-simulated and re-imagined; the a_full / a_obj ensemble
outputs are compared with the E1 JSONL record by record (identity check). For every record: goal frame (one per
episode), real final frame and imagined final frame -> DINOv2 features -> the checkpoint's VQ-VAE decoder (uint8 224 px).
Writes feat/E3/m1/e1dec_<set>_H<H>_<def>.npz (goal, goal_idx, real, imag, line = E1 JSONL line index) and
results/E3/m1/e1dec_<set>_H<H>_<def>_meta.json. usage: python e3_regen_e1.py <set> <H> <def>"""
import os, sys, time, json, zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
import declib as D
from e1_m1_eval import PER_EP, BATCH_EP, MAX_TRIES

E1_RES = os.path.expanduser('~/pmx/B/results/E1/m1')
E1_FEAT = os.path.expanduser('~/pmx/B/feat/E1/m1')


def main():
    st, H, key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tag = f'{st}_H{H}_{key}'
    out_npz = f'{BP.FEAT}/e1dec_{tag}.npz'
    if os.path.exists(out_npz):
        print('done already', out_npz)
        return
    t0 = time.time()
    ref = [json.loads(l) for l in open(f'{E1_RES}/eval_{tag}.jsonl')]
    cell_seed = zlib.crc32(f'{st}_{H}_{key}'.encode()) % 100000
    split = json.load(open(f'{E1_RES}/split_e1.json'))
    cal = json.load(open(f'{E1_RES}/calib_e1.json'))
    s_max = cal['s_max'][str(H)][key]
    J = torch.load(f'{E1_FEAT}/judges_e1.pt', weights_only=False)
    nets = {name: E.load_ensemble(J['sds'][name]) for name in ('a_full', 'a_obj')}
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    if st == 'val':
        dset, trajs = T_['dset'], list(range(len(T_['dset'])))
    else:
        dset, trajs = T_['dset_train'], split['T']
    rng = np.random.RandomState(cell_seed)
    rstars, fams = E.stratified_targets(rng)
    queue = [dict(tid=i, rstar=float(rstars[i]), fam=str(fams[i]), tries=0) for i in range(len(rstars))]
    dropped, n_ep, batch, line = [], 0, 0, 0
    pool = R.pool()
    G_dec, R_dec, I_dec, gidx, lines = [], [], [], [], []
    diffs = dict(p_full_real=0.0, p_full_imag=0.0, p_obj_real=0.0, p_obj_imag=0.0, r=0.0)
    while queue:
        eps = E.sample_episodes_from(dset, trajs, BATCH_EP, H, seed=cell_seed * 1000 + batch)
        jobs, assigned = [], []
        for i, ep in enumerate(eps):
            if not queue:
                break
            tg, queue = queue[:PER_EP], queue[PER_EP:]
            for t in tg:
                t['tries'] += 1
            assigned.append(tg)
            jobs.append((ep, [(t['tid'], t['rstar'], t['fam']) for t in tg], key, s_max,
                         (cell_seed * 100003 + batch * 1000 + i) % 4294967295, True))
        outs = pool.map(E.grid_episode_job, jobs, chunksize=2)
        for tg, o in zip(assigned, outs):
            un = {u['tid'] for u in o['unreachable']}
            for t in tg:
                if t['tid'] in un:
                    (dropped if t['tries'] >= MAX_TRIES else queue).append(t)
        simg, spro, acts, fimg, fpro, gi, rr = [], [], [], [], [], [], []
        gimg = np.stack([o['goal_img'] for o in outs]); gpro = np.stack([o['goal_proprio'] for o in outs])
        for j, (ep, o) in enumerate(zip(eps[:len(outs)], outs)):
            items = [o['expert']] + o['done']
            for d in items:
                simg.append(o['start_img']); spro.append(o['start_proprio']); acts.append(d['acts'])
                fimg.append(d['final_img']); fpro.append(d['final_proprio']); gi.append(j); rr.append(d['margins']['r'])
        n_ep_before = n_ep
        n_ep += len(outs)
        if not simg:
            batch += 1
            continue
        zg, _ = R.encode_frames(m, pre, gimg, gpro)
        zg_r = zg[torch.as_tensor(gi)]
        zr, _ = R.encode_frames(m, pre, np.stack(fimg), np.stack(fpro))
        zi, _ = R.imagine(m, pre, np.stack(simg), np.stack(spro), np.stack(acts))
        P = {}
        for name in ('a_full', 'a_obj'):
            _, P[f'p{name[1:]}_real'] = E.ens_prob(nets[name], zr, zg_r)
            _, P[f'p{name[1:]}_imag'] = E.ens_prob(nets[name], zi, zg_r)
        for k in range(len(simg)):
            rec = ref[line + k]
            for kk, vv in P.items():
                diffs[kk] = max(diffs[kk], abs(float(vv[k]) - rec[kk]))
            diffs['r'] = max(diffs['r'], abs(rr[k] - rec['r']))
        G_dec.append(D.decode(zg, bs=32))
        R_dec.append(D.decode(zr, bs=32))
        I_dec.append(D.decode(zi, bs=32))
        gidx += [n_ep_before + j for j in gi]
        lines += list(range(line, line + len(simg)))
        line += len(simg)
        print(f'{tag}: batch {batch} episodes {n_ep} records {line}/{len(ref)} max diffs {json.dumps({k: round(v, 7) for k, v in diffs.items()})} '
              f'({time.time() - t0:.0f}s)', flush=True)
        batch += 1
    pool.close()
    assert line == len(ref), (line, len(ref))
    os.makedirs(BP.FEAT, exist_ok=True)
    np.savez_compressed(out_npz, goal=np.concatenate(G_dec), goal_idx=np.array(gidx), real=np.concatenate(R_dec),
                        imag=np.concatenate(I_dec), line=np.array(lines))
    summ = dict(tag=tag, n_records=line, n_episodes=n_ep, n_dropped=len(dropped), max_abs_diff_vs_e1=diffs, time_s=time.time() - t0)
    R.jdump(summ, f'{BP.RES}/e1dec_{tag}_meta.json')
    print('E3_REGEN_DONE', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
