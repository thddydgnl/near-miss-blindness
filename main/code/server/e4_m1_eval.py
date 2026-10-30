"""B E4a / E5 (M1) evaluation: re-imagine E1 curve cells with fine-tuned DINO-WM variants (DESIGN_main v3.1 section 9).
  cache <set> <H> <def>          regenerate the E1 cell exactly (same code path / seeds as e1_m1_eval.py, as e3_regen_e1.py)
                                 and store the imagination inputs: start frames + proprio, actions, goal features, line index
  imagine <set> <H> <def> <variants>
                                 for each variant (orig = public checkpoint, or <cond>_s<seed> = ckpt/E4/m1/<...>.pt):
                                 imagine every record and judge with the frozen a_full / a_obj ensembles and d_vis;
                                 the 'orig' pass is checked against the E1 JSONL (identity).
Writes feat/E4/m1/cell_<set>_H<H>_<def>.npz and results/E4/m1/imag_<set>_H<H>_<def>_<variant>.npz.
usage: python e4_m1_eval.py cache val 5 r ; python e4_m1_eval.py imagine val 5 r orig,S97_s0,..."""
import os, sys, time, json, zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
import e4lib as E4
from e1_m1_eval import PER_EP, BATCH_EP, MAX_TRIES

E1_RES = os.path.expanduser('~/pmx/B/results/E1/m1')
E1_FEAT = os.path.expanduser('~/pmx/B/feat/E1/m1')
FT_CK = os.path.expanduser('~/pmx/B/ckpt/E4/m1')
CELL_DIR = os.path.expanduser('~/pmx/B/feat/E4/m1')
OUT_DIR = os.path.expanduser('~/pmx/B/results/E4/m1')


def cache(st, H, key, T_):
    tag = f'{st}_H{H}_{key}'
    path = f'{CELL_DIR}/cell_{tag}.npz'
    if os.path.exists(path):
        print('cached', path)
        return
    t0 = time.time()
    ref = [json.loads(l) for l in open(f'{E1_RES}/eval_{tag}.jsonl')]
    cell_seed = zlib.crc32(f'{st}_{H}_{key}'.encode()) % 100000
    split = json.load(open(f'{E1_RES}/split_e1.json'))
    cal = json.load(open(f'{E1_RES}/calib_e1.json'))
    s_max = cal['s_max'][str(H)][key]
    m, pre = T_['model'], T_['pre']
    dset, trajs = (T_['dset'], list(range(len(T_['dset'])))) if st == 'val' else (T_['dset_train'], split['T'])
    rng = np.random.RandomState(cell_seed)
    rstars, fams = E.stratified_targets(rng)
    queue = [dict(tid=i, rstar=float(rstars[i]), fam=str(fams[i]), tries=0) for i in range(len(rstars))]
    dropped, n_ep, batch, line = [], 0, 0, 0
    pool = R.pool()
    S_img, S_pro, ACT, G, gidx, rr = [], [], [], [], [], []
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
        n0 = n_ep
        if outs:
            zg, _ = R.encode_frames(m, pre, np.stack([o['goal_img'] for o in outs]), np.stack([o['goal_proprio'] for o in outs]))
            G.append(zg)
        for j, o in enumerate(outs):
            for d in [o['expert']] + o['done']:
                S_img.append(o['start_img']); S_pro.append(o['start_proprio']); ACT.append(d['acts'])
                gidx.append(n0 + j); rr.append(d['margins']['r'])
        n_ep += len(outs)
        batch += 1
    pool.close()
    assert len(rr) == len(ref) and max(abs(a - b['r']) for a, b in zip(rr, ref)) == 0.0, 'regenerated cell differs from E1'
    np.savez(path, start_img=np.stack(S_img), start_pro=np.stack(S_pro).astype(np.float32), acts=np.stack(ACT),
             gidx=np.array(gidx), zg=torch.cat(G).numpy())
    print(f'CACHED {tag}: {len(rr)} records, {n_ep} episodes ({time.time() - t0:.0f}s)', flush=True)


def imagine(st, H, key, variants, T_):
    tag = f'{st}_H{H}_{key}'
    C = np.load(f'{CELL_DIR}/cell_{tag}.npz')
    ref = [json.loads(l) for l in open(f'{E1_RES}/eval_{tag}.jsonl')]
    m, pre = T_['model'], T_['pre']
    snap = E4.snapshot(m)
    J = torch.load(f'{E1_FEAT}/judges_e1.pt', weights_only=False)
    nets = {name: E.load_ensemble(J['sds'][name]) for name in ('a_full', 'a_obj')}
    zg = torch.from_numpy(C['zg'])[torch.as_tensor(C['gidx'])]
    os.makedirs(OUT_DIR, exist_ok=True)
    for v in variants:
        out = f'{OUT_DIR}/imag_{tag}_{v}.npz'
        if os.path.exists(out):
            print('exists', out)
            continue
        t0 = time.time()
        E4.restore(m, snap)
        if v != 'orig':
            E4.load_variant(m, f'{FT_CK}/{v}.pt')
        m.eval()
        zi, _ = R.imagine(m, pre, C['start_img'], C['start_pro'], C['acts'])
        res = {}
        for name in ('a_full', 'a_obj'):
            P, e = E.ens_prob(nets[name], zi, zg)
            res[f'p{name[1:]}_imag'], res[f'p{name[1:]}_imag_m'] = e.astype(np.float32), P.astype(np.float32)
        res['dv_imag'] = R.d_vis(zi, zg).numpy().astype(np.float32)
        ident = None
        if v == 'orig':
            ident = dict(max_abs_p_full=float(np.max(np.abs(res['p_full_imag'] - np.array([r['p_full_imag'] for r in ref])))),
                         max_abs_p_obj=float(np.max(np.abs(res['p_obj_imag'] - np.array([r['p_obj_imag'] for r in ref])))))
        np.savez(out, **res)
        print(f'{tag} {v}: imagined {len(zi)} ({time.time() - t0:.0f}s) identity {ident}', flush=True)
    E4.restore(m, snap)


def main():
    mode, st, H, key = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    T_ = R.load_model()
    if mode == 'cache':
        cache(st, H, key, T_)
    else:
        imagine(st, H, key, sys.argv[5].split(','), T_)
    print('E4_EVAL_DONE', mode, st, H, key, flush=True)


if __name__ == '__main__':
    main()
