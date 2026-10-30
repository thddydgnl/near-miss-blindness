"""Smoke test: dataset sizes, fast sim + render vs env.rollout, expert replay = goal, encode/imagine timing,
imagined-vs-real distances for a few sequences."""
import sys, time, json
import numpy as np
import torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import r12lib as R
import r21lib as L


def main():
    t0 = time.time()
    T = R.load_model()
    m, pre = T['model'], T['pre']
    dv, dt = T['dset'], T['dset_train']
    lv = np.array([dv.get_seq_length(i) for i in range(len(dv))])
    lt = np.array([dt.get_seq_length(i) for i in range(len(dt))])
    print(f'load {time.time()-t0:.1f}s; val trajs {len(dv)} len min/med/max {lv.min()}/{np.median(lv)}/{lv.max()}; '
          f'train trajs {len(dt)} len med {np.median(lt)}; shapes val {set(dv.shapes)}', flush=True)
    print('action mean/std', pre.action_mean.tolist(), pre.action_std.tolist(), 'num_hist', m.num_hist)
    env = R.get_env()
    eps = R.sample_episodes(dv, 4, 5, seed=7)
    rng = np.random.RandomState(0)
    pol = [('expert', 'expert', {}), ('iid5', 'iid', {'sigma': 5}), ('off5', 'offset', {'beta_max': 5}),
           ('scale3', 'scale', {'k_max': 0.3}), ('gross', 'gross', {'q': 1.0, 'sig_big': 30, 'sig_small': 1})]
    # 1) fast sim vs env.rollout (states and rendered frames), start obs vs env.prepare
    maxd_s, maxd_img, maxd_start = 0.0, 0, 0
    for ep in eps:
        seqs = [(n, R.make_actions(ep['expert'], f, p, rng)[0]) for n, f, p in pol]
        out = R.episode_job((ep, seqs, True))
        env.update_env({'shape': ep['shape']})
        for k, (n, a) in enumerate(seqs):
            obses, sts = env.rollout(123, ep['init_state'], a)
            maxd_s = max(maxd_s, float(np.abs(sts[-1] - out['final_states'][k]).max()))
            maxd_img = max(maxd_img, int(np.abs(obses['visual'][-1].astype(int) - out['final_imgs'][k].astype(int)).max()))
            if k == 0:
                maxd_start = max(maxd_start, int(np.abs(obses['visual'][0].astype(int) - out['start_img'].astype(int)).max()))
        print('episode', ep['traj'], ep['offset'], 'margins', [round(mp['r'], 3) for mp in out['margins']],
              'agent/block/ang of off5', round(out['margins'][2]['agent'], 1), round(out['margins'][2]['block'], 1),
              round(out['margins'][2]['ang'], 3), flush=True)
    print(f'fast sim vs env.rollout: max|state diff| {maxd_s:.2e}, max|final img diff| {maxd_img}, start img diff {maxd_start}')
    # 2) timing of sim+render
    t1 = time.time()
    n = 0
    for ep in eps:
        for _ in range(10):
            R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=False)
            n += 1
    t2 = time.time()
    for ep in eps:
        for _ in range(10):
            R.sim_run(env, ep['shape'], ep['init_state'], ep['expert'], render=True)
    t3 = time.time()
    print(f'sim only {(t2-t1)/n*1000:.1f} ms/traj, sim+render {(t3-t2)/n*1000:.1f} ms/traj')
    # 3) encode + imagine (all 4 episodes x 5 policies)
    jobs = []
    for ep in eps:
        seqs = [(nm, R.make_actions(ep['expert'], f, p, rng)[0]) for nm, f, p in pol]
        jobs.append((ep, seqs, True))
    outs = [R.episode_job(j) for j in jobs]
    simg = np.stack([o['start_img'] for o in outs for _ in pol])
    spro = np.stack([o['start_proprio'] for o in outs for _ in pol])
    acts = np.stack([s[1] for j in jobs for s in j[1]])
    fimg = np.concatenate([o['final_imgs'] for o in outs])
    fpro = np.concatenate([o['final_proprios'] for o in outs])
    gimg = np.stack([o['goal_img'] for o in outs for _ in pol])
    gpro = np.stack([o['goal_proprio'] for o in outs for _ in pol])
    torch.cuda.synchronize(); t4 = time.time()
    zr, pr = R.encode_frames(m, pre, fimg, fpro)
    zg, pg = R.encode_frames(m, pre, gimg, gpro)
    torch.cuda.synchronize(); t5 = time.time()
    zi, pi = R.imagine(m, pre, simg, spro, acts)
    torch.cuda.synchronize(); t6 = time.time()
    print(f'encode {len(fimg)*2} frames {t5-t4:.2f}s ; imagine {len(acts)} seqs {t6-t5:.2f}s')
    # bigger timing batch
    big = 512
    simg_b = np.repeat(simg[:1], big, 0); spro_b = np.repeat(spro[:1], big, 0); acts_b = np.repeat(acts[:1], big, 0)
    torch.cuda.synchronize(); t7 = time.time()
    R.imagine(m, pre, simg_b, spro_b, acts_b, bs=256)
    torch.cuda.synchronize(); t8 = time.time()
    print(f'imagine {big} seqs (bs 256) {t8-t7:.2f}s ; peak mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
    zf, zs = R.encode_frames(m, pre, simg[::len(pol)], spro[::len(pol)])
    dvr = R.d_vis(zr, zg).numpy(); dvi = R.d_vis(zi, zg).numpy()
    dfr = R.d_full(zr, zg, pr, pg).numpy(); dfi = R.d_full(zi, zg, pi, pg).numpy()
    dri = R.d_vis(zi, zr).numpy()
    rr = [mp['r'] for o in outs for mp in o['margins']]
    for k in range(len(rr)):
        print(f'{pol[k % len(pol)][0]:7s} r {rr[k]:6.2f} | d_vis real {dvr[k]:.4f} imag {dvi[k]:.4f} | d_full real {dfr[k]:.4f} '
              f'imag {dfi[k]:.4f} | d(imag,real) {dri[k]:.4f}')
    dstart = R.d_vis(zf.repeat_interleave(len(pol), 0), zg).numpy()
    print('d_vis(start, goal) per episode', np.round(dstart[::len(pol)], 4))
    print('SMOKE_OK', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
