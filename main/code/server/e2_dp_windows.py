"""B E2 M1 step 2: mode-O imagination of the evaluation episodes of the 24 DP checkpoints and the a_std judgments
(DESIGN_main v1 section 0.2, E0 report section 8.8, v2.2 section 7).
Per episode and H in {3, 5, 7, 10}: window = the last 5H control steps [T_end - 5H, T_end). If T_end < 5H the window is
left-padded with zero relative actions from the initial state (exact standstill in the simulator, v2.2).
Real: exact replay of the recorded relative actions from the initial state -> start frame (1 frame + proprio, DINO-WM
planning convention) at the window start and the real final frame. Imagined: open-loop DINO-WM rollout from the start
frame with the window's actions. Judge: a_std ensemble (K = 5, fixed in E1) against the canonical goal frame, on the
real and the imagined final frame of the same window.
For judge (d) (E3) the real and imagined final frames and the canonical goal frame are decoded by the checkpoint's
VQ-VAE decoder and stored as uint8.
Truth = the recorded final state of the executed episode (r_obj etc. from dp_rollout). The replay used to render the
window frames runs in another process; pymunk's contact ordering can differ there, so the replayed final state can
deviate slightly (first run: up to 2.4e-3 px in one episode). The deviation is recorded per window
(replay_err, replay_r_obj_diff) and summarised; the truth labels never use the replay.
Writes results/E2/m1/dp_windows.jsonl and feat/E2/m1/dp_dec_H<H>.npz. usage: python e2_dp_windows.py"""
import os, sys, time, json, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
import declib as D
from dp_rollout import step_rel, state7, final_parts

HS = (3, 5, 7, 10)
FS = R.FS
E1_FEAT = os.path.expanduser('~/pmx/B/feat/E1/m1')


def window_job(args):
    """CPU worker: one episode -> per H the start frame, real final frame and the window's actions."""
    ep, Hs = args
    torch.set_num_threads(1)
    env = R.get_env()
    init, a_rel, T = ep['init'], np.asarray(ep['a_rel'], np.float64), int(ep['T_end'])
    out = []
    for H in Hs:
        W = FS * H
        pad = max(0, W - T)
        acts = np.concatenate([np.zeros((pad, 2)), a_rel]) if pad else a_rel
        s = len(acts) - W
        R.sim_reset(env, 'T', init)
        for a in acts[:s]:
            step_rel(env, a)
        s0 = state7(env)
        img0 = env._render_frame('rgb_array')
        pro0 = np.concatenate([s0[:2], s0[5:]]).astype(np.float32)
        for a in acts[s:]:
            step_rel(env, a)
        sf = state7(env)
        imgf = env._render_frame('rgb_array')
        prof = np.concatenate([sf[:2], sf[5:]]).astype(np.float32)
        out.append(dict(H=H, t0=int(T - W), pad=int(pad), sf=sf, replay_err=float(np.abs(sf[:5] - ep['states'][T][:5]).max()),
                        img0=img0, pro0=pro0, imgf=imgf, prof=prof, acts=np.asarray(acts[s:], np.float64)))
    return out


def main():
    t0 = time.time()
    files = sorted(glob.glob(f'{BP.RES}/dp_rollouts/s*_ep*.pt'))
    assert len(files) == 24, f'expected 24 rollout files, found {len(files)}'
    eps, pols = [], []
    for f in files:
        Dd = torch.load(f, weights_only=False)
        name = os.path.basename(f)[:-3]
        assert Dd['seeds'] == list(range(100000, 100100))
        for k, e in enumerate(Dd['episodes']):
            eps.append(e)
            pols.append((name, k))
    A = torch.load(f'{E1_FEAT}/astd_e1.pt', weights_only=False)
    nets = E.load_ensemble(A['sds'])
    zgC = A['zgC']
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    pool = R.pool()
    wins = pool.map(window_job, [(e, HS) for e in eps], chunksize=4)
    pool.close()
    t_sim = time.time() - t0
    print(f'windows simulated: {sum(len(w) for w in wins)} ({t_sim:.0f}s)', flush=True)
    gdec = D.decode(zgC)[0]
    out_path = f'{BP.RES}/dp_windows.jsonl'
    n_pad = {H: 0 for H in HS}
    max_err, max_dr, n_dev, n_band_flip = 0.0, 0.0, 0, 0
    with open(out_path + '.tmp', 'w') as fh:
        for H in HS:
            idx = [(i, w) for i, ws in enumerate(wins) for w in ws if w['H'] == H]
            zr, _ = R.encode_frames(m, pre, np.stack([w['imgf'] for _, w in idx]), np.stack([w['prof'] for _, w in idx]))
            zi, _ = R.imagine(m, pre, np.stack([w['img0'] for _, w in idx]), np.stack([w['pro0'] for _, w in idx]),
                              np.stack([w['acts'] for _, w in idx]))
            zg = zgC.expand(len(zr), -1, -1)
            Pr, er = E.ens_prob(nets, zr, zg)
            Pi, ei = E.ens_prob(nets, zi, zg)
            dv_ir, dv_rg, dv_ig = R.d_vis(zi, zr).numpy(), R.d_vis(zr, zg).numpy(), R.d_vis(zi, zg).numpy()
            dec_r, dec_i = D.decode(zr, bs=32), D.decode(zi, bs=32)
            uid = np.array([f'{pols[i][0]}/{pols[i][1]}' for i, _ in idx])
            np.savez_compressed(f'{BP.FEAT}/dp_dec_H{H}.npz', real=dec_r, imag=dec_i, goal=gdec, uid=uid)
            for k, (i, w) in enumerate(idx):
                name, epi = pols[i]
                e = eps[i]
                fp = final_parts(e['states'][int(e['T_end'])])
                assert abs(fp['r_obj'] - e['r_obj']) < 1e-12, (name, epi)
                dr = abs(final_parts(w['sf'])['r_obj'] - e['r_obj'])
                n_pad[H] += int(w['pad'] > 0)
                max_err = max(max_err, w['replay_err'])
                max_dr = max(max_dr, dr)
                n_dev += int(w['replay_err'] > 1e-6)
                n_band_flip += int(R.r_bin(final_parts(w['sf'])['r_obj']) != R.r_bin(e['r_obj']))
                rec = dict(policy=name, ep=epi, seed=int(e['seed']), H=H, t0=w['t0'], pad=w['pad'], T_end=int(e['T_end']),
                           success_dp=bool(e['success_dp']), final_cov=float(e['final_cov']), replay_err=w['replay_err'],
                           replay_r_obj_diff=dr,
                           bin_obj=R.r_bin(fp['r_obj']), success_obj=bool(fp['r_obj'] < 1), **fp,
                           p_std_real=round(float(er[k]), 6), p_std_imag=round(float(ei[k]), 6),
                           p_std_real_m=[round(float(x), 5) for x in Pr[:, k]], p_std_imag_m=[round(float(x), 5) for x in Pi[:, k]],
                           dv_imag_vs_real=round(float(dv_ir[k]), 6), dv_real_goal=round(float(dv_rg[k]), 6),
                           dv_imag_goal=round(float(dv_ig[k]), 6), dec_row=k)
                fh.write(json.dumps(rec) + '\n')
            ro = np.array([eps[i]['r_obj'] for i, _ in idx])
            bins = np.array([R.r_bin(x) for x in ro])
            print(f'H{H}: n {len(idx)} padded {n_pad[H]} | P(judged success | near) real {np.mean(er[bins == "near"] >= 0.5):.3f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    os.replace(out_path + '.tmp', out_path)
    summ = dict(n_episodes=len(eps), n_windows=sum(len(w) for w in wins), padded=n_pad, max_replay_err_px=max_err,
                max_replay_r_obj_diff=max_dr, n_windows_replay_err_gt_1e6=n_dev, n_windows_replay_band_flip=n_band_flip,
                sim_s=t_sim, time_s=time.time() - t0)
    R.jdump(summ, f'{BP.RES}/dp_windows_meta.json')
    print('E2_DP_WINDOWS_DONE', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
