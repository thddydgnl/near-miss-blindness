"""B E0: mode-O imagination of Diffusion-Policy episodes (rolled out in the DINO-WM PushT env by dp_rollout.py).

For each episode and horizon H: window = the last 5H control steps [t0, T_end), t0 = T_end - 5H (skipped if t0 < 0).
Real: fast replay of the recorded relative actions from the episode's init state (exact, dp_conventions C5) -> start
frame at t0 (1 frame + proprio, DINO-WM planning convention as in R2-12) and real final frame at T_end.
Imagined: open-loop DINO-WM rollout from the start frame with the recorded relative actions of the window.
Stored per window: real r components (object pose vs the fixed goal pose), DINOv2 features of real-final / imagined-final
/ start frames (fp16), proprio embeddings, imagination error d_vis(imag, real), and two goal frames for the standard task
(block at the goal pose (256,256,pi/4); agent (i) at the real final agent position [diagnostic], (ii) at a fixed
canonical position (256, 400) = the env's reset agent position). No judge is applied here: judges for the standard-goal
protocol are an E1 decision (see the E0 report).
usage: python dp_imagine.py --rollouts f1.pt[,f2.pt] --H 5,10 --out out.pt"""
import argparse, os, sys, time, json
import numpy as np
import torch
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R

FS = R.FS
CANON_AGENT = (256.0, 400.0)


def goal_state(agent_xy):
    return np.array([agent_xy[0], agent_xy[1], 256.0, 256.0, np.pi / 4, 0.0, 0.0])


def window_job(args):
    """CPU worker: replay one episode, render start frame at t0, final frame at T_end, two goal frames."""
    ep, Hs = args
    torch.set_num_threads(1)
    env = R.get_env()
    out = []
    init, a_rel, T = ep['init'], ep['a_rel'], ep['T_end']
    for H in Hs:
        t0 = T - FS * H
        if t0 < 0:
            continue
        R.sim_reset(env, 'T', init)
        from dp_rollout import step_rel, state7
        for a in a_rel[:t0]:
            step_rel(env, a)
        s0 = state7(env)
        img0 = env._render_frame('rgb_array')
        pro0 = np.concatenate([s0[:2], s0[5:]]).astype(np.float32)
        for a in a_rel[t0:T]:
            step_rel(env, a)
        sf = state7(env)
        imgf = env._render_frame('rgb_array')
        prof = np.concatenate([sf[:2], sf[5:]]).astype(np.float32)
        gimgs, gpros = [], []
        for ag in (sf[:2], CANON_AGENT):
            gs = goal_state(ag)
            R.sim_reset(env, 'T', gs)           # note: _set_state's single physics step does not move a resting T
            gimgs.append(env._render_frame('rgb_array'))
            gpros.append(np.array([ag[0], ag[1], 0.0, 0.0], np.float32))
        out.append(dict(H=H, t0=t0, s0=s0, sf=sf, replay_err=float(np.abs(sf[:5] - ep['states'][T][:5]).max()),
                        img0=img0, pro0=pro0, imgf=imgf, prof=prof, gimgs=gimgs, gpros=gpros,
                        acts=np.asarray(a_rel[t0:T], np.float64)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', required=True)
    ap.add_argument('--H', default='5,10')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    Hs = [int(x) for x in args.H.split(',')]
    t0 = time.time()
    eps, tags = [], []
    for f in args.rollouts.split(','):
        D = torch.load(f, weights_only=False)
        for e in D['episodes']:
            eps.append(e)
            tags.append(os.path.basename(f))
    T = R.load_model()
    m, pre = T['model'], T['pre']
    t_load = time.time() - t0
    pool = R.pool()
    t1 = time.time()
    wins = pool.map(window_job, [(e, Hs) for e in eps], chunksize=2)
    pool.close()
    t_sim = time.time() - t1
    recs, start_img, start_pro, fin_img, fin_pro, acts, gA_img, gA_pro, gC_img, gC_pro = [], [], [], [], [], [], [], [], [], []
    for i, (e, ws) in enumerate(zip(eps, wins)):
        for w in ws:
            fp = __import__('dp_rollout').final_parts(w['sf'])
            recs.append(dict(src=tags[i], seed=e['seed'], ep=i, H=w['H'], t0=w['t0'], T_end=e['T_end'],
                             success_dp=e['success_dp'], final_cov=e['final_cov'], replay_err=w['replay_err'], **fp))
            start_img.append(w['img0']); start_pro.append(w['pro0']); fin_img.append(w['imgf']); fin_pro.append(w['prof'])
            gA_img.append(w['gimgs'][0]); gA_pro.append(w['gpros'][0]); gC_img.append(w['gimgs'][1]); gC_pro.append(w['gpros'][1])
            acts.append(w['acts'])
    t2 = time.time()
    zr, pr = R.encode_frames(m, pre, np.stack(fin_img), np.stack(fin_pro))
    zs, ps = R.encode_frames(m, pre, np.stack(start_img), np.stack(start_pro))
    zgA, pgA = R.encode_frames(m, pre, np.stack(gA_img), np.stack(gA_pro))
    zgC, pgC = R.encode_frames(m, pre, np.stack(gC_img[:1]), np.stack(gC_pro[:1]))       # canonical goal: one frame
    torch.cuda.synchronize()
    t_enc = time.time() - t2
    zi = torch.zeros_like(zr)
    pi = torch.zeros_like(pr)
    t3 = time.time()
    for H in Hs:
        idx = [k for k, r in enumerate(recs) if r['H'] == H]
        if not idx:
            continue
        a = np.stack([acts[k] for k in idx])
        v, p = R.imagine(m, pre, np.stack([start_img[k] for k in idx]), np.stack([start_pro[k] for k in idx]), a)
        zi[idx], pi[idx] = v, p
    torch.cuda.synchronize()
    t_imag = time.time() - t3
    dv_ir = R.d_vis(zi, zr).numpy()
    dv_rA, dv_iA = R.d_vis(zr, zgA).numpy(), R.d_vis(zi, zgA).numpy()
    zC = zgC.expand(len(zr), -1, -1)
    dv_rC, dv_iC = R.d_vis(zr, zC).numpy(), R.d_vis(zi, zC).numpy()
    for k, r in enumerate(recs):
        r.update(dv_imag_vs_real=float(dv_ir[k]), dv_real_goalA=float(dv_rA[k]), dv_imag_goalA=float(dv_iA[k]),
                 dv_real_goalC=float(dv_rC[k]), dv_imag_goalC=float(dv_iC[k]))
    timing = dict(n_windows=len(recs), load_s=t_load, sim_render_s=t_sim, encode_s=t_enc, imagine_s=t_imag,
                  total_s=time.time() - t0)
    torch.save(dict(recs=recs, zr=zr, zi=zi, zs=zs, zgA=zgA, zgC=zgC, pr=pr, pi=pi, ps=ps, pgA=pgA, pgC=pgC,
                    timing=timing, Hs=Hs), args.out)
    print('TIMING', json.dumps(timing))
    for H in Hs:
        rr = [r for r in recs if r['H'] == H]
        if rr:
            print(f"H{H}: n {len(rr)} replay_err max {max(r['replay_err'] for r in rr):.2e} | r_obj<1 {np.mean([r['r_obj'] < 1 for r in rr]):.2f}"
                  f" | median d_vis(imag,real) {np.median([r['dv_imag_vs_real'] for r in rr]):.3f}")
    print('DP_IMAGINE_DONE')


if __name__ == '__main__':
    main()
