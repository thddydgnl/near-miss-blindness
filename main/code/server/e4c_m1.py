"""B E4c (M1, descriptive; DESIGN_main v3.1 section 9): where the DINO-WM training loss puts its weight.
Data = the S97 fine-tuning windows (the base training data of E5): per window the real feature innovations
dz_k = z(frame k) - z(frame k-1), k = 1..3 (the loss targets are frames 1..3).
  1. spatial shares of the innovation energy |dz|^2 on the target frame's T-block patches, agent patches (not block)
     and background (patch masks from the rendered frames)
  2. the same shares of the public checkpoint's own prediction error |z_hat - z|^2 (teacher forcing, eval mode)
  3. R2-11-style variance share: top-50 PCA of dz (2,000 transitions), unique R^2 of block-pose change
     (dx, dy, sin dtheta, cos dtheta - 1) and of agent displacement (dx, dy), PC-variance weighted
  4. lambda for the boundary-weighted loss (v): block patches get weight lambda so that they hold 50% of the weighted
     innovation energy (lambda = E_rest / E_block, at least 1)
Writes results/E4/m1/e4c_m1.json. usage: python e4c_m1.py"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e4lib as E4
from e4_m1_finetune import Data

N_PCA_ROWS = 2000


def sim_job(args):
    shape, init, acts = args
    torch.set_num_threads(1)
    env = R.get_env()
    return E4.run_frames(env, shape, init, acts, render=False)[0]


def main():
    t0 = time.time()
    D = Data()
    AM = np.load(os.path.expanduser('~/pmx/B/feat/E4/m1/ft_meta.npz'))['amask']
    sel = np.concatenate([D.sel['S97_s'], D.sel['S97_f']])
    T_ = R.load_model()
    m, pre = T_['model'], T_['pre']
    m.eval()
    E_blk, E_agt, E_bg = 0.0, 0.0, 0.0
    P_blk, P_agt, P_bg = 0.0, 0.0, 0.0
    rows_dz, rows_idx = [], []
    rng = np.random.RandomState(0)
    pca_pick = set(rng.choice(len(sel), N_PCA_ROWS, replace=False).tolist())
    with torch.no_grad():
        for b0 in range(0, len(sel), 64):
            idx = sel[b0:b0 + 64]
            vis, pro, act, bm = D.batch(idx)
            dz = vis[:, 1:] - vis[:, :-1]                                   # (B, 3, 196, 384)
            e = (dz ** 2).sum(-1)                                          # (B, 3, 196)
            amask = torch.from_numpy(AM[idx]).cuda()
            bmask = bm.bool()
            agent_only = amask & ~bmask
            E_blk += float(e[bmask].sum()); E_agt += float(e[agent_only].sum()); E_bg += float(e[~(bmask | amask)].sum())
            zp = m.proprio_encoder(pro)
            za = m.action_encoder(act)
            Pn = vis.shape[2]
            z = torch.cat([vis, zp[:, :, None].expand(-1, -1, Pn, -1), za[:, :, None].expand(-1, -1, Pn, -1)], -1)
            zh = m.predict(z[:, :3])
            pe = ((zh[..., :384] - vis[:, 1:]) ** 2).sum(-1)
            P_blk += float(pe[bmask].sum()); P_agt += float(pe[agent_only].sum()); P_bg += float(pe[~(bmask | amask)].sum())
            for q, i in enumerate(range(b0, b0 + len(idx))):
                if i in pca_pick:
                    rows_dz.append(dz[q, 0].reshape(-1).half().cpu())      # transition 0 -> 1 of this window
                    rows_idx.append(int(idx[q]))
    tot, ptot = E_blk + E_agt + E_bg, P_blk + P_agt + P_bg
    lam = max(1.0, (E_agt + E_bg) / E_blk) if E_blk > 0 else 1.0
    # R2-11-style unique R^2 of block-pose change vs agent displacement on PCA-50 of dz (transition 0 -> 1)
    X = torch.stack(rows_dz).float()
    X = X - X.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    k = 50
    Y = (U[:, :k] * S[:k]).numpy()                                         # PC scores
    var = (S[:k] ** 2).numpy()
    z = np.load(f'{os.path.expanduser("~/pmx/B/feat/E4/m1")}/ft_meta.npz')
    win_of, acts_n = z['win_of'], z['act']
    dt = T_['dset_train']
    jobs = []
    for i in rows_idx:
        w = win_of[i]
        traj, off = int(z['win_traj'][w]), int(z['win_off'][w])
        a = pre.denormalize_actions(torch.from_numpy(acts_n[i].reshape(20, 2))).numpy().astype(np.float64)
        jobs.append((dt.shapes[traj], dt.states[traj, off].numpy().astype(np.float64), a))
    pool = R.pool()
    sts = pool.map(sim_job, jobs, chunksize=8)
    pool.close()
    s0 = np.stack([s[0] for s in sts]); s1 = np.stack([s[1] for s in sts])
    dth = (s1[:, 4] - s0[:, 4] + np.pi) % (2 * np.pi) - np.pi
    Xb = np.stack([s1[:, 2] - s0[:, 2], s1[:, 3] - s0[:, 3], np.sin(dth), np.cos(dth) - 1], 1)
    Xa = np.stack([s1[:, 0] - s0[:, 0], s1[:, 1] - s0[:, 1]], 1)

    def r2(Xf):
        Xf = np.concatenate([Xf, np.ones((len(Xf), 1))], 1)
        beta, *_ = np.linalg.lstsq(Xf, Y, rcond=None)
        res = Y - Xf @ beta
        return 1 - (res ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    r_all, r_nob, r_noa = r2(np.concatenate([Xb, Xa], 1)), r2(Xa), r2(Xb)
    wv = var / var.sum()
    out = dict(n_windows=int(len(sel)), innovation_energy_share=dict(block=E_blk / tot, agent=E_agt / tot, background=E_bg / tot),
               prediction_error_share=dict(block=P_blk / ptot, agent=P_agt / ptot, background=P_bg / ptot),
               lambda_block=float(lam), lambda_rule='block patches hold 50% of the weighted innovation energy (lambda = E_rest / E_block, >= 1)',
               pca=dict(n_rows=len(rows_idx), k=k, explained=float(var.sum() / (S ** 2).sum()),
                        r2_all=float((wv * r_all).sum()), unique_r2_block=float((wv * (r_all - r_nob)).sum()),
                        unique_r2_agent=float((wv * (r_all - r_noa)).sum())),
               time_s=time.time() - t0)
    R.jdump(out, f'{BP.RES}/e4c_m1.json')
    print('E4C_M1_DONE', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
