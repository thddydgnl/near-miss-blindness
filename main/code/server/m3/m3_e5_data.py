"""B E5 M3: frames and tables for the exposure-matched prescriptions (DESIGN_main v3.1 section 9), per data seed.
Base data: p = 0.03 ('std' table) and p = 0.1 ('p01' table) of the E1 frame bank. The failure scenes of the base table
keep their scene and count; only the alignment error of the failure episode changes:
  near  r = 1 + 0.2 v2        (same rule as R2-9's near-miss failures, r in [1, 1.2))
  pair  r = 0.8 + 0.2 v3      (same rule as R2-9's paired boundary successes, r in [0.8, 1)), same scene as its failure
  full  r = 1 + 2 U, U ~ U[0, 1) from generator seed 3000 + data seed (CureWM-style full range, r in [1, 3])
The scene parameters are re-drawn exactly as world.build_seed (generator seed 2000 + data seed, first draw) and checked
against the bank's tables. Frames are rendered and encoded with the bank's PCA (eval.pt).
Tables written (same fields as the bank tables): p{p}_base, p{p}_near, p{p}_full, p{p}_cp (failures -> near-miss,
group 2, plus appended pair rows, group 3, partner links), for p in {0.03, 0.1}. Also object / gripper token masks of the
4 frames after each action (8 x 8 token grid), for E4c and the boundary-weighted loss (v).
Writes feat/E5/m3/ext_s<seed>.pt. usage: python m3_e5_data.py [seeds]"""
import os as _os
# B guard: GPU 1 only, torch hub inside ~/pmx (see ../bpaths.py)
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, time, json
import torch
from world import (Encoder, sample_scene, sub, actions, frame_params, FEAT_DIR, N_TRAIN, P_LOW, P_MID, NEAR, XR, ZR, IMG, OPEN_IN, FW)

OUT = os.path.expanduser(os.environ.get('B_E5_FEAT', '~/pmx/B/feat/E5/m3'))
TOK = 8


def token_masks(S, e):
    """object / gripper masks on the 8 x 8 token grid for frames 1..4 of scenes S under alignment error e: (n, 4, 64)."""
    n = e.shape[0]
    cell = IMG / TOK
    px = lambda x: (x - XR[0]) / (XR[1] - XR[0]) * IMG
    py = lambda z: (ZR[1] - z) / (ZR[1] - ZR[0]) * IMG
    g = torch.arange(TOK, device=e.device, dtype=torch.float32) * cell

    def rect(x0, x1, z0, z1):
        c0, c1 = px(x0), px(x1)                     # columns
        r0, r1 = py(z1), py(z0)                     # rows (z up)
        col = (g[None] + cell > c0[:, None]) & (g[None] < c1[:, None])
        row = (g[None] + cell > r0[:, None]) & (g[None] < r1[:, None])
        return (row[:, :, None] & col[:, None, :]).reshape(n, TOK * TOK)
    obj, grip = [], []
    for t in range(1, 5):
        fp = frame_params(S, e, t)
        w = S['w']
        obj.append(rect(fp['ox'] - w, fp['ox'] + w, fp['oz'], fp['oz'] + 0.10))
        cap = fp.get('cap', torch.zeros(n, dtype=torch.bool, device=e.device))
        inner = torch.where(fp['closed'], torch.where(cap, w, 0.01 * torch.ones_like(w)), OPEN_IN * torch.ones_like(w))
        outer = inner + FW
        grip.append(rect(fp['gx'] - outer, fp['gx'] + outer, fp['gz'], fp['gz'] + 0.16))
    return torch.stack(obj, 1), torch.stack(grip, 1)


def main():
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    seeds = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2').split(',')]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    EV = torch.load(os.path.join(FEAT_DIR, 'eval.pt'), weights_only=False)
    enc = Encoder(dev)
    enc.mu, enc.V = EV['pca_mu'].to(dev), EV['pca_V'].to(dev)
    for s in seeds:
        out = f'{OUT}/ext_s{s}.pt'
        if os.path.exists(out):
            print('exists', out)
            continue
        B = torch.load(os.path.join(FEAT_DIR, f'bank_s{s}_N{N_TRAIN}.pt'), weights_only=False)
        n_base_frames = B['F'].shape[0]
        tabs = {k: {kk: vv.to(dev) for kk, vv in v.items()} for k, v in B['tabs'].items()}
        gen = torch.Generator(device=dev)
        gen.manual_seed(2000 + s)
        S = sample_scene(N_TRAIN, gen, dev)
        # identity check against the bank: base r = 1 + v for u < 0.03 failures, v otherwise
        r_chk = torch.where(S['u'] < P_LOW, 1 + S['v'], S['v'])
        assert torch.allclose(r_chk, tabs['std']['r']), 'scene parameters do not reproduce the bank'
        g2 = torch.Generator(device=dev)
        g2.manual_seed(3000 + s)
        r_full_all = 1 + 2 * torch.rand(N_TRAIN, generator=g2, device=dev)
        fail01 = S['u'] < P_MID
        idx = fail01.nonzero().squeeze(1)                    # all failure scenes of p = 0.1 (superset of p = 0.03's)
        Sf = sub(S, idx)
        e = dict(near=Sf['sign'] * (NEAR[0] + (NEAR[1] - NEAR[0]) * Sf['v2']) * Sf['w'],
                 pair=Sf['sign'] * (0.8 + 0.2 * Sf['v3']) * Sf['w'],
                 full=Sf['sign'] * r_full_all[idx] * Sf['w'])
        frames, fidx, cursor = [], {}, n_base_frames
        for k, ek in e.items():
            fr = torch.stack([enc.frames(Sf, ek, t) for t in range(1, 5)], 1)          # [n_f, 4, 64, 64]
            frames.append(fr.reshape(-1, fr.shape[2], fr.shape[3]))
            fidx[k] = (cursor + torch.arange(fr.shape[0] * 4, device=dev)).reshape(-1, 4)
            cursor += fr.shape[0] * 4
        F_ext = torch.cat(frames).cpu()
        pos = torch.full((N_TRAIN,), -1, dtype=torch.long, device=dev)
        pos[idx] = torch.arange(len(idx), device=dev)
        rr = lambda ek, SS: ek.abs() / SS['w']
        out_tabs, masks = {}, {}
        for p, base in (('0.03', 'std'), ('0.1', 'p01')):
            T = {k: v.clone() for k, v in tabs[base].items()}
            fl = (~T['succ']).nonzero().squeeze(1)
            assert torch.all(pos[fl] >= 0)
            out_tabs[f'p{p}_base'] = T
            for k in ('near', 'full'):
                Tk = {kk: vv.clone() for kk, vv in T.items()}
                q = pos[fl]
                Tk['f'][fl] = fidx[k][q]
                Tk['act'][fl] = actions(sub(S, fl), e[k][q])
                Tk['r'][fl] = rr(e[k][q], sub(S, fl))
                Tk['group'][fl] = 2 if k == 'near' else 4
                out_tabs[f'p{p}_{k}'] = Tk
            # cp: failures -> near (group 2) + appended pair rows (group 3) with partner links
            Tn = out_tabs[f'p{p}_near']
            n0, nf = Tn['f0'].shape[0], len(fl)
            q = pos[fl]
            cp = {kk: vv.clone() for kk, vv in Tn.items()}
            cp['f0'] = torch.cat([Tn['f0'], Tn['f0'][fl]])
            cp['f'] = torch.cat([Tn['f'], fidx['pair'][q]])
            cp['act'] = torch.cat([Tn['act'], actions(sub(S, fl), e['pair'][q])])
            cp['succ'] = torch.cat([Tn['succ'], torch.ones(nf, dtype=torch.bool, device=dev)])
            cp['r'] = torch.cat([Tn['r'], rr(e['pair'][q], sub(S, fl))])
            cp['group'] = torch.cat([Tn['group'], torch.full((nf,), 3, device=dev, dtype=Tn['group'].dtype)])
            cp['partner'] = torch.full((n0 + nf,), -1, dtype=torch.long, device=dev)
            cp['partner'][fl] = n0 + torch.arange(nf, device=dev)
            cp['partner'][n0 + torch.arange(nf, device=dev)] = fl
            out_tabs[f'p{p}_cp'] = cp
            # token masks of the base table rows (alignment error from r, sign, w)
            eb = S['sign'] * T['r'] * S['w']
            masks[f'p{p}_base'] = token_masks(S, eb)
        res = dict(F_ext=F_ext, n_base_frames=n_base_frames, tabs={k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in out_tabs.items()},
                   masks={k: (v[0].cpu(), v[1].cpu()) for k, v in masks.items()},
                   info=dict(n_fail_p003=int((S['u'] < P_LOW).sum()), n_fail_p01=int(fail01.sum()), n_ext_frames=int(F_ext.shape[0]),
                             r_full_range=[float(r_full_all[idx].min()), float(r_full_all[idx].max())]))
        torch.save(res, out + '.tmp')
        os.replace(out + '.tmp', out)
        print(f'seed {s}: {res["info"]} ({time.time() - t0:.0f}s)', flush=True)
    print('M3_E5_DATA_DONE', flush=True)


if __name__ == '__main__':
    main()
