"""B E7 convention check (DESIGN_main 12.8; before any judge is trained; descriptive with a stop rule).
200 val expert episodes at H = 5: ratio = d_vis(imagined final, real final) / d_vis(real start, real final). A median >= 1
means imagination is no closer to the real outcome than the start frame, i.e. the world model's action convention does
not match the simulator (the Wall-type mismatch) -> exit code 2 (stop and report). Reference: the same ratio with the
actions scaled x2 (should be worse if the convention is right). Also measures the pixel scale of tau = 0.5 (ball centroid
shift in the 224 px frame; encoder frame = 196 px, patch 14 px). usage: python e7_convcheck.py"""
import os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E

N_EP, H = 200, 5


def centroid(img):
    im = np.asarray(img).astype(float)
    g = (im[..., 1] > im[..., 0] + 30) & (im[..., 1] > im[..., 2] + 30)      # greenish ball site (rgba .3 .6 .3)
    ys, xs = np.nonzero(g)
    return (float(xs.mean()), float(ys.mean()), int(g.sum())) if len(xs) else None


def main():
    t0 = time.time()
    T_ = R.load_model()
    m, pre, dset = T_['model'], T_['pre'], T_['dset']
    eps = E.sample_episodes_from(dset, list(range(len(dset))), N_EP, H, seed=E.SEED_BASE + 9000)
    pool = R.pool()
    outs = pool.map(R.episode_job, [(ep, [('expert', ep['expert'])], True) for ep in eps], chunksize=4)
    pool.close()
    simg = np.stack([o['start_img'] for o in outs]); spro = np.stack([o['start_proprio'] for o in outs])
    fimg = np.stack([o['final_imgs'][0] for o in outs]); fpro = np.stack([o['final_proprios'][0] for o in outs])
    acts = np.stack([ep['expert'] for ep in eps])
    zs, _ = R.encode_frames(m, pre, simg, spro)
    zr, _ = R.encode_frames(m, pre, fimg, fpro)
    zi, _ = R.imagine(m, pre, simg, spro, acts)
    zi2, _ = R.imagine(m, pre, simg, spro, 2.0 * acts)
    d_ir, d_sr, d_is = R.d_vis(zi, zr).numpy(), R.d_vis(zs, zr).numpy(), R.d_vis(zi, zs).numpy()
    d_i2r = R.d_vis(zi2, zr).numpy()
    ratio = d_ir / np.maximum(d_sr, 1e-9)
    ratio2 = d_i2r / np.maximum(d_sr, 1e-9)
    disp = np.array([float(np.linalg.norm(o['goal_state'][:2] - ep['init_state'][:2])) for o, ep in zip(outs, eps)])
    env = R.get_env()

    def frame(x, y):
        env.set_state(np.array([x, y]), np.zeros(2)); env.set_marker()
        return np.asarray(env._render_frame())
    c0, c1 = centroid(frame(1.0, 1.0)), centroid(frame(1.0, 1.5))         # both in the free column of the U maze
    px = (float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) if (c0 and c1) else None)
    out = dict(n=N_EP, H=H, ratio_median=float(np.median(ratio)), ratio_q10=float(np.quantile(ratio, 0.1)),
               ratio_q90=float(np.quantile(ratio, 0.9)), ratio_x2_median=float(np.median(ratio2)),
               d_vis_imag_real_median=float(np.median(d_ir)), d_vis_start_real_median=float(np.median(d_sr)),
               d_vis_imag_start_median=float(np.median(d_is)), displacement_median=float(np.median(disp)),
               displacement_q10_q90=[float(np.quantile(disp, 0.1)), float(np.quantile(disp, 0.9))],
               tau_pixels_224=px, tau_pixels_196=(px * 196 / 224 if px else None), tau_patches=(px * 196 / 224 / 14 if px else None),
               centroids=[c0, c1], stop_rule_median_ge_1=bool(np.median(ratio) >= 1), time_s=time.time() - t0)
    R.jdump(out, f'{BP.RES}/convcheck_{E.SUF}.json')
    print(json.dumps(out, indent=1))
    if out['stop_rule_median_ge_1']:
        print('CONVCHECK_E7_STOP (median ratio >= 1, DESIGN_main 12.8)')
        sys.exit(2)
    print('CONVCHECK_E7_DONE', f"median ratio {out['ratio_median']:.3f} (x2 actions {out['ratio_x2_median']:.3f}), {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
