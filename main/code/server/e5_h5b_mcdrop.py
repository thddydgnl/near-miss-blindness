"""B E5 H5b (M1, evaluation side, no retraining; DESIGN_main v3.1 section 9): MC-dropout imagination of the mode-O H5
windows with the public DINO-WM checkpoint. K = 16 stochastic rollouts per window with every predictor dropout layer
active as in training (attention, MLP and projection dropout 0.1; DINOv2, proprio / action encoders in eval mode),
torch seed 1000 + k for sample k. Each sample's imagined final frame is judged by the a_std ensemble (> 0.5);
a = fraction of the K samples judged success.
  dev   the a_std judge-training rollouts (24 checkpoints x 25 episodes, seeds 300000-300599; disjoint from E2)
  eval  the E2 evaluation rollouts (24 checkpoints x 100 episodes, seeds 100000-100099)
Windows are rebuilt with e2_dp_windows.window_job (H = 5). Writes results/E5/m1/mcdrop_<set>.npz.
usage: python e5_h5b_mcdrop.py dev|eval"""
import os, sys, time, json, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r12lib as R
import e1lib as E
from e2_dp_windows import window_job
from e4_m1_finetune import patch_attention_train

K = 16
SETS = {'dev': os.path.expanduser('~/pmx/B/results/E1/m1/astd_rollouts'), 'eval': os.path.expanduser('~/pmx/B/results/E2/m1/dp_rollouts')}
E1_FEAT = os.path.expanduser('~/pmx/B/feat/E1/m1')
OUT = os.path.expanduser('~/pmx/B/results/E5/m1')


def main():
    st = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    out = f'{OUT}/mcdrop_{st}.npz'
    if os.path.exists(out):
        print('exists', out)
        return
    t0 = time.time()
    eps, pol, epi = [], [], []
    for f in sorted(glob.glob(f'{SETS[st]}/s*_ep*.pt')):
        D = torch.load(f, weights_only=False)
        for k, e in enumerate(D['episodes']):
            eps.append(e); pol.append(os.path.basename(f)[:-3]); epi.append(k)
    T_ = R.load_model()
    patch_attention_train()
    m, pre = T_['model'], T_['pre']
    A = torch.load(f'{E1_FEAT}/astd_e1.pt', weights_only=False)
    nets = E.load_ensemble(A['sds'])
    zgC = A['zgC']
    pool = R.pool()
    wins = pool.map(window_job, [(e, (5,)) for e in eps], chunksize=4)
    pool.close()
    w = [x[0] for x in wins]
    simg, spro = np.stack([x['img0'] for x in w]), np.stack([x['pro0'] for x in w])
    acts = np.stack([x['acts'] for x in w])
    r_obj = np.array([e['r_obj'] for e in eps])
    zr, _ = R.encode_frames(m, pre, np.stack([x['imgf'] for x in w]), np.stack([x['prof'] for x in w]))
    zg = zgC.expand(len(zr), -1, -1)
    _, p_real = E.ens_prob(nets, zr, zg)
    m.eval()
    _, p_det = E.ens_prob(nets, R.imagine(m, pre, simg, spro, acts)[0], zg)            # deterministic (dropout off)
    m.predictor.train()                                                                 # MC dropout: predictor only
    P = np.zeros((K, len(eps)), np.float32)
    for k in range(K):
        torch.manual_seed(1000 + k)
        zi, _ = R.imagine(m, pre, simg, spro, acts)
        _, P[k] = E.ens_prob(nets, zi, zg)
        print(f'{st}: MC sample {k + 1}/{K} ({time.time() - t0:.0f}s)', flush=True)
    m.eval()
    np.savez_compressed(out, policy=np.array(pol), ep=np.array(epi), r_obj=r_obj, p_real=p_real.astype(np.float32),
                        p_det=p_det.astype(np.float32), p_mc=P, replay_err=np.array([x['replay_err'] for x in w]))
    print('E5_MCDROP_DONE', st, json.dumps(dict(n=len(eps), mean_agreement=float(((P > 0.5).mean(0)).mean()), time_s=time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()
