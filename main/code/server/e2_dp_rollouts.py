"""B E2 M1 step 1: roll out the 24 Diffusion-Policy checkpoints (seed 42/43/44 x epoch 20,25,30,40,50,80,220,1000; EMA
weights) on the E2 evaluation initial states, seeds 100000-100099, shared by every checkpoint (paired design, DESIGN_main
v1 section 0.2). Protocol = dp_rollout.py (DINO-WM PushT env, DP initial-state formula, n_obs 2, n_action 8, DDPM 100,
stop at coverage > 0.95 or 300 control steps; recorded relative actions a_t = (A_t - agent_t)/100).
One file per checkpoint (resumable). usage: python e2_dp_rollouts.py [s42_ep0020,s43_ep0025,...]"""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import dp_rollout as DPR

SEEDS_DP = (42, 43, 44)
STAGES = (20, 25, 30, 40, 50, 80, 220, 1000)
EVAL_SEEDS = list(range(100000, 100100))
ALL = [f's{s}_ep{e:04d}' for s in SEEDS_DP for e in STAGES]


def main():
    out_dir = f'{BP.RES}/dp_rollouts'
    os.makedirs(out_dir, exist_ok=True)
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else ALL
    device = torch.device('cuda:0')
    t0 = time.time()
    for n in names:
        o = f'{out_dir}/{n}.pt'
        if os.path.exists(o):
            print('exists', n, flush=True)
            continue
        s, e = n.split('_')
        ckpt = os.path.expanduser(f'~/pmx/B/ckpt/dp/{s}/ema_{e}.pt')
        t1 = time.time()
        pol, _ = DPR.load_policy(ckpt, device)
        eps = DPR.rollout(pol, EVAL_SEEDS, device, batch=100, torch_seed=0, log=lambda *a, **k: None)
        summ = DPR.summarize(eps)
        summ.update(ckpt=ckpt, sec=time.time() - t1)
        torch.save(dict(episodes=eps, summary=summ, seeds=EVAL_SEEDS), o + '.tmp')
        os.replace(o + '.tmp', o)
        print(f'{n}: success_robj {summ["success_robj"]:.2f} near {summ["near_robj"]:.2f} far {summ["far_robj"]:.2f} '
              f'success_dp {summ["success_dp"]:.2f} T_end q10/50/90 {summ["T_end_q"]} ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)',
              flush=True)
        del pol
        torch.cuda.empty_cache()
    print('E2_DP_ROLLOUTS_DONE', ','.join(names), flush=True)


if __name__ == '__main__':
    main()
