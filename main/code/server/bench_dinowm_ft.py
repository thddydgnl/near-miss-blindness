"""B E0 throughput micro-benchmark (no results used for any decision): time per DINO-WM PushT predictor fine-tuning step
(E4a success-biased / balanced fine-tuning and E5 data-side prescriptions on M1). Same model / loss as dino_wm train.py
(frozen DINOv2 encoder, predictor + proprio/action encoders trained, decoder not trained), batch 32 (config), slices of
num_hist 3 + num_pred 1 frames at frameskip 5 from the train split videos. A copy of the model is trained; the shared
checkpoint is not modified.
usage: python bench_dinowm_ft.py [n_steps=30]"""
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
import r21lib as L


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    L.patch_attention(True)
    T = L.load_task('pusht', load_dset=True)
    m, cfg = T['model'], T['cfg']
    import hydra
    datasets, _ = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    ds = datasets['train']
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    for p in m.predictor.parameters():
        p.requires_grad_(True)
    for mod in (m.proprio_encoder, m.action_encoder):
        for p in mod.parameters():
            p.requires_grad_(True)
    m.train()
    m.encoder.eval()
    params = [{'params': m.predictor.parameters(), 'lr': cfg.training.predictor_lr},
              {'params': list(m.proprio_encoder.parameters()) + list(m.action_encoder.parameters()),
               'lr': cfg.training.action_encoder_lr}]
    opt = torch.optim.AdamW(params)
    it = iter(loader)
    times, data_t = [], []
    for s in range(n_steps):
        t0 = time.time()
        obs, act, state = next(it)
        t1 = time.time()
        obs = {k: v.cuda(non_blocking=True) for k, v in obs.items()}
        act = act.cuda(non_blocking=True)
        z_pred, _, _, loss, lc = m(obs, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t1)
        data_t.append(t1 - t0)
        if s % 10 == 0:
            print(f'step {s} loss {float(loss):.4f} gpu {times[-1]:.3f}s data {data_t[-1]:.3f}s', flush=True)
    res = dict(n_slices=len(ds), batch=32, gpu_step_s_median=float(np.median(times[5:])), data_s_median=float(np.median(data_t[5:])),
               peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, slices_per_epoch=len(ds))
    print('BENCH', json.dumps(res))
    L.jdump(res, f'{BP.RES}/bench_dinowm_ft.json')


if __name__ == '__main__':
    main()
