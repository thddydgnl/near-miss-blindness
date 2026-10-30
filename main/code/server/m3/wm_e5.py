"""B E5 M3: train the R2-9 predictor from scratch with an exposure-matched prescription (DESIGN_main v3.1 section 9).
Same model, optimiser, schedule, steps (T 6,000), batch (256), init seed (1000 + seed) and sampling generator seed
(77 + seed) as R2-9 wm.py; only the table and the batch composition differ:
  rw   (i)   p<p>_base, 25% failures (64 of 256) + 75% successes
  near (ii)  p<p>_near, 25% failures (failure episodes re-rendered at r in [1, 1.2)) + 75% successes
  cp   (iii) p<p>_cp, 25% near failures + their 25% pair successes + 50% base successes (R2-9 cp composition)
  full (iv)  p<p>_full, 25% failures (re-rendered at r in [1, 3]) + 75% successes
  bw   (v)   rw + object-token weights (lambda_obj from E4c, mean 1 per frame)
Writes ckpt/E5/m3/e5_p<p>_<presc>_s<seed>.pt. usage: python wm_e5.py --p 0.03 --presc near --seed 0 [--lam 1]"""
import os as _os
_os.environ.setdefault('TORCH_HOME', _os.path.expanduser('~/pmx/torch'))
if _os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or _os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')

import os, sys, math, time, json, argparse
import torch
from wm import Predictor, load_norm
from world import FEAT_DIR, N_TRAIN

CK = os.path.expanduser(os.environ.get('B_E5_CK', '~/pmx/B/ckpt/E5/m3'))
EXT = os.path.expanduser(os.environ.get('B_E5_FEAT', '~/pmx/B/feat/E5/m3'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', required=True, choices=['0.03', '0.1'])
    ap.add_argument('--presc', required=True, choices=['rw', 'near', 'cp', 'full', 'bw'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--T', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam', type=float, default=1.0)
    args = ap.parse_args()
    name = f'e5_p{args.p}_{args.presc}_s{args.seed}'
    os.makedirs(CK, exist_ok=True)
    if os.path.exists(f'{CK}/{name}.pt'):
        print('exists', name)
        return
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    B = torch.load(os.path.join(FEAT_DIR, f'bank_s{args.seed}_N{N_TRAIN}.pt'), weights_only=False)
    X = torch.load(f'{EXT}/ext_s{args.seed}.pt', weights_only=False)
    assert X['n_base_frames'] == B['F'].shape[0]
    Fb = torch.cat([B['F'], X['F_ext']]).to(dev)
    del B
    tkey = {'rw': 'base', 'bw': 'base', 'near': 'near', 'full': 'full', 'cp': 'cp'}[args.presc]
    tab = {k: v.to(dev) for k, v in X['tabs'][f'p{args.p}_{tkey}'].items()}
    mu, sd = load_norm(dev)
    seq = torch.cat([tab['f0'][:, None], tab['f']], 1)
    act = tab['act']
    fails = (~tab['succ']).nonzero().squeeze(1)
    succs = tab['succ'].nonzero().squeeze(1)
    g0, g2 = (tab['group'] == 0).nonzero().squeeze(1), (tab['group'] == 2).nonzero().squeeze(1)
    W = None
    if args.presc == 'bw':
        obj = X['masks'][f'p{args.p}_base'][0].to(dev)                  # (n, 4, 64) masks of frames 1..4
        W = 1.0 + (args.lam - 1.0) * obj.float()
        W = W / W.mean(-1, keepdim=True)
    torch.manual_seed(1000 + args.seed)
    model = Predictor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    gen = torch.Generator(device=dev)
    gen.manual_seed(77 + args.seed)
    Bs, k = args.batch, args.batch // 4

    def draw(pool, n):
        return pool[torch.randint(0, len(pool), (n,), generator=gen, device=dev)]

    def sample():
        if args.presc == 'cp':
            f = draw(g2, k)
            tf = torch.randint(0, 4, (k,), generator=gen, device=dev)
            rest = draw(g0, Bs - 2 * k)
            ep = torch.cat([f, tab['partner'][f], rest])
            t = torch.cat([tf, tf, torch.randint(0, 4, (Bs - 2 * k,), generator=gen, device=dev)])
        else:
            ep = torch.cat([draw(fails, k), draw(succs, Bs - k)])
            t = torch.randint(0, 4, (Bs,), generator=gen, device=dev)
        return ep, t

    log = []
    for step in range(args.T):
        lr = args.lr * min(1.0, (step + 1) / 300) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / args.T)))
        for g in opt.param_groups:
            g['lr'] = lr
        ep, t = sample()
        zi = (Fb[seq[ep, t]].float() - mu) / sd
        zo = (Fb[seq[ep, t + 1]].float() - mu) / sd
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(zi, act[ep, t])
        e = ((pred.float() - zo) ** 2).mean(-1)                        # (B, 64)
        loss = (e * W[ep, t]).mean() if W is not None else e.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            log.append(dict(step=step + 1, loss=float(loss), sec=time.time() - t0))
    torch.save(dict(state=model.state_dict(), cfg=vars(args), log=log), f'{CK}/{name}.pt')
    print(f'{name} done loss {float(loss):.4f} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
