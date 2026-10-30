"""B E0: Diffusion Policy (Chi et al. 2023), CNN U-Net, low-dim (keypoint) PushT, standard demos
(pusht_cchi_v7_replay.zarr, standard config: horizon 16, n_obs 2, n_action 8, 90 train episodes, dataset seed 42,
batch 256, AdamW 1e-4, cosine with 500 warm-up steps, EMA power 0.75, DDPM 100 steps).
The loop is the one of diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py without the env runner
and wandb (policies are evaluated later in the DINO-WM PushT env). Saves EMA-only snapshots at the given epochs
(epoch = number of completed epochs) and a full resumable checkpoint (latest.pt) every --ckpt_every epochs.
usage: python train_dp.py --seed 42 --out ~/pmx/B/ckpt/dp/s42 --epochs 2000 --snaps 5,10,20,...
"""
import argparse, copy, json, math, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpaths as BP
sys.path.insert(0, BP.DP_REPO)


def load_cfg():
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    base = os.path.join(BP.DP_REPO, 'diffusion_policy', 'config')
    cfg = OmegaConf.load(os.path.join(base, 'train_diffusion_unet_lowdim_workspace.yaml'))
    cfg.task = OmegaConf.load(os.path.join(base, 'task', 'pusht_lowdim.yaml'))
    del cfg['defaults']
    cfg.task.dataset.zarr_path = BP.DP_DATA
    return cfg


def build(cfg, device):
    import hydra
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    policy = hydra.utils.instantiate(cfg.policy)
    ema_model = copy.deepcopy(policy)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    ema_model.set_normalizer(normalizer)
    policy.to(device)
    ema_model.to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
    ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
    return policy, ema_model, ema, optimizer, dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--snaps', default='')
    ap.add_argument('--ckpt_every', type=int, default=25)
    ap.add_argument('--max_steps_per_epoch', type=int, default=0, help='smoke only')
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    snaps = sorted({int(x) for x in args.snaps.split(',') if x})
    device = torch.device('cuda:0')
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from diffusion_policy.common.pytorch_util import dict_apply
    from torch.utils.data import DataLoader
    cfg = load_cfg()
    cfg.training.seed = args.seed
    cfg.training.num_epochs = args.epochs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    policy, ema_model, ema, optimizer, dataset = build(cfg, device)
    train_loader = DataLoader(dataset, **cfg.dataloader)
    val_loader = DataLoader(dataset.get_validation_dataset(), **cfg.val_dataloader)
    n_params = sum(p.numel() for p in policy.model.parameters())
    epoch, global_step = 0, 0
    latest = os.path.join(out, 'latest.pt')
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        policy.load_state_dict(ck['model'])
        ema_model.load_state_dict(ck['ema_model'])
        optimizer.load_state_dict(ck['optimizer'])
        ema.decay, ema.optimization_step = ck['ema_decay'], ck['ema_optimization_step']
        epoch, global_step = ck['epoch'], ck['global_step']
        torch.set_rng_state(ck['rng_torch'].cpu()); torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
        np.random.set_state(ck['rng_np']); random.setstate(ck['rng_py'])
        print(f'resumed from {latest}: epoch {epoch} step {global_step}', flush=True)
    lr_scheduler = get_scheduler(cfg.training.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=cfg.training.lr_warmup_steps,
                                 num_training_steps=(len(train_loader) * cfg.training.num_epochs)
                                 // cfg.training.gradient_accumulate_every,
                                 last_epoch=global_step - 1)
    meta = dict(seed=args.seed, epochs=args.epochs, snaps=snaps, n_params=int(n_params), n_train_samples=len(dataset),
                batches_per_epoch=len(train_loader), dataset_seed=int(cfg.task.dataset.seed),
                max_train_episodes=int(cfg.task.dataset.max_train_episodes), zarr=BP.DP_DATA,
                dp_commit='5ba07ac6661db573af695b419a7947ecb704690f', torch=torch.__version__)
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('META', json.dumps(meta), flush=True)
    logf = open(os.path.join(out, 'train_log.jsonl'), 'a')
    t_start = time.time()
    while epoch < args.epochs:
        t0 = time.time()
        policy.train()
        losses = []
        for bi, batch in enumerate(train_loader):
            if args.max_steps_per_epoch and bi >= args.max_steps_per_epoch:
                break
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            raw_loss = policy.compute_loss(batch)
            loss = raw_loss / cfg.training.gradient_accumulate_every
            loss.backward()
            if global_step % cfg.training.gradient_accumulate_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            ema.step(policy)
            losses.append(float(raw_loss))
            global_step += 1
        epoch += 1
        rec = dict(epoch=epoch, step=global_step, train_loss=float(np.mean(losses)), lr=float(lr_scheduler.get_last_lr()[0]),
                   ema_decay=float(ema.decay), sec=time.time() - t0)
        if epoch % 25 == 0 or epoch in snaps or epoch == args.epochs:
            policy.eval()
            with torch.no_grad():
                vl = [float(policy.compute_loss(dict_apply(b, lambda x: x.to(device, non_blocking=True))))
                      for b in val_loader]
            rec['val_loss'] = float(np.mean(vl))
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        if epoch % 10 == 0 or epoch in snaps or epoch <= 3:
            el = time.time() - t_start
            vtxt = f"{rec['val_loss']:.4f}" if 'val_loss' in rec else 'skip'   # val runs every 25 epochs and at snapshots
            print(f"s{args.seed} epoch {epoch}/{args.epochs} loss {rec['train_loss']:.4f} "
                  f"val {vtxt} lr {rec['lr']:.2e} {rec['sec']:.1f}s/ep", flush=True)
        if epoch in snaps:
            p = os.path.join(out, f'ema_ep{epoch:04d}.pt')
            torch.save(dict(ema_model=ema_model.state_dict(), epoch=epoch, global_step=global_step, seed=args.seed), p + '.tmp')
            os.replace(p + '.tmp', p)
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            ck = dict(model=policy.state_dict(), ema_model=ema_model.state_dict(), optimizer=optimizer.state_dict(),
                      ema_decay=ema.decay, ema_optimization_step=ema.optimization_step, epoch=epoch,
                      global_step=global_step, rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
                      rng_np=np.random.get_state(), rng_py=random.getstate())
            torch.save(ck, latest + '.tmp')
            os.replace(latest + '.tmp', latest)
    open(os.path.join(out, 'DONE'), 'w').write(f'{epoch} {time.time() - t_start:.0f}s\n')
    print(f'TRAIN_DONE s{args.seed} epoch {epoch}', flush=True)


if __name__ == '__main__':
    main()
