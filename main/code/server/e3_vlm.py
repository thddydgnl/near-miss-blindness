"""B E3 M1 judge (d): Qwen2.5-VL-7B-Instruct on (goal frame, final frame) pairs of VQ-VAE-decoded 224 px PushT images.
Same model, transformers version (4.57.1) and score as M2's judge (d): P(Yes) from the next-token logits restricted to
{Yes, No}. bf16 weights with part of them on the CPU (accelerate device_map with a GPU budget) because GPU 1 is shared
with paper A; SDPA attention (no flash-attn wheel for this driver). Prompts fixed before any run (DESIGN_main v2.2
section 7): dset_r, dset_robj (dset goal frame = outcome of the expert segment) and std (DP standard goal frame).
Results are written in chunks (resumable) and merged into <out>_<final key>.npy.
usage: python e3_vlm.py --npz F --variant dset_r|dset_robj|std --pairs goal:final[,goal:imag] --out PREFIX
       [--bs 16] [--gpu_mem 10GiB] [--rows a:b]"""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
    raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh')
os.environ.setdefault('HF_HOME', os.path.expanduser('~/pmx/hf'))
import argparse, json, time, glob
import numpy as np
import torch
from PIL import Image

MID = 'Qwen/Qwen2.5-VL-7B-Instruct'
_HEAD = 'You are checking whether a robot finished a pushing task. These are two top-down images of a 2D scene. '
PROMPTS = {
    'dset_r': _HEAD + ('Each image shows a gray T-shaped block and a small blue circle (the pusher); ignore the light-green '
                       'T-shaped area, which is not the goal here. The first image shows the goal. The second image shows the '
                       'result after the robot pushed. Compare the two images. Are the gray T-shaped block and the blue circle '
                       'both at the same place as in the goal image, with the block in the same orientation, up to a small '
                       'tolerance? Answer with Yes or No.'),
    'dset_robj': _HEAD + ('Each image shows a gray T-shaped block and a small blue circle (the pusher); ignore the light-green '
                          'T-shaped area, which is not the goal here, and ignore the blue circle. The first image shows the goal. '
                          'The second image shows the result after the robot pushed. Compare the two images. Is the gray '
                          'T-shaped block at the same place and in the same orientation as in the goal image, up to a small '
                          'tolerance? Answer with Yes or No.'),
    'std': _HEAD + ('The light-green T-shaped area marks the target. The first image shows the goal: the gray T-shaped block '
                    'placed exactly on the light-green target. The second image shows the result after the robot pushed the '
                    'block. Ignore the small blue circle. Is the gray T-shaped block on the target, at the same place and in '
                    'the same orientation as in the goal image, up to a small tolerance? Answer with Yes or No.'),
}
CH = 1024


def revision():
    snaps = glob.glob(os.path.join(os.environ['HF_HOME'], 'hub', 'models--Qwen--Qwen2.5-VL-7B-Instruct', 'snapshots', '*'))
    return [os.path.basename(s) for s in snaps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--variant', required=True, choices=list(PROMPTS))
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--gpu_mem', default='10GiB')
    ap.add_argument('--rows', default=None)
    args = ap.parse_args()
    t0 = time.time()
    pairs = [p.split(':') for p in args.pairs.split(',')]
    todo = [(g, f) for g, f in pairs if not os.path.exists(f'{args.out}_{f}.npy')]
    if not todo:
        print('done already', args.out)
        return
    D = np.load(args.npz)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import transformers
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MID, dtype=torch.bfloat16, device_map='auto', max_memory={0: args.gpu_mem, 'cpu': '96GiB'},
        attn_implementation='sdpa')
    model.eval()
    proc = AutoProcessor.from_pretrained(MID)
    proc.tokenizer.padding_side = 'left'
    yes_id = proc.tokenizer.encode('Yes', add_special_tokens=False)[0]
    no_id = proc.tokenizer.encode('No', add_special_tokens=False)[0]
    text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'},
                                                                  {'type': 'text', 'text': PROMPTS[args.variant]}]}],
                                    tokenize=False, add_generation_prompt=True)
    dev_map = getattr(model, 'hf_device_map', {})
    n_cpu = sum(1 for v in dev_map.values() if v == 'cpu')
    load_s = time.time() - t0
    print(f'model loaded ({load_s:.0f}s), modules on cpu: {n_cpu}/{len(dev_map)}, gpu mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB',
          flush=True)
    timing = {}
    for gk, fk in todo:
        G, F = D[gk], D[fk]
        gidx = D['goal_idx'] if 'goal_idx' in D.files else None     # one goal frame per episode, indexed per row
        n = len(F)
        a, b = (0, n) if args.rows is None else [int(x) for x in args.rows.split(':')]
        single_goal = G.ndim == 3
        scores = np.full(n, np.nan, np.float32)
        t1 = time.time()
        for c0 in range(a, b, CH):
            c1 = min(b, c0 + CH)
            part = f'{args.out}_{fk}.part{c0:07d}.npy'
            if os.path.exists(part):
                scores[c0:c1] = np.load(part)
                continue
            sc = []
            for i in range(c0, c1, args.bs):
                idx = list(range(i, min(i + args.bs, c1)))
                images = []
                for j in idx:
                    g = G if single_goal else (G[gidx[j]] if gidx is not None else G[j])
                    images += [Image.fromarray(g), Image.fromarray(F[j])]
                inp = proc(text=[text] * len(idx), images=images, return_tensors='pt', padding=True).to('cuda:0')
                with torch.inference_mode():
                    try:
                        lg = model(**inp, logits_to_keep=1).logits[:, -1, :].float()
                    except TypeError:
                        lg = model(**inp).logits[:, -1, :].float()
                sc.append(torch.softmax(lg[:, [yes_id, no_id]], -1)[:, 0].cpu().numpy())
            sc = np.concatenate(sc)
            np.save(part, sc)
            scores[c0:c1] = sc
            print(f'{os.path.basename(args.out)} {fk}: {c1 - a}/{b - a} ({(c1 - a) / (time.time() - t1):.2f} pairs/s)', flush=True)
        if args.rows is None:
            np.save(f'{args.out}_{fk}.npy', scores)
            for p in glob.glob(f'{args.out}_{fk}.part*.npy'):
                os.remove(p)
        timing[fk] = dict(n=int(b - a), s=time.time() - t1)
    meta = dict(model=MID, revision=revision(), transformers=transformers.__version__, torch=torch.__version__,
                variant=args.variant, prompt=PROMPTS[args.variant], pairs=args.pairs, npz=args.npz, bs=args.bs,
                gpu_mem=args.gpu_mem, modules_on_cpu=n_cpu, n_modules=len(dev_map), load_s=load_s, timing=timing,
                yes_id=int(yes_id), no_id=int(no_id))
    with open(f'{args.out}_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('E3_VLM_DONE', json.dumps({k: v for k, v in meta.items() if k != 'prompt'}), flush=True)


if __name__ == '__main__':
    main()
