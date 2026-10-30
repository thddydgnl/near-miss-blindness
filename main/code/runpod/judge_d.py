"""Judge (d) (public VLM, fixed prompt including the task sentence) for M2, E0b retry in the approved order:
(1) the scripted lift chunk + a start/end comparison prompt with Qwen2.5-VL-7B-Instruct; (2) a 32B VLM
(Qwen2.5-VL-32B-Instruct, 4-bit NF4 via bitsandbytes because bf16 does not fit a 48 GB A40).
Images are the Cosmos model-view frames (224 px): start = S0 third-person, end = S_lift third-person (+ S_lift wrist
in variant "compare3"). Score = P(Yes) from the next-token logits restricted to {Yes, No}.
Criterion (fixed before E1 data, worklog): held split d17-d19 AUROC(success vs failure) >= 0.80. The Yes threshold
is chosen on the training split d10-d16 (max balanced accuracy) and fixed before E1.
Run:  HF_HOME=/workspace/cache/hf_vlm $VP /workspace/code/judge_d.py --model 7b --variants compare2 compare3
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["7b", "32b"], default="7b")
ap.add_argument("--variants", nargs="+", default=["compare2", "compare3"])
ap.add_argument("--cf", default="/workspace/e0b/cf")
ap.add_argument("--out", default="/workspace/results/judge_d")
ap.add_argument("--max_per_split", type=int, default=100000)
ap.add_argument("--max_train", type=int, default=0, help="seeded random subset of the training split (threshold choice only)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
TASKS = [1, 2, 4, 5, 6, 8]
TRAIN, HELD = set(range(10, 17)), set(range(17, 20))
PROMPTS = {
    "compare2": ("You are checking whether a robot finished a manipulation task. The task was: \"{task}\". "
                 "The first image shows the scene before the robot started. The second image shows the scene at the end, "
                 "after the robot opened its gripper and moved its arm up. Compare the two images. "
                 "Was the task completed successfully? Answer with Yes or No."),
    "compare3": ("You are checking whether a robot finished a manipulation task. The task was: \"{task}\". "
                 "The first image shows the scene before the robot started. The second image shows the scene at the end, "
                 "after the robot opened its gripper and moved its arm up. The third image is the robot's wrist camera at the "
                 "end, looking down at the table. Was the task completed successfully? Answer with Yes or No."),
}
MID = {"7b": "Qwen/Qwen2.5-VL-7B-Instruct", "32b": "Qwen/Qwen2.5-VL-32B-Instruct"}[args.model]

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="flash_attention_2")
if args.model == "32b":
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, **kw)
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"
load_s = time.time() - t0
yes_id = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = proc.tokenizer.encode("No", add_special_tokens=False)[0]

rows = []
for jp in sorted(glob.glob(os.path.join(args.cf, "t*_d*.json"))):
    r = json.load(open(jp))
    if r.get("skipped") or r["task"] not in TASKS or r["demo"] not in (TRAIN | HELD):
        continue
    r["split"] = "train" if r["demo"] in TRAIN else "held"
    rows.append(r)
rows = [r for r in rows if r["split"] == "held"][: args.max_per_split] + [r for r in rows if r["split"] == "train"][: args.max_per_split]
if args.max_train:
    import random
    tr = [r for r in rows if r["split"] == "train"]
    random.Random(0).shuffle(tr)
    rows = [r for r in rows if r["split"] == "held"] + tr[: args.max_train]


@torch.inference_mode()
def judge(batch, variant):
    texts, images = [], []
    n_img = 2 if variant == "compare2" else 3
    for r in batch:
        z = np.load(os.path.join(args.cf, r["cf_id"] + ".npz"))
        ims = [z["S0_primary"], z["Slift_primary"]] + ([z["Slift_wrist"]] if n_img == 3 else [])
        content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": PROMPTS[variant].format(task=r["language"])}]
        texts.append(proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
        images.extend(Image.fromarray(x) for x in ims)
    inp = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
    logits = model(**inp).logits[:, -1, :].float()
    return torch.softmax(logits[:, [yes_id, no_id]], -1)[:, 0].cpu().numpy()


def auroc(s, f):
    s, f = np.asarray(s), np.asarray(f)
    if len(s) == 0 or len(f) == 0:
        return None
    return float(((s[:, None] > f[None]).sum() + 0.5 * (s[:, None] == f[None]).sum()) / (len(s) * len(f)))


res = {"model": MID, "load_s": load_s, "n": len(rows)}
B = 8 if args.model == "7b" else 4
for v in args.variants:
    t1 = time.time()
    sc = []
    for i in range(0, len(rows), B):
        sc.extend(judge(rows[i:i + B], v).tolist())
    dt = (time.time() - t1) / len(rows)
    y = np.array([r["success_settled"] for r in rows], bool)
    sc = np.array(sc)
    split = np.array([r["split"] for r in rows])
    rr = np.array([r["r_settled"] for r in rows])
    out = {"s_per_judgment": dt}
    for sp in ("train", "held"):
        m = split == sp
        out[f"auroc_{sp}"] = auroc(sc[m & y], sc[m & ~y])
        out[f"auroc_{sp}_success_vs_near"] = auroc(sc[m & y], sc[m & ~y & (rr >= 1) & (rr < 2)])
        out[f"auroc_{sp}_success_vs_far"] = auroc(sc[m & y], sc[m & ~y & (rr >= 3)])
    # threshold on the training split (max balanced accuracy), applied to held
    m = split == "train"
    best = max(((0.5 * ((sc[m & y] > th).mean() + (sc[m & ~y] <= th).mean()), th) for th in np.unique(sc[m])), default=(None, 0.5))
    th = float(best[1])
    mh = split == "held"
    out.update(threshold=th, held_balanced_acc=float(0.5 * ((sc[mh & y] > th).mean() + (sc[mh & ~y] <= th).mean())),
               held_acc=float(((sc[mh] > th) == y[mh]).mean()),
               held_auroc_by_task={t: auroc(sc[mh & y & (np.array([r["task"] for r in rows]) == t)],
                                            sc[mh & ~y & (np.array([r["task"] for r in rows]) == t)]) for t in TASKS},
               passes_criterion=bool(out["auroc_held"] is not None and out["auroc_held"] >= 0.80))
    res[v] = out
    np.save(os.path.join(args.out, f"scores_{args.model}_{v}.npy"), sc)
    print(v, json.dumps(out), flush=True)
json.dump(dict(res, ids=[r["cf_id"] for r in rows]), open(os.path.join(args.out, f"judge_d_{args.model}.json"), "w"), indent=1)
