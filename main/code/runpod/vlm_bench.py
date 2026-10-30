"""Judge (d) throughput check for E3 cost estimation: a public VLM (Qwen2.5-VL-7B-Instruct, Apache-2.0) scoring
final third-person frames with a fixed yes/no prompt that includes the task sentence. Score = P(Yes) from the
next-token logits restricted to {Yes, No}. Uses real final frames of recorded CureWM counterfactual episodes
(success and failure) so the numbers also give a first look at the judge's perceptual floor (exploratory, E0 only).
Run:  HF_HOME=/workspace/cache/hf_vlm $VP /workspace/code/vlm_bench.py
"""
import glob
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = ("The image shows a robot workspace after the robot finished its attempt at the task: \"{task}\". "
          "Was the task completed successfully? Answer with Yes or No.")
OUT = "/workspace/results/throughput/vlm_bench.json"

margins = {json.loads(l)["key"]: json.loads(l) for l in open("/workspace/results/margin/cf_margins.jsonl")}
items = []
for t in [1, 5, 8, 2, 6]:
    idx = [json.loads(l) for l in open(f"/workspace/data/cf_goal/t{t}/index.jsonl")]
    random.Random(t).shuffle(idx)
    succ = [r for r in idx if margins.get(f"t{t}/{r['name']}", {}).get("success_libero_final") is True][:5]
    fail = [r for r in idx if margins.get(f"t{t}/{r['name']}", {}).get("success_libero_final") is False][:5]
    for r in succ + fail:
        with np.load(f"/workspace/data/cf_goal/t{t}/episodes/{r['name']}.npz") as z:
            img = z["frames"][-1]
        items.append(dict(key=f"t{t}/{r['name']}", task=r["language"], img=Image.fromarray(img),
                          success=margins[f"t{t}/{r['name']}"]["success_libero_final"],
                          r=margins[f"t{t}/{r['name']}"]["r_final"]))
print("items", len(items), flush=True)

t0 = time.time()
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map="cuda",
                                                               attn_implementation="flash_attention_2")
except Exception as e:  # fall back to PyTorch SDPA attention
    print("flash_attention_2 unavailable:", e, flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map="cuda",
                                                               attn_implementation="sdpa")
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"
load_s = time.time() - t0
yes_id = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = proc.tokenizer.encode("No", add_special_tokens=False)[0]


@torch.inference_mode()
def judge(batch):
    texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                                                     {"type": "text", "text": PROMPT.format(task=b["task"])}]}],
                                      tokenize=False, add_generation_prompt=True) for b in batch]
    inp = proc(text=texts, images=[b["img"] for b in batch], return_tensors="pt", padding=True).to("cuda")
    logits = model(**inp).logits[:, -1, :].float()
    return torch.softmax(logits[:, [yes_id, no_id]], -1)[:, 0].cpu().numpy()


res = {"model": MID, "load_s": load_s, "n_items": len(items)}
judge(items[:2])
for B in [1, 8]:
    torch.cuda.synchronize(); t = time.time()
    scores = []
    for i in range(0, len(items), B):
        scores.extend(judge(items[i:i + B]).tolist())
    torch.cuda.synchronize()
    res[f"batch{B}_s_per_judgment"] = (time.time() - t) / len(items)
res["vram_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
for it, s in zip(items, scores):
    it["p_yes"] = float(s)
succ = [it["p_yes"] for it in items if it["success"]]
fail = [it["p_yes"] for it in items if not it["success"]]
res["mean_p_yes_success"] = float(np.mean(succ)) if succ else None
res["mean_p_yes_failure"] = float(np.mean(fail)) if fail else None
res["frac_yes_failure"] = float(np.mean([p > 0.5 for p in fail])) if fail else None
res["frac_yes_success"] = float(np.mean([p > 0.5 for p in succ])) if succ else None
res["items"] = [{k: v for k, v in it.items() if k != "img"} for it in items]
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))
