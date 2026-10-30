"""Follow-up to vlm_bench.py (same 50 real final frames): does judge (d) discriminate better with an object-centric
question per task and/or with the 224 px frame upsampled to 448 px? Exploratory E0 check of the VLM perceptual floor.
Run:  HF_HOME=/workspace/cache/hf_vlm $VP /workspace/code/vlm_bench2.py
"""
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MID = "Qwen/Qwen2.5-VL-7B-Instruct"
QUESTION = {
    "t1": "Is the black bowl sitting on the stove's cooking area?",
    "t2": "Is the wine bottle standing on top of the wooden cabinet?",
    "t5": "Is the plate located directly in front of the stove?",
    "t6": "Is the cream cheese box inside the black bowl?",
    "t8": "Is the black bowl sitting on the plate?",
}
PROMPTS = {
    "task_success": ("The image shows a robot workspace after the robot finished its attempt at the task: \"{task}\". "
                     "Was the task completed successfully? Answer with Yes or No."),
    "object_question": "Look at the robot workspace in the image. {question} Answer with Yes or No.",
}
prev = json.load(open("/workspace/results/throughput/vlm_bench.json"))
items = prev["items"]
for it in items:
    t, name = it["key"].split("/")
    with np.load(f"/workspace/data/cf_goal/{t}/episodes/{name}.npz") as z:
        it["img224"] = Image.fromarray(z["frames"][-1])
    it["img448"] = it["img224"].resize((448, 448), Image.BICUBIC)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda",
                                                           attn_implementation="flash_attention_2")
proc = AutoProcessor.from_pretrained(MID)
proc.tokenizer.padding_side = "left"
yes_id = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = proc.tokenizer.encode("No", add_special_tokens=False)[0]


@torch.inference_mode()
def judge(batch, pkey, ikey):
    texts = []
    for b in batch:
        q = PROMPTS[pkey].format(task=b["task"], question=QUESTION[b["key"].split("/")[0]])
        texts.append(proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                                              tokenize=False, add_generation_prompt=True))
    inp = proc(text=texts, images=[b[ikey] for b in batch], return_tensors="pt", padding=True).to("cuda")
    logits = model(**inp).logits[:, -1, :].float()
    return torch.softmax(logits[:, [yes_id, no_id]], -1)[:, 0].cpu().numpy()


def auroc(s, f):
    return float(sum((a > b) + 0.5 * (a == b) for a in s for b in f) / (len(s) * len(f)))


res = {}
for pkey in PROMPTS:
    for ikey in ["img224", "img448"]:
        sc = []
        for i in range(0, len(items), 8):
            sc.extend(judge(items[i:i + 8], pkey, ikey).tolist())
        s = [p for p, it in zip(sc, items) if it["success"]]
        f = [p for p, it in zip(sc, items) if not it["success"]]
        res[f"{pkey}/{ikey}"] = dict(auroc=auroc(s, f), mean_p_yes_success=float(np.mean(s)), mean_p_yes_failure=float(np.mean(f)),
                                     frac_yes_success=float(np.mean([p > 0.5 for p in s])),
                                     frac_yes_failure=float(np.mean([p > 0.5 for p in f])),
                                     per_task_auroc={t: auroc([p for p, it in zip(sc, items) if it["success"] and it["key"].startswith(t + "/")],
                                                              [p for p, it in zip(sc, items) if not it["success"] and it["key"].startswith(t + "/")])
                                                     for t in QUESTION})
        print(pkey, ikey, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"{pkey}/{ikey}"].items()}), flush=True)
json.dump(res, open("/workspace/results/throughput/vlm_bench2.json", "w"), indent=1)
