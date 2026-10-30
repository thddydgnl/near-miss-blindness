"""Frame judges and features used inside the M2 runner (GPU process).
  features(): frozen DINOv2-base CLS + mean patch token per view (third-person, wrist) of a 224 px model-view frame;
              saved with every judged frame so judge (a) / AUROC analyses can be recomputed from results only.
  judge_a():  oracle (a) as a per-task logistic regression on the two-view features (weights from judge_a.py logreg);
              returns the logit, or None if no weights file is present.
  judge_a_cnn(): oracle (a) as the fine-tuned ResNet-18 (two views, per-task heads; judge_a.py --stage cnn), the E0b
              primary-judge candidate; returns the logit (> 0 = success), or None if no weights file is present.
  judge_d():  VLM (d) P(Yes) with the start/end comparison prompt (judge_d.py), only when enabled.
"""
from __future__ import annotations

import os

import numpy as np
import torch

_DINO = None
_A = None
_VLM = None
_CNN = None
TASKS_A = [1, 2, 4, 5, 6, 8]


def _dino():
    global _DINO
    if _DINO is None:
        from transformers import AutoImageProcessor, AutoModel
        _DINO = (AutoImageProcessor.from_pretrained("facebook/dinov2-base"),
                 AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).cuda().eval())
    return _DINO


@torch.inference_mode()
def features(primary224: np.ndarray, wrist224: np.ndarray) -> np.ndarray:
    proc, net = _dino()
    out = []
    for im in (primary224, wrist224):
        x = proc(images=np.asarray(im), return_tensors="pt")["pixel_values"].cuda().half()
        h = net(pixel_values=x).last_hidden_state[0].float()
        out.append(torch.cat([h[0], h[1:].mean(0)]).cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def judge_a(task: int, feat: np.ndarray, path="/workspace/judges/judge_a_logreg.npz"):
    global _A
    if _A is None:
        _A = dict(np.load(path)) if os.path.exists(path) else {}
    if f"w_t{task}" not in _A:
        return None
    x = (feat.astype(np.float32) - _A[f"mu_t{task}"]) / _A[f"sd_t{task}"]
    return float(x @ _A[f"w_t{task}"] + _A[f"b_t{task}"])


class JudgeCNN(torch.nn.Module):
    """Same architecture as judge_a.py (--stage cnn)."""

    def __init__(self):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Identity()
        self.backbone = m
        self.head = torch.nn.Linear(1024, len(TASKS_A))

    def forward(self, x, t):
        B = x.shape[0]
        f = self.backbone(x.reshape(B * 2, 3, x.shape[-2], x.shape[-1])).reshape(B, 1024)
        return self.head(f).gather(1, t[:, None])[:, 0]


@torch.inference_mode()
def judge_a_cnn(task: int, primary224: np.ndarray, wrist224: np.ndarray, path="/workspace/judges/judge_a_cnn.pt"):
    global _CNN
    if _CNN is None:
        if not os.path.exists(path):
            _CNN = False
        else:
            m = JudgeCNN().cuda().eval()
            m.load_state_dict(torch.load(path, map_location="cuda"))
            _CNN = m
    if _CNN is False or task not in TASKS_A:
        return None
    x = torch.from_numpy(np.stack([primary224, wrist224])[None]).cuda().permute(0, 1, 4, 2, 3).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return float(_CNN((x - mean) / std, torch.tensor([TASKS_A.index(task)], device="cuda")).float())


VLM_PROMPT = ("You are checking whether a robot finished a manipulation task. The task was: \"{task}\". "
              "The first image shows the scene before the robot started. The second image shows the scene at the end, "
              "after the robot opened its gripper and moved its arm up. Compare the two images. "
              "Was the task completed successfully? Answer with Yes or No.")  # judge_d.py "compare2"


@torch.inference_mode()
def judge_d(start224: np.ndarray, end224: np.ndarray, language: str, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    """VLM (d) P(Yes) with the start/end comparison prompt (E0b judge_d.py compare2). Loaded lazily from the
    cache in $B_VLM_CACHE (runner: --vlm)."""
    global _VLM
    from PIL import Image
    if _VLM is None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        cache = os.environ.get("B_VLM_CACHE")  # e.g. /workspace/cache/hf_vlm/hub (keeps HF_HOME for Cosmos/DINOv2)
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda",
                                                               attn_implementation="flash_attention_2", cache_dir=cache)
        pr = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
        _VLM = (m, pr, pr.tokenizer.encode("Yes", add_special_tokens=False)[0], pr.tokenizer.encode("No", add_special_tokens=False)[0])
    m, pr, yes_id, no_id = _VLM
    content = [{"type": "image"}, {"type": "image"}, {"type": "text", "text": VLM_PROMPT.format(task=language)}]
    text = pr.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inp = pr(text=[text], images=[Image.fromarray(np.asarray(start224)), Image.fromarray(np.asarray(end224))],
             return_tensors="pt").to("cuda")
    logits = m(**inp).logits[0, -1, :].float()
    return float(torch.softmax(logits[[yes_id, no_id]], -1)[0])
