"""Feasibility of judge (a) (oracle discriminator on real final frames, design 4.1 requires held-out accuracy >= 0.95):
frozen DINOv2-base features (CLS + mean patch token) of the real final third-person frame of every recorded CureWM
counterfactual episode (LIBERO-Goal), success label = LIBERO predicate at the final state (test_margin.py), a
per-task logistic regression evaluated leave-one-source-demo-out. Accuracy is reported overall and by margin band.
Exploratory E0 check (episode end frames often have the gripper over the object; E1 would add a retract chunk).
Run:  $VP /workspace/code/oracle_probe.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

margins = {}
for f in ["/workspace/results/margin/cf_margins.jsonl", "/workspace/results/margin_t9/cf_margins.jsonl"]:
    for l in open(f):
        m = json.loads(l)
        margins[m["key"]] = m
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
net = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).cuda().eval()

feats, rows = [], []
for key, m in sorted(margins.items()):
    t, name = key.split("/")
    with np.load(f"/workspace/data/cf_goal/{t}/episodes/{name}.npz") as z:
        img = z["frames"][-1]
    with torch.inference_mode():
        x = proc(images=img, return_tensors="pt")["pixel_values"].cuda().half()
        h = net(pixel_values=x).last_hidden_state[0].float()
    feats.append(torch.cat([h[0], h[1:].mean(0)]).cpu().numpy())
    rows.append(dict(key=key, task=m["task"], demo=name.split("_")[0], y=int(m["success_libero_final"]), r=m["r_final"]))
X = np.stack(feats)
X = (X - X.mean(0)) / (X.std(0) + 1e-6)


def fit_predict(Xtr, ytr, Xte, l2=1e-2, steps=300):
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (torch.tensor(Xte, dtype=torch.float32) @ w + b).detach().numpy() > 0


pred = np.zeros(len(rows), dtype=bool)
done = np.zeros(len(rows), dtype=bool)
for t in sorted({r["task"] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r["task"] == t]
    demos = sorted({rows[i]["demo"] for i in idx})
    for d in demos:
        te = [i for i in idx if rows[i]["demo"] == d]
        tr = [i for i in idx if rows[i]["demo"] != d]
        ytr = np.array([rows[i]["y"] for i in tr])
        if len(set(ytr)) < 2:
            pred[te] = bool(ytr[0]) if len(ytr) else True
        else:
            pred[te] = fit_predict(X[tr], ytr, X[te])
        done[te] = True
y = np.array([r["y"] for r in rows], dtype=bool)
rr = np.array([r["r"] for r in rows])
acc = lambda m: float((pred[m] == y[m]).mean()) if m.any() else None
res = dict(n=len(rows), acc_all=acc(done), acc_success=acc(done & y), acc_failure=acc(done & ~y),
           acc_by_band={b: acc(done & (rr >= lo) & (rr < hi)) for b, (lo, hi) in
                        {"r<0.8": (0, 0.8), "0.8<=r<1": (0.8, 1), "1<=r<2": (1, 2), "2<=r<3": (2, 3), "r>=3": (3, 1e9)}.items()},
           n_by_band={b: int((done & (rr >= lo) & (rr < hi)).sum()) for b, (lo, hi) in
                      {"r<0.8": (0, 0.8), "0.8<=r<1": (0.8, 1), "1<=r<2": (1, 2), "2<=r<3": (2, 3), "r>=3": (3, 1e9)}.items()},
           acc_by_task={int(t): acc(done & (np.array([r["task"] for r in rows]) == t)) for t in sorted({r["task"] for r in rows})})
json.dump(res, open("/workspace/results/oracle_probe.json", "w"), indent=1)
print(json.dumps(res, indent=1))
