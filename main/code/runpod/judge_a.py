"""Judge (a) (oracle discriminator on real final frames) for M2, E0b retry with the approved changes:
more training frames (separate start states), the wrist view, and the scripted lift chunk before the judged frame.
Data: /workspace/e0b/cf (gen_e0b.py). Label = LIBERO success at the settled end state (r < 1 and aux).
Split by source demo (start state): train d10-d16, held d17-d19 (d00-d09 are the severity-calibration set).
Models: (1) frozen DINOv2-base features (CLS + mean patch, primary and wrist) + per-task logistic regression
(L2 strength chosen by leave-one-demo-out inside the training demos); (2) small CNN fine-tune (ImageNet ResNet-18,
shared backbone over the two views, per-task heads) if (1) is below 0.95.
Decision metric (fixed before running, worklog 15:2xZ): E1-matched held accuracy = held accuracy reweighted by r band
to E1's planned composition (r < 0.8: 1/6, 0.8-1: 0.0758, 1-2: 0.3788, 2-3: 0.3788, r >= 3: 0); also reported:
raw held accuracy, per band, per task.
Run:  $VP /workspace/code/judge_a.py --stage features|logreg|cnn
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["features", "logreg", "cnn"], required=True)
ap.add_argument("--cf", default="/workspace/e0b/cf")
ap.add_argument("--out", default="/workspace/results/judge_a")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--train_demos", default="10-16", help="training source demos, e.g. 10-16 or 0-16")
ap.add_argument("--tag", default="")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
TASKS = [1, 2, 4, 5, 6, 8]
TRAIN, HELD = set(range(int(args.train_demos.split("-")[0]), int(args.train_demos.split("-")[1]) + 1)), set(range(17, 20))
assert not (TRAIN & HELD)
BANDS = {"r<0.8": (0, 0.8), "0.8-1": (0.8, 1), "1-2": (1, 2), "2-3": (2, 3), ">=3": (3, 1e9)}
E1_W = {"r<0.8": 1 / 6, "0.8-1": 0.0758, "1-2": 0.3788, "2-3": 0.3788, ">=3": 0.0}
# E1 as run (coordinator decision 2026-10-21): per task 100 successes (r < 0.8) + 500 in [0.8, 3) over 11 bins of 0.2
# + 100 far (r >= 3) -> the E1 judge decision metric (worklogs/E1_m2.md 0.2, fixed before regenerating)
E1_W_FAR = {"r<0.8": 100 / 700, "0.8-1": (500 / 11) / 700, "1-2": (2500 / 11) / 700, "2-3": (2500 / 11) / 700,
            ">=3": 100 / 700}


def load_rows():
    rows = []
    for jp in sorted(glob.glob(os.path.join(args.cf, "t*_d*.json"))):
        r = json.load(open(jp))
        if r.get("skipped") or r["task"] not in TASKS or r["demo"] not in (TRAIN | HELD):
            continue
        r["split"] = "train" if r["demo"] in TRAIN else "held"
        rows.append(r)
    return rows


def band_of(r):
    for b, (lo, hi) in BANDS.items():
        if lo <= r < hi:
            return b


def report(rows, pred):
    y = np.array([r["success_settled"] for r in rows], bool)
    ok = pred == y
    bands = np.array([band_of(r["r_settled"]) for r in rows])
    per_band = {b: (float(ok[bands == b].mean()) if (bands == b).any() else None, int((bands == b).sum())) for b in BANDS}
    e1 = sum(E1_W[b] * per_band[b][0] for b in BANDS if per_band[b][0] is not None and E1_W[b] > 0)
    e1 /= sum(E1_W[b] for b in BANDS if per_band[b][0] is not None and E1_W[b] > 0)
    e1f = sum(E1_W_FAR[b] * per_band[b][0] for b in BANDS if per_band[b][0] is not None)
    e1f /= sum(E1_W_FAR[b] for b in BANDS if per_band[b][0] is not None)
    tasks = np.array([r["task"] for r in rows])
    return dict(n=len(rows), acc_raw=float(ok.mean()), acc_e1_matched=float(e1), acc_e1_composition=float(e1f),
                balanced_acc=float(0.5 * (ok[y].mean() + ok[~y].mean())), acc_by_band=per_band,
                acc_by_task={int(t): float(ok[tasks == t].mean()) for t in sorted(set(tasks))})


if args.stage == "features":
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    net = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).cuda().eval()
    rows = load_rows()
    feats = []
    for r in rows:
        z = np.load(os.path.join(args.cf, r["cf_id"] + ".npz"))
        f = []
        for view in ("Slift_primary", "Slift_wrist"):
            with torch.inference_mode():
                x = proc(images=z[view], return_tensors="pt")["pixel_values"].cuda().half()
                h = net(pixel_values=x).last_hidden_state[0].float()
            f.append(torch.cat([h[0], h[1:].mean(0)]).cpu().numpy())
        feats.append(np.concatenate(f))
    np.savez(os.path.join(args.out, "dino_feats.npz"), X=np.stack(feats), ids=np.array([r["cf_id"] for r in rows]))
    print("features", len(rows))

elif args.stage == "logreg":
    rows = load_rows()
    d = np.load(os.path.join(args.out, "dino_feats.npz"))
    pos = {c: i for i, c in enumerate(d["ids"])}
    X = d["X"][[pos[r["cf_id"]] for r in rows]]

    def fit(Xtr, ytr, l2):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        w = torch.zeros(Xtr.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        f = lambda Xe: ((torch.tensor((Xe - mu) / sd, dtype=torch.float32) @ w + b) > 0).detach().numpy()
        f.params = (mu.astype(np.float32), sd.astype(np.float32), w.detach().numpy().astype(np.float32),
                    b.detach().numpy().astype(np.float32))
        return f

    pred = np.zeros(len(rows), bool)
    chosen, exported = {}, {}
    for t in TASKS:
        tr = [i for i, r in enumerate(rows) if r["task"] == t and r["split"] == "train"]
        he = [i for i, r in enumerate(rows) if r["task"] == t and r["split"] == "held"]
        ytr = np.array([rows[i]["success_settled"] for i in tr], float)
        best = None
        for l2 in (1e-4, 1e-3, 1e-2, 1e-1):  # leave-one-demo-out inside the training demos
            accs = []
            for dd in sorted({rows[i]["demo"] for i in tr}):
                a = [i for i in tr if rows[i]["demo"] != dd]
                v = [i for i in tr if rows[i]["demo"] == dd]
                ya = np.array([rows[i]["success_settled"] for i in a], float)
                if len(set(ya)) < 2:
                    continue
                f = fit(X[a], ya, l2)
                accs.append(np.mean(f(X[v]) == np.array([rows[i]["success_settled"] for i in v])))
            m = float(np.mean(accs)) if accs else 0
            if best is None or m > best[1]:
                best = (l2, m)
        chosen[t] = best
        f = fit(X[tr], ytr, best[0])
        pred[he] = f(X[he])
        exported[f"mu_t{t}"], exported[f"sd_t{t}"], exported[f"w_t{t}"], exported[f"b_t{t}"] = f.params
    held = [i for i, r in enumerate(rows) if r["split"] == "held"]
    res = report([rows[i] for i in held], pred[held])
    res["l2_chosen"] = {int(k): v for k, v in chosen.items()}
    res["n_train"] = sum(r["split"] == "train" for r in rows)
    json.dump(res, open(os.path.join(args.out, f"logreg_held{args.tag}.json"), "w"), indent=1)
    os.makedirs("/workspace/judges", exist_ok=True)
    np.savez(f"/workspace/judges/judge_a_logreg{args.tag}.npz", **exported)  # used by b_judges.judge_a inside the runner
    np.savez(os.path.join(args.out, f"judge_a_logreg{args.tag}.npz"), **exported)  # small (6 x 3,072 weights): kept with results
    print(json.dumps(res, indent=1))

else:  # small CNN fine-tune: ImageNet ResNet-18 shared over the two views, per-task heads
    import torchvision
    rows = load_rows()
    def load_xy(rs):
        X = np.stack([np.stack([np.load(os.path.join(args.cf, r["cf_id"] + ".npz"))[v] for v in ("Slift_primary", "Slift_wrist")])
                      for r in rs])  # (N, 2, 224, 224, 3)
        return torch.from_numpy(X), torch.tensor([r["success_settled"] for r in rs], dtype=torch.float32), \
            torch.tensor([TASKS.index(r["task"]) for r in rs])
    tr_rows = [r for r in rows if r["split"] == "train"]
    he_rows = [r for r in rows if r["split"] == "held"]
    Xtr, ytr, ttr = load_xy(tr_rows)
    Xhe, yhe, the = load_xy(he_rows)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).cuda()

    class Judge(nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Linear(1024, len(TASKS))

        def forward(self, x, t):
            B = x.shape[0]
            f = self.backbone(x.reshape(B * 2, 3, x.shape[-2], x.shape[-1])).reshape(B, 1024)
            return self.head(f).gather(1, t[:, None])[:, 0]

    def prep(xb, train):
        x = xb.cuda().permute(0, 1, 4, 2, 3).float() / 255.0
        if train:  # random shift +-8 px and brightness/contrast jitter
            B = x.shape[0] * 2
            flat = x.reshape(B, 3, 224, 224)
            th = torch.zeros(B, 2, 3, device="cuda")
            th[:, 0, 0] = th[:, 1, 1] = 1
            th[:, :, 2] = (torch.rand(B, 2, device="cuda") * 2 - 1) * (8 / 224) * 2
            flat = F.grid_sample(flat, F.affine_grid(th, flat.shape, align_corners=False), padding_mode="border",
                                 align_corners=False)
            flat = (flat * (0.8 + 0.4 * torch.rand(B, 1, 1, 1, device="cuda")) + 0.1 * (torch.rand(B, 1, 1, 1, device="cuda") - 0.5)).clamp(0, 1)
            x = flat.reshape(x.shape)
        return (x - mean) / std

    torch.manual_seed(0)
    model = Judge().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps_per_epoch = (len(tr_rows) + 31) // 32
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.epochs * steps_per_epoch)
    pos_w = (1 - ytr.mean()) / ytr.mean().clamp(min=1e-3)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_rows))
        tot = 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(prep(Xtr[b], True), ttr[b].cuda())
            loss = F.binary_cross_entropy_with_logits(logit.float(), ytr[b].cuda(), pos_weight=pos_w.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        print("epoch", ep, "loss", tot / len(tr_rows), flush=True)
    model.eval()
    with torch.inference_mode():
        logits = torch.cat([model(prep(Xhe[i:i + 64], False), the[i:i + 64].cuda()).float().cpu() for i in range(0, len(he_rows), 64)])
    res = report(he_rows, (logits > 0).numpy())
    res["n_train"] = len(tr_rows)
    json.dump(res, open(os.path.join(args.out, f"cnn_held{args.tag}.json"), "w"), indent=1)
    np.save(os.path.join(args.out, f"cnn_held_logits{args.tag}.npy"), logits.numpy())
    with open(os.path.join(args.out, f"cnn_held_ids{args.tag}.json"), "w") as f:
        json.dump([dict(cf_id=r["cf_id"], task=r["task"], demo=r["demo"], r_settled=r["r_settled"],
                        success_settled=r["success_settled"]) for r in he_rows], f)
    torch.save(model.state_dict(), f"/workspace/e0b/judge_a_cnn{args.tag}.pt")
    print(json.dumps(res, indent=1))
