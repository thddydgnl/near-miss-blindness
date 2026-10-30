import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ex = torch.load("/workspace/feat9/example_frames.pt")
fig, ax = plt.subplots(2, 5, figsize=(11, 4.6))
for row, (r, k0) in enumerate(((0.5, 0), (1.1, 10))):
    lab = "success" if r < 1 else "near-miss"
    for t in range(5):
        ax[row, t].imshow(ex[k0 + 2 * t].permute(1, 2, 0).clamp(0, 1).numpy())
        ax[row, t].axis("off")
        ax[row, t].set_title("r=%.1f (%s), f%d" % (r, lab, t), fontsize=8)
fig.tight_layout()
fig.savefig("/workspace/results9/fig0_example_episodes.png", dpi=100)
print("ok")
