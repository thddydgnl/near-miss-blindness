"""Paper figures for B (near-miss blindness), drawn only from the frozen result JSON in main/results.

No new computation: every plotted value is read from the analysis JSON written by the E1/E4 stages.
Run from anywhere:  python3 paper/build/make_figs.py
Outputs: paper/figures/fig_boundary.{pdf,png}
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
B_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(B_ROOT, "main", "results")
OUT = os.path.join(B_ROOT, "paper", "figures")

# Palette: dataviz reference slots 1-3 (blue, orange, aqua) and an ordinal blue ramp (steps 250/450/650),
# both validated with scripts/validate_palette.js against a white surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {"0.3": "#86b6ef", "0.1": "#2a78d6", "0.03": "#104281"}
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec"


def load(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_m1(ax):
    """M1 val split, original r, judge (a) ensemble: AUROC by imagination horizon (E1 + E4b ceiling)."""
    e1 = load("E1_server/m1/e1_m1_analysis.json")["cells"]
    e4 = load("E4_server/e4_analysis.json")["4b_M1"]
    hs = [3, 5, 7, 10]
    real_sn = [e1[f"val/H{h}/r"]["a"]["auc_sn_real"] for h in hs]
    imag_sn = [e1[f"val/H{h}/r"]["a"]["auc_sn_imag"] for h in hs]
    imag_sf = [e1[f"val/H{h}/r"]["a"]["auc_sf_imag"] for h in hs]
    ceil_sn = [e4[f"val_H{h}_r"]["imag_ceiling"] for h in hs]
    series = [
        (imag_sf, ORANGE, "-", "D", "full", "far-miss, imagined"),
        (real_sn, BLUE, "--", "o", "none", "near-miss, real frame"),
        (ceil_sn, AQUA, ":", "s", "full", "near-miss, feature probe"),
        (imag_sn, BLUE, "-", "o", "full", "near-miss, imagined"),
    ]
    for ys, c, ls, mk, fill, lab in series:
        ax.plot(hs, ys, color=c, linestyle=ls, linewidth=1.4, marker=mk, markersize=4.5,
                markerfacecolor=(c if fill == "full" else "white"), markeredgecolor=c,
                markeredgewidth=1.1, label=lab, zorder=3)
    ax.set_xticks(hs)
    ax.set_xlim(2.5, 10.5)
    ax.set_ylim(0.64, 1.01)
    ax.set_xlabel("imagination horizon H (planning steps)", fontsize=8, color=INK)
    ax.set_ylabel("AUROC, success vs failure", fontsize=8, color=INK)
    ax.set_title("(a) Public DINO-WM PushT (M1)", fontsize=8.5, color=INK, loc="left", pad=12)
    # guide for the pre-registered cell, drawn only above the legend so it does not cross its text
    ax.axvline(5, ymin=0.55, ymax=1.0, color="#555555", linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
    leg = ax.legend(loc="lower left", fontsize=6.5, frameon=False, handlelength=2.6,
                    borderaxespad=0.2, labelspacing=0.25, title="success vs", title_fontsize=6.5)
    leg._legend_box.align = "left"
    for t in leg.get_texts() + [leg.get_title()]:
        t.set_color(INK)


def panel_m3(ax):
    """M3 controlled latent model, frozen R2-9 judge, 3 seeds pooled: P(judged success) by failure distance r."""
    pooled = load("E1_server/m3/e1_m3_analysis.json")["pooled"]
    ax.axvspan(1.0, 2.0, color=BAND, zorder=0, linewidth=0)
    ax.text(1.42, 1.035, "near-miss band", fontsize=6.5, color=INK2, ha="center", va="bottom")
    ax.text(2.45, 1.035, "extrapolation", fontsize=6.5, color=INK2, ha="center", va="bottom")
    real = pooled["0.03"]["curve"]
    ax.plot(real["r"], real["real"], color=INK2, linestyle="--", linewidth=1.2, label="real frame (all p)", zorder=4)
    for p in ("0.3", "0.1", "0.03"):
        c = pooled[p]["curve"]
        ax.plot(c["r"], c["imag"], color=RAMP[p], linewidth=1.6, label=f"imagined, p = {p}", zorder=3)
    ax.set_xscale("log")
    ticks = [0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.6", "0.8", "1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_xlim(0.6, 3.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("failure distance r = error / threshold (log)", fontsize=8, color=INK)
    ax.set_ylabel("P(judged success)", fontsize=8, color=INK)
    ax.set_title("(b) Controlled latent model (M3)", fontsize=8.5, color=INK, loc="left", pad=12)
    leg = ax.legend(loc="center right", fontsize=6.5, frameon=False, handlelength=2.2, labelspacing=0.25,
                    bbox_to_anchor=(1.0, 0.62))
    for t in leg.get_texts():
        t.set_color(INK)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 1.74), gridspec_kw={"wspace": 0.28})
    for ax in axes:
        style_axes(ax)
    panel_m1(axes[0])
    panel_m3(axes[1])
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.83)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_boundary.{ext}"), dpi=220, facecolor="white")
    print("wrote", os.path.join(OUT, "fig_boundary.pdf"))


if __name__ == "__main__":
    main()
