"""Figures for the TMLR manuscript of B (near-miss blindness).

Diagrams in Figure 1 are drawn by hand (panel (b) is a schematic, not data). Every data figure reads the frozen
analysis JSON in main/results; nothing is recomputed beyond selecting and rescaling values for display.
Run from anywhere:  python3 paper/tmlr/build/make_figs.py [--preview DIR]
Outputs: paper/tmlr/figures/fig*.pdf (vector, fonts embedded). --preview also writes 200-dpi PNG copies to DIR.
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TMLR = os.path.normpath(os.path.join(HERE, ".."))
B_ROOT = os.path.normpath(os.path.join(TMLR, "..", ".."))
RES = os.path.join(B_ROOT, "main", "results")
FIG = os.path.join(TMLR, "figures")

# Palette: dataviz reference categorical slots 1-3 (validated: scripts/validate_palette.js, light mode), the ordinal
# blue ramp (steps 250/450/650) and chart ink. Identity never rests on color alone: every series also differs in line
# style or marker, so the figures read in grayscale.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP3 = ("#86b6ef", "#2a78d6", "#104281")
GRAY, GRAY_LIGHT, GRAY_DARK = "#a8a69f", "#cfcdc6", "#7d7b75"
INK, INK2, MUTED = "#2b2a27", "#52514e", "#898781"
GRID, AXIS = "#e6e5df", "#c3c2b7"
TINT = {"gray": ("#f4f3f0", "#d6d4cd"), "blue": ("#edf4fd", "#b7d3f6"), "orange": ("#fdf1eb", "#f3c4ad"),
        "aqua": ("#e8f7f0", "#a9dfc9"), "white": ("#ffffff", "#d6d4cd")}
BAND_SUCC, BAND_NEAR, BAND_FAR = "#eef8f3", "#fdf3ee", "#f4f3f0"
W = 6.5  # TMLR text width in inches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.6,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})

PREVIEW = None


def load(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)


def style_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, length=3, width=0.6)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def panel_title(ax, text, pad=6):
    ax.set_title(text, loc="left", fontsize=8, color=INK, fontweight="bold", pad=pad)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name + ".pdf")
    fig.savefig(path, facecolor="white")
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=200, facecolor="white")
    plt.close(fig)
    return path


def bands(ax, xmax, label_y=None, near_label="near miss"):
    ax.axvspan(0.01, 1.0, color=BAND_SUCC, lw=0, zorder=0)
    ax.axvspan(1.0, 2.0, color=BAND_NEAR, lw=0, zorder=0)
    ax.axvspan(3.0, xmax, color=BAND_FAR, lw=0, zorder=0)
    if label_y is not None:
        ax.text(1.41, label_y, near_label, ha="center", va="bottom", fontsize=6.3, color=INK2)


def log_r_axis(ax, ticks, lo, hi):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(lo, hi)


# ---- diagram helpers (data units = inches on a full-figure axis)

def box(ax, x, y, w, h, text, tint="gray", fs=6.9, weight="normal", radius=0.06):
    fc, ec = TINT[tint]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec,
                                lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight=weight,
            linespacing=1.3, zorder=3)
    return (x, y, w, h)


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def arrow(ax, p0, p1, color=MUTED, lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, shrinkA=0,
                                 shrinkB=0, zorder=1, connectionstyle=f"arc3,rad={rad}"))


def line(ax, pts, color=MUTED, lw=1.0):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, zorder=1, solid_capstyle="butt")


# ---------------------------------------------------------------- Figure 1: protocol and the two error shapes
def fig1_protocol():
    fig = plt.figure(figsize=(W, 1.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, 1.95)
    ax.axis("off")
    ax.text(0.06, 1.82, "(a) Boundary-dense counterfactuals, judged twice", fontsize=8, fontweight="bold",
            color=INK, va="center")
    yt, yb, h = 1.05, 0.22, 0.55
    p1 = box(ax, 0.06, 0.635, 0.66, 0.55, "successful\nepisode")
    p2 = box(ax, 0.88, 0.635, 0.78, 0.55, "perturbed\nactions\n(severity s)", "orange")
    t1 = box(ax, 1.86, yt, 0.70, h, "simulator")
    t2 = box(ax, 2.72, yt, 0.92, h, "real final frame\ntrue r = e / τ")
    b1 = box(ax, 1.86, yb, 0.70, h, "world model\n(open loop)", "blue")
    b2 = box(ax, 2.72, yb, 0.92, h, "imagined\nfinal frame", "blue")
    j = box(ax, 3.82, 0.48, 0.62, 0.86, "same\njudge", "aqua", weight="bold")
    arrow(ax, right(p1), left(p2))
    xm = 1.74
    line(ax, [right(p2), (xm, right(p2)[1])])
    line(ax, [(xm, left(b1)[1]), (xm, left(t1)[1])])
    arrow(ax, (xm, left(t1)[1]), left(t1))
    arrow(ax, (xm, left(b1)[1]), left(b1))
    arrow(ax, right(t1), left(t2))
    arrow(ax, right(b1), left(b2))
    arrow(ax, right(t2), (j[0], yt + h / 2))
    arrow(ax, right(b2), (j[0], yb + h / 2))
    ax.text(1.0, 0.36, "same initial observation,\nsame actions", fontsize=6.2, color=INK2, ha="center", va="center",
            linespacing=1.3)
    ax.text(3.82 + 0.31, 0.3, "real vs. imagined\nverdict per r", fontsize=6.2, color=INK2, ha="center",
            va="center", linespacing=1.3)

    # (b) schematic: what a world-model evaluator can do to the outcome step
    bx = fig.add_axes([0.745, 0.24, 0.245, 0.6])
    style_axes(bx, grid_axis=None)
    r = np.exp(np.linspace(math.log(0.5), math.log(4.0), 400))
    bx.axvspan(0.5, 1.0, color=BAND_SUCC, lw=0, zorder=0)
    bx.axvspan(1.0, 2.0, color=BAND_NEAR, lw=0, zorder=0)
    bx.axvspan(3.0, 4.0, color=BAND_FAR, lw=0, zorder=0)
    bx.plot(r, (r < 1.0).astype(float), color=INK, lw=1.2, label="true outcome", zorder=3)
    bx.plot(r, 1 / (1 + np.exp(9.0 * (np.log(r) - math.log(1.45)))), color=BLUE, lw=1.8, label="optimism",
            zorder=4)
    bx.plot(r, 1 / (1 + np.exp(3.0 * (np.log(r) - math.log(0.97)))), color=ORANGE, lw=1.8, ls="--",
            label="blur in both directions", zorder=4)
    log_r_axis(bx, [0.5, 1, 2, 4], 0.5, 4.0)
    bx.set_ylim(-0.03, 1.62)
    bx.set_yticks([0, 0.5, 1])
    bx.set_xlabel("failure distance r", color=INK, labelpad=1)
    bx.set_ylabel("P(judged success)", color=INK, labelpad=1)
    bx.legend(loc="upper right", frameon=False, fontsize=5.9, handlelength=1.8, labelspacing=0.2,
              borderaxespad=0.0, bbox_to_anchor=(1.0, 1.0))
    fig.text(0.705, 0.93, "(b) Two error shapes", fontsize=8, fontweight="bold", color=INK, va="center")
    return save(fig, "fig1_protocol")


# ---------------------------------------------------------------- Figure 2: failure-distance curves, three models
def fig2_curves():
    m1 = load("E1_server/m1/e1_m1_analysis.json")["cells"]["val/H5/r"]["a"]["curve"]
    m2 = load("E1_m2/analysis/e1_analysis.json")
    pt = m2["cells"]["a|pooled6"]["point"]
    edges = m2["bins"]
    m3 = load("E1_server/m3/e1_m3_analysis.json")["pooled"]

    fig, axes = plt.subplots(1, 3, figsize=(W, 1.66), sharey=True, gridspec_kw={"wspace": 0.1})
    real_kw = dict(color=GRAY_DARK, ls="--", lw=1.2, marker="o", ms=3.2, mfc="white", mec=GRAY_DARK, mew=0.9,
                   zorder=4)
    # (a) M1
    ax = axes[0]
    style_axes(ax)
    bands(ax, 6.0, 1.04)
    xs = [c["r_med"] for c in m1]
    ax.plot(xs, [c["p_real"] for c in m1], label="real frame", **real_kw)
    ax.plot(xs, [c["p_imag"] for c in m1], color=BLUE, lw=1.7, marker="o", ms=3.4, mfc=BLUE, mec="white", mew=0.6,
            label="imagined, H = 5", zorder=5)
    log_r_axis(ax, [0.3, 0.5, 1, 2, 3, 5], 0.26, 6.0)
    ax.set_ylabel("P(judged success)", color=INK)
    ax.set_xlabel("failure distance r (log scale)", color=INK)
    panel_title(ax, "(a) M1: public DINO-WM, PushT", pad=12)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.94))
    # (b) M2 (binned; the outer bins are r < 0.8 and r >= 3)
    ax = axes[1]
    style_axes(ax)
    mids = [0.7] + [(lo + hi) / 2 for lo, hi in edges[1:-1]] + [3.3]
    ax.axvspan(0.55, 1.0, color=BAND_SUCC, lw=0, zorder=0)
    ax.axvspan(1.0, 2.0, color=BAND_NEAR, lw=0, zorder=0)
    ax.axvspan(3.0, 3.5, color=BAND_FAR, lw=0, zorder=0)
    ax.text(1.5, 1.04, "near miss", ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.plot(mids, pt["real"]["bins"], label="real frame", **real_kw)
    ax.plot(mids, pt["k1"]["bins"], color=BLUE, lw=1.7, marker="o", ms=3.4, mfc=BLUE, mec="white", mew=0.6,
            label="imagined, 1 chunk", zorder=5)
    ax.plot(mids, pt["k4"]["bins"], color=RAMP3[0], lw=1.4, ls="-.", marker="s", ms=3.0, mfc=RAMP3[0], mec="white",
            mew=0.5, label="imagined, 4 chunks", zorder=3)
    ax.set_xlim(0.55, 3.5)
    ax.set_xticks([0.7, 1, 1.5, 2, 2.5, 3, 3.3])
    ax.set_xticklabels(["<0.8", "1", "1.5", "2", "2.5", "3", "≥3"])
    ax.set_xlabel("failure distance r (bins)", color=INK)
    panel_title(ax, "(b) M2: Cosmos Policy, LIBERO", pad=12)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.94))
    # (c) M3
    ax = axes[2]
    style_axes(ax)
    bands(ax, 3.1, 1.04, near_label="near miss (training failures)")
    ax.plot(m3["0.03"]["curve"]["r"], m3["0.03"]["curve"]["real"], color=GRAY_DARK, ls="--", lw=1.2, label="real",
            zorder=4)
    for p, col, ls in zip(("0.3", "0.1", "0.03"), RAMP3, (":", "-.", "-")):
        cv = m3[p]["curve"]
        ax.plot(cv["r"], cv["imag"], color=col, lw=1.8, ls=ls, label=f"p = {p}", zorder=5)
    log_r_axis(ax, [0.6, 0.8, 1, 1.5, 2, 3], 0.6, 3.1)
    ax.set_xlabel("failure distance r (log scale)", color=INK)
    panel_title(ax, "(c) M3: by failure rate p", pad=12)
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(-0.03, 0.0), fontsize=5.5, handlelength=1.3,
              handletextpad=0.3, labelspacing=0.15, borderaxespad=0.2)
    for a in axes:
        a.set_ylim(-0.03, 1.1)
        a.set_yticks([0, 0.25, 0.5, 0.75, 1])
        a.title.set_fontsize(7.5)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.86)
    return save(fig, "fig2_curves")


# ---------------------------------------------------------------- Figure 3: resolution vs horizon on both public tasks, two-sided shift
def fig3_m1_horizon():
    e1 = load("E1_server/m1/e1_m1_analysis.json")["cells"]
    e4 = load("E4_server/e4_analysis.json")["4b_M1"]
    e7 = load("E7_runpod/pm/e7_analysis.json")["cells"]
    hs = [3, 5, 7, 10]
    c = {h: e1[f"val/H{h}/r"]["a"] for h in hs}
    pm = {h: e7[f"val/H{h}/r"]["a"] for h in hs}
    fig, (ax, px, bx) = plt.subplots(1, 3, figsize=(W, 1.60), gridspec_kw={"wspace": 0.32})

    def resolution(a, cells, probe, title, legend):
        style_axes(a)
        series = [
            ([cells[h]["auc_sf_imag"] for h in hs], ORANGE, "-", "D", True, "far, imagined"),
            ([cells[h]["auc_sn_real"] for h in hs], BLUE, "--", "o", False, "near, real frame"),
            ([probe[h] for h in hs], AQUA, ":", "s", True, "near, probe"),
            ([cells[h]["auc_sn_imag"] for h in hs], BLUE, "-", "o", True, "near, imagined"),
        ]
        for ys, col, ls, mk, filled, lab in series:
            a.plot(hs, ys, color=col, ls=ls, lw=1.5, marker=mk, ms=4.0, mfc=(col if filled else "white"), mec=col,
                   mew=1.0, label=lab, zorder=3)
        a.axvline(5, color=GRAY, lw=0.9, ls=(0, (4, 2)), zorder=1)  # the primary cell (named in the caption)
        a.set_xticks(hs)
        a.set_xlim(2.5, 10.5)
        a.set_ylim(0.55, 1.01)
        a.set_xlabel("horizon H (imagination steps)", color=INK)
        panel_title(a, title)
        if legend:
            leg = a.legend(loc="lower left", frameon=False, fontsize=6.0, handlelength=1.8, labelspacing=0.2,
                           borderaxespad=0.1, title="success vs.", title_fontsize=6.0)
            leg._legend_box.align = "left"

    resolution(ax, c, {h: e4[f"val_H{h}_r"]["imag_ceiling"] for h in hs}, "(a) PushT", True)
    ax.set_ylabel("AUROC of success", color=INK)
    resolution(px, pm, {h: pm[h]["auc_sn_ceiling"] for h in hs}, "(b) PointMaze", False)

    style_axes(bx)
    for key, col, lab in (("p_success", GRAY_DARK, "success"), ("p_near", BLUE, "near miss")):
        bx.plot(hs, [c[h][key + "_real"] for h in hs], color=col, ls="--", lw=1.3, marker="o", ms=4, mfc="white",
                mec=col, mew=1.0, label=f"{lab}, real frame", zorder=3)
        bx.plot(hs, [c[h][key + "_imag"] for h in hs], color=col, ls="-", lw=1.7, marker="o", ms=4, mfc=col,
                mec="white", mew=0.6, label=f"{lab}, imagined", zorder=4)
    bx.set_xticks(hs)
    bx.set_xlim(2.5, 10.5)
    bx.set_ylim(0, 1.0)
    bx.set_xlabel("horizon H (imagination steps)", color=INK)
    bx.set_ylabel("P(judged success)", color=INK)
    panel_title(bx, "(c) PushT, both directions")
    bx.legend(loc="lower left", frameon=False, fontsize=6.0, labelspacing=0.2, handlelength=1.8,
              bbox_to_anchor=(0.0, 0.2))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.89)
    return save(fig, "fig3_m1_horizon")


# ---------------------------------------------------------------- Figure 4: judge vs imagination (2x2 decomposition)
def fig4_decomp():
    e3 = load("E3_server/m1/e3_m1_analysis.json")
    h3m2 = load("E1_m2/analysis/h3_m2.json")["paired"]
    m2d = load("E1_m2/analysis/e1_analysis.json")["cells"]["d|pooled6"]["point"]

    def cell(real, imag):
        # no share is printed for a row where imagination lowers the rate, so none is computed here
        return {"p_near_real": real, "p_near_imag": imag, "share": None}

    groups = [
        ("Reference judge (a), primary cells", [
            ("M1  PushT, r, H = 5", e3["e1_cells"]["val_H5_r"]["a"]),
            ("M2  LIBERO, 1 chunk†", {"p_near_real": h3m2["p_real_near"], "p_near_imag": h3m2["p_imag_near"],
                                         "share": h3m2["share"]}),
            ("M3  p = 0.03", e3["m3"]["0.03"]),
        ]),
        ("Reference judge (a), auxiliary cells", [
            ("M1  object pose r_obj", e3["e1_cells"]["val_H5_r_obj"]["a"]),
            ("M1  policy episodes", e3["dp"]["H5"]["a_std"]),
        ]),
        ("VLM judge (d)", [
            ("M1  PushT, r, H = 5", e3["e1_cells"]["val_H5_r"]["d"]),
            ("M1  policy episodes", e3["dp"]["H5"]["d"]),
            ("M2  LIBERO, 1 chunk", cell(m2d["real"]["near"], m2d["k1"]["near"])),
        ]),
    ]
    rows, ytick, labels = [], [], []
    y = 0.0
    heads = []
    for gname, cells in groups:
        heads.append((y, gname))
        y -= 0.62
        for lab, c in cells:
            rows.append((y, c))
            ytick.append(y)
            labels.append(lab)
            y -= 0.62
        y -= 0.2
    fig, ax = plt.subplots(figsize=(W, 1.85))
    style_axes(ax, grid_axis="x")
    gap = 0.004
    below_shown = False
    for i, (yy, c) in enumerate(rows):
        floor, imag = c["p_near_real"], c["p_near_imag"]
        ax.barh(yy, floor, height=0.42, color=GRAY_LIGHT, zorder=3,
                label="real frame (judge's own error)" if i == 0 else None)
        if imag > floor:
            ax.barh(yy, imag - floor - gap, left=floor + gap, height=0.42, color=BLUE, zorder=3,
                    label="added by imagination" if i == 0 else None)
            text = f"{floor:.3f} → {imag:.3f}   share {c['share']:.2f}"
        else:
            ax.plot(imag, yy, "D", ms=4.4, mfc="white", mec=INK2, mew=1.0, zorder=4,
                    label=None if below_shown else "imagined, below real")
            below_shown = True
            text = f"{floor:.3f} → {imag:.3f}"
        ax.text(max(imag, floor) + 0.012, yy, text, va="center", fontsize=6.4, color=INK)
    for yy, g in heads:
        ax.text(-0.3, yy, g, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.6,
                color=INK, fontweight="bold")
    ax.set_yticks(ytick)
    ax.set_yticklabels(labels, fontsize=6.6, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(y + 0.3, 0.35)
    ax.set_xlabel("near misses judged successful", color=INK)
    handles, labs = ax.get_legend_handles_labels()
    order = [labs.index(k) for k in ("real frame (judge's own error)", "added by imagination",
                                       "imagined, below real")]
    handles, labs = [handles[i] for i in order], [labs[i] for i in order]
    ax.legend(handles, labs, loc="center right", frameon=False, fontsize=6.3, bbox_to_anchor=(1.0, 0.47))
    fig.subplots_adjust(left=0.245, right=0.99, bottom=0.15, top=0.98)
    return save(fig, "fig4_decomp")


# ---------------------------------------------------------------- Figure 5: M3 dose-response in the training failure rate
def fig5_m3_dose():
    pooled = load("E1_server/m3/e1_m3_analysis.json")["pooled"]
    e4 = load("E4_server/e4_analysis.json")["4a_M3"]
    ps = ["0.3", "0.1", "0.03"]
    fig = plt.figure(figsize=(W, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    bands(ax, 3.1, 1.04, near_label="near miss")
    ax.text(2.45, 1.04, "extrapolated", ha="center", va="bottom", fontsize=6.3, color=INK2)
    ax.plot(pooled["0.03"]["curve"]["r"], pooled["0.03"]["curve"]["real"], color=GRAY_DARK, ls="--", lw=1.2,
            label="real frame", zorder=4)
    for p, col, ls in zip(ps, RAMP3, (":", "-.", "-")):
        cv = pooled[p]["curve"]
        ax.plot(cv["r"], cv["imag"], color=col, lw=1.8, ls=ls, label=f"imagined, p = {p}", zorder=3)
    log_r_axis(ax, [0.6, 0.8, 1, 1.5, 2, 3], 0.6, 3.1)
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("failure distance r (log scale)", color=INK)
    ax.set_ylabel("P(judged success)", color=INK)
    panel_title(ax, "(a) Imagined verdict, by p", pad=12)
    ax.legend(loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.27), ncol=2, labelspacing=0.25,
              fontsize=6.3, handlelength=2.2, columnspacing=1.2)
    nm = [e4[f"p{p}"]["NM"] for p in ps]
    sm = [e4[f"p{p}"]["SM"] for p in ps]
    for k, (vals, title, ymax, fmt) in enumerate(((nm, "(b) Near miss → success", 0.62, "{:.3f}"),
                                                   (sm, "(c) Success → failure", 0.026, "{:.4f}"))):
        bx = fig.add_subplot(gs[k + 1])
        style_axes(bx)
        xs = np.arange(3)
        bx.bar(xs, vals, width=0.5, color=RAMP3, zorder=3)
        for x, v in zip(xs, vals):
            bx.text(x, v + ymax * 0.025, "0" if v == 0 else fmt.format(v), ha="center", va="bottom", fontsize=6.3,
                    color=INK)
        bx.set_xticks(xs)
        bx.set_xticklabels(["0.3", "0.1", "0.03"])
        bx.set_xlabel("training failure rate p", color=INK)
        bx.set_ylim(0, ymax)
        bx.tick_params(axis="x", length=0)
        bx.set_title(title, loc="right", fontsize=7.3, color=INK, fontweight="bold", pad=12)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.33, top=0.88)
    return save(fig, "fig5_m3_dose")


# ---------------------------------------------------------------- Figure 6: policy level
def fig6_policy():
    m3 = load("E2_server/m3/e2_m3_analysis.json")["0.03"]["r29"]["table"]
    m1 = load("E2_server/m1/e2_m1_analysis.json")["dp"]["H5"]
    names = {"pi1_precise": "π1 precise", "pi2_good": "π2 good", "pi3_medium": "π3 medium",
             "pi4_sloppy": "π4 sloppy", "pi5_far_failer": "π5 far-failer",
             "pi6_near_failer": "π6 near-failer"}
    colors = {"pi5_far_failer": ORANGE, "pi6_near_failer": BLUE}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 1.62), gridspec_kw={"wspace": 0.42, "width_ratios": [1.3, 1]})
    style_axes(ax, grid_axis="x")
    for i, row in enumerate(m3):
        y = len(m3) - 1 - i
        col = colors.get(row["policy"], GRAY_DARK)
        ax.plot([row["s"], row["sh"]], [y, y], color=GRAY_LIGHT, lw=2.0, zorder=2, solid_capstyle="round")
        ax.plot(row["s"], y, "o", ms=5.5, mfc="white", mec=col, mew=1.3, zorder=4)
        ax.plot(row["sh"], y, "o", ms=5.5, mfc=col, mec="white", mew=0.9, zorder=5)
        ax.text(1.085, y, f"{row['n']:.3f}", va="center", ha="right", fontsize=6.4, color=INK2)
    ax.text(1.085, len(m3) - 0.4, "near-miss share", va="bottom", ha="right", fontsize=6.2, color=INK2)
    ax.set_yticks(range(len(m3)))
    ax.set_yticklabels([names[r["policy"]] for r in m3][::-1], fontsize=6.8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.42, 1.09)
    ax.set_ylim(-0.6, len(m3) - 0.05)
    ax.set_xlabel("success rate", color=INK)
    ax.plot([], [], "o", ms=5.5, mfc="white", mec=GRAY_DARK, mew=1.3, label="real")
    ax.plot([], [], "o", ms=5.5, mfc=GRAY_DARK, mec="white", label="imagined")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.03), ncol=2, handletextpad=0.2,
              columnspacing=0.8)
    panel_title(ax, "(a) M3: six policies, p = 0.03", pad=8)

    style_axes(bx)
    tab = m1["table"]
    bx.axhline(0, color=AXIS, lw=0.8, zorder=1)
    bx.plot([r["n"] for r in tab], [r["o_imag"] for r in tab], "o", ms=4.6, mfc=BLUE, mec="white", mew=0.7, zorder=3)
    rho = m1["point"]["spearman_n_oimag"]
    bx.text(0.03, 0.95, f"Spearman {rho:.3f}", transform=bx.transAxes, fontsize=6.6, color=INK2, va="top")
    bx.set_xlim(-0.01, 0.31)
    bx.set_ylim(-0.03, 0.09)
    bx.set_xlabel("near-miss share $n_\\pi$", color=INK)
    bx.set_ylabel("overestimation $o_\\pi^{\\rm imag}$", color=INK)
    panel_title(bx, "(b) M1: 24 policy checkpoints", pad=8)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.2, top=0.9)
    return save(fig, "fig6_policy")


# ---------------------------------------------------------------- Appendix figure: M2 real vs imagined frames
def figA_m2_frames():
    im = Image.open(os.path.join(RES, "E1_m2/analysis/e1_frames_real_vs_imag_k1.png")).convert("RGB")
    cw, rh = im.size[0] // 4, im.size[1] // 12
    cut = 16  # drop the caption strip burned into the top of each source row
    # Rows 4 and 5 are task t4 (bowl on top of the cabinet). The burned-in captions give the judge (a) logits of the
    # re-rendered frames and of the E1 record ("job"); both agree in sign for these rows: real 9.2 / imagined -1.9
    # (job -1.9) for the success, real -0.9 / imagined 2.2 (job 2.2) for the near miss.
    rows = [(4, "success, r = 0.40"), (5, "near miss, r = 1.04")]
    order = [0, 2, 1, 3]  # source columns: 0 real 3rd-person, 1 imagined 3rd-person, 2 real wrist, 3 imagined wrist
    heads = ["real, third-person", "real, wrist", "imagined, third-person", "imagined, wrist"]
    verdict = {(4, 0): "judge (a): success (logit 9.2)", (4, 2): "judge (a): failure (logit −1.9)",
               (5, 0): "judge (a): failure (logit −0.9)", (5, 2): "judge (a): success (logit 2.2)"}
    fig = plt.figure(figsize=(W * 0.82, 2.55))
    sub = fig.add_gridspec(2, 4, wspace=0.05, hspace=0.42)
    for ri, (row, lab) in enumerate(rows):
        for ci, src in enumerate(order):
            a = fig.add_subplot(sub[ri, ci])
            a.imshow(im.crop((src * cw, row * rh + cut, (src + 1) * cw, (row + 1) * rh)))
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color(AXIS)
                s.set_linewidth(0.6)
            if ri == 0:
                a.set_title(heads[ci], fontsize=6.8, color=INK, pad=3)
            if ci == 0:
                a.set_ylabel(lab, fontsize=6.6, color=INK, labelpad=3)
            if ci in (0, 2):
                a.text(1.03, -0.1, verdict[(row, ci)], transform=a.transAxes, ha="center", va="top", fontsize=6.3,
                       color=INK2)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.09, top=0.93)
    return save(fig, "figA_m2_frames")


FIGS = [fig1_protocol, fig2_curves, fig3_m1_horizon, fig4_decomp, fig6_policy, figA_m2_frames]


def main():
    global PREVIEW
    if "--preview" in sys.argv:
        PREVIEW = sys.argv[sys.argv.index("--preview") + 1]
    for f in FIGS:
        print("wrote", f())


if __name__ == "__main__":
    main()
