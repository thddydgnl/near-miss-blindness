"""Shared paths, fonts and drawing helpers for the B explainer document (연구설명).

Everything here is presentation only: values are read from the frozen result JSON in main/results.
"""
import json
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
EXPL = os.path.normpath(os.path.join(HERE, ".."))
B_ROOT = os.path.normpath(os.path.join(EXPL, "..", ".."))
RES = os.path.join(B_ROOT, "main", "results")
FIG = os.path.join(EXPL, "figures")
OUT = os.path.join(EXPL, "out")

FONT_TTC = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_FAMILY = "Apple SD Gothic Neo"
# face index in the system collection -> style name (odd indices are the hidden ".Apple SD Gothic NeoI" faces)
FONT_FACES = {0: "Regular", 2: "Medium", 4: "SemiBold", 6: "Bold", 8: "Light"}

# Palette: dataviz reference categorical slots 1-3, the ordinal blue ramp (steps 250/450/650) and chart ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP3 = ("#86b6ef", "#2a78d6", "#104281")
GRAY, GRAY_LIGHT, GRAY_DARK = "#a8a69f", "#cfcdc6", "#7d7b75"
INK, INK2, MUTED = "#2b2a27", "#52514e", "#898781"
GRID, AXIS = "#e6e5df", "#c3c2b7"
# soft box tints for diagrams: (fill, edge)
TINT = {
    "gray": ("#f4f3f0", "#d6d4cd"),
    "blue": ("#edf4fd", "#b7d3f6"),
    "orange": ("#fdf1eb", "#f3c4ad"),
    "aqua": ("#e8f7f0", "#a9dfc9"),
    "yellow": ("#fdf6e3", "#ecd28f"),
    "white": ("#ffffff", "#d6d4cd"),
}
BAND_NEAR, BAND_SUCC, BAND_FAR = "#fdf3ee", "#eef8f3", "#f4f3f0"


def load(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)


def register_fonts():
    """Extract the Apple SD Gothic Neo faces from the system TTC once and register them with matplotlib."""
    from fontTools.ttLib import TTCollection

    cache = os.environ.get("B_EXPLAINER_FONT_CACHE") or os.path.join(tempfile.gettempdir(), "b_explainer_fonts")
    os.makedirs(cache, exist_ok=True)
    coll = None
    for idx, style in FONT_FACES.items():
        path = os.path.join(cache, f"AppleSDGothicNeo-{style}.ttf")
        if not os.path.exists(path):
            coll = coll or TTCollection(FONT_TTC)
            coll.fonts[idx].save(path)
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": [FONT_FAMILY, "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.unicode_minus": False,  # U+2212 is missing from the font; a hyphen keeps one typeface
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
    })


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
    ax.set_title(text, loc="left", fontsize=8.5, color=INK, fontweight="semibold", pad=pad)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


# ---- diagram helpers (data units = inches on a full-figure axis) ----

def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, tint="gray", fs=7.6, weight="regular", color=INK, radius=0.07, lw=0.9):
    fc, ec = TINT[tint]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight=weight, linespacing=1.35, zorder=3)
    return (x, y, w, h)


def pill_width(text, fs=6.4, pad=0.09):
    """Estimated width in inches of a status label: Hangul is about twice as wide as Latin at the same size."""
    n_wide = sum(1 for ch in text if ord(ch) > 0x2E80)
    n_narrow = len(text) - n_wide
    return (n_wide * 0.092 + n_narrow * 0.052) * fs / 6.4 + 2 * pad


def pill(ax, cx, cy, text, tint="aqua", fs=6.4, h=0.2, pad=0.09):
    """A small status label centred at (cx, cy)."""
    w = pill_width(text, fs, pad)
    fc, ec = TINT[tint]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=fc, ec=ec, lw=0.7, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, zorder=5)
    return w


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def arrow(ax, p0, p1, color=MUTED, lw=1.0, ls="-", rad=0.0, ms=8, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def line(ax, pts, color=MUTED, lw=1.0, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
