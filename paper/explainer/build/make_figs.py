"""Figures for the B explainer document (연구설명), Korean labels in Apple SD Gothic Neo.

Diagrams are drawn by hand; every data figure reads the frozen analysis JSON in main/results (no new computation).
Run:  python3 paper/explainer/build/make_figs.py
Outputs: paper/explainer/figures/fig01_*.png ... fig12_*.png
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import (AQUA, AXIS, BAND_FAR, BAND_NEAR, BAND_SUCC, BLUE, GRAY, GRAY_DARK, GRAY_LIGHT, GRID, INK,
                    INK2, MUTED, ORANGE, RAMP3, RES, arrow, bottom, box, canvas, left, line, load, panel_title,
                    pill, pill_width, register_fonts, right, save, style_axes, top)

W = 6.6  # full text width in inches


# ---------------------------------------------------------------- 1. two kinds of world model
def fig01_world_models():
    fig, ax = canvas(W, 3.0)
    ax.text(0.1, 2.83, "① latent 월드모델", fontsize=8.6, fontweight="semibold", color=INK, va="center")
    ax.text(1.33, 2.83, "특징 공간에서 다음 상태를 예측한다  ·  예: DINO-WM (이 연구의 M1, M3)",
            fontsize=7.2, color=INK2, va="center")
    y, h = 1.92, 0.62
    a1 = box(ax, 0.10, y, 0.95, h, "관측 프레임")
    a2 = box(ax, 1.33, y, 1.02, h, "고정 인코더\nDINOv2")
    a3 = box(ax, 2.63, y, 1.08, h, "patch 특징\n(현재 상태)", "blue")
    a4 = box(ax, 3.99, y, 1.08, h, "예측기 (ViT)\n+ 행동", "blue")
    a5 = box(ax, 5.35, y, 1.15, h, "다음 patch 특징\n(예측한 상태)", "blue")
    for p, q in ((a1, a2), (a2, a3), (a3, a4), (a4, a5)):
        arrow(ax, right(p), left(q))
    arrow(ax, (5.92, y), (4.53, y), rad=-0.42, color=BLUE, lw=1.0)
    ax.text(5.23, 1.5, "예측을 다시 입력으로 넣어 반복한다 = 상상", fontsize=6.9, color=INK2, ha="center")
    ax.text(0.10, 1.62, "픽셀을 만들지 않는다. 계획과 판정도\n특징 위에서 한다.", fontsize=6.9, color=INK2,
            va="center", linespacing=1.4)
    line(ax, [(0.1, 1.28), (6.5, 1.28)], color="#e3e1db", lw=0.8)

    ax.text(0.1, 1.08, "② 비디오 월드모델", fontsize=8.6, fontweight="semibold", color=INK, va="center")
    ax.text(1.38, 1.08, "미래 장면을 영상으로 생성한다  ·  예: Cosmos Policy (이 연구의 M2)",
            fontsize=7.2, color=INK2, va="center")
    b1 = box(ax, 0.10, 0.18, 1.22, 0.62, "카메라 영상\n+ 과제 지시문")
    b2 = box(ax, 1.62, 0.18, 1.62, 0.62, "비디오 확산 모델\nCosmos-Predict2 2B", "orange")
    arrow(ax, right(b1), left(b2))
    outs = [("행동 청크 (16 step)", 0.63), ("행동 뒤의 미래 영상", 0.36), ("가치 V (성공 기대)", 0.09)]
    for text, yy in outs:
        b = box(ax, 3.62, yy, 1.62, 0.24, text, "orange", fs=7.2, radius=0.05)
        arrow(ax, right(b2), left(b))
    ax.text(5.42, 0.36, "세 출력을\n한 번에 낸다", fontsize=6.9, color=INK2, va="center", linespacing=1.4)
    return save(fig, "fig01_world_models.png")


# ---------------------------------------------------------------- 2. real vs imagined evaluation
def fig02_policy_eval():
    fig, ax = canvas(W, 2.3)
    s = box(ax, 0.08, 0.85, 0.92, 0.6, "같은\n초기 상태")
    yt, yb, h = 1.48, 0.22, 0.56
    ax.text(1.27, yt + h + 0.12, "실제 평가", fontsize=7.6, fontweight="semibold", color=INK, va="center")
    ax.text(1.27, yb + h + 0.12, "상상 속 평가", fontsize=7.6, fontweight="semibold", color=INK, va="center")
    t1 = box(ax, 1.27, yt, 0.82, h, "정책 π")
    t2 = box(ax, 2.33, yt, 1.08, h, "실제 환경\n(시뮬레이터)")
    t3 = box(ax, 3.65, yt, 0.98, h, "실제\n최종 프레임")
    t4 = box(ax, 4.87, yt, 0.66, h, "판정기", "aqua")
    t5 = box(ax, 5.77, yt, 0.75, h, "실제\n성공률 s", "white")
    b1 = box(ax, 1.27, yb, 0.82, h, "정책 π의\n행동")
    b2 = box(ax, 2.33, yb, 1.08, h, "월드모델\n상상", "blue")
    b3 = box(ax, 3.65, yb, 0.98, h, "상상\n최종 프레임", "blue")
    b4 = box(ax, 4.87, yb, 0.66, h, "같은\n판정기", "aqua")
    b5 = box(ax, 5.77, yb, 0.75, h, "상상\n성공률 ŝ", "white")
    for p, q in ((t1, t2), (t2, t3), (t3, t4), (t4, t5), (b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        arrow(ax, right(p), left(q))
    line(ax, [right(s), (1.13, right(s)[1])])
    line(ax, [(1.13, left(b1)[1]), (1.13, left(t1)[1])])
    arrow(ax, (1.13, left(t1)[1]), left(t1))
    arrow(ax, (1.13, left(b1)[1]), left(b1))
    arrow(ax, (6.145, yt), (6.145, yb + h), style="<|-|>", ms=6, color=INK2, lw=0.9)
    ax.text(6.06, (yt + yb + h) / 2, "차이 =\n과대 추정 o", fontsize=6.9, color=INK2, va="center", ha="right",
            linespacing=1.35)
    return save(fig, "fig02_policy_eval.png")


# ---------------------------------------------------------------- 3. two ways to break the boundary (concept)
def fig03_concept():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"wspace": 0.22})
    r = np.exp(np.linspace(math.log(0.5), math.log(4.0), 400))
    real = (r < 1.0).astype(float)
    curves = [
        ("(a) 낙관: 경계가 바깥으로 밀린다", 1 / (1 + np.exp(9.0 * (np.log(r) - math.log(1.45))))),
        ("(b) 양방향 흐림: 계단이 비탈이 된다", 1 / (1 + np.exp(3.2 * (np.log(r) - math.log(0.98))))),
    ]
    for ax, (title, imag) in zip(axes, curves):
        style_axes(ax, grid_axis=None)
        ax.axvspan(0.5, 1.0, color=BAND_SUCC, lw=0, zorder=0)
        ax.axvspan(1.0, 2.0, color=BAND_NEAR, lw=0, zorder=0)
        ax.axvspan(3.0, 4.0, color=BAND_FAR, lw=0, zorder=0)
        for xm, lab in ((0.71, "성공"), (1.41, "near-miss"), (2.45, "mid"), (3.45, "far-miss")):
            ax.text(xm, 1.08, lab, ha="center", va="bottom", fontsize=6.8, color=INK2)
        ax.plot(r, real, color=INK, lw=1.4, label="실제 결과", zorder=3)
        ax.plot(r, imag, color=BLUE, lw=2.0, label="상상 판정", zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 3, 4])
        ax.set_xticklabels(["0.5", "1", "2", "3", "4"])
        ax.minorticks_off()
        ax.set_xlim(0.5, 4.0)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("실패 거리 r (로그 눈금)", color=INK)
        panel_title(ax, title, pad=16)
    axes[0].set_ylabel("성공 판정 확률", color=INK)
    axes[0].annotate("", xy=(1.45, 0.5), xytext=(1.0, 0.5),
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))
    axes[0].text(1.16, 0.57, "경계 이동", ha="center", fontsize=6.8, color=INK2)
    axes[1].annotate("성공 일부가\n실패로", xy=(0.82, 0.74), xytext=(0.53, 0.42), fontsize=6.8, color=INK2,
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))
    axes[1].annotate("near-miss 일부가\n성공으로", xy=(1.35, 0.27), xytext=(1.75, 0.55), fontsize=6.8,
                     color=INK2, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7))
    axes[1].legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.98))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.82)
    return save(fig, "fig03_concept.png")


# ---------------------------------------------------------------- 4. measurement pipeline
def fig04_pipeline():
    fig, ax = canvas(W, 2.45)
    yt, yb, h = 1.62, 0.42, 0.62
    p1 = box(ax, 0.08, yt, 0.86, h, "① 성공한\n궤적")
    p2 = box(ax, 1.18, yt, 1.04, h, "② 행동 교란\n(세기 조절)", "orange")
    p3 = box(ax, 2.50, yt, 1.0, h, "③ 시뮬레이터\n실행")
    p4 = box(ax, 3.78, yt, 1.3, h, "실제 최종 프레임\n+ 참값 r = e / τ")
    q3 = box(ax, 2.50, yb, 1.0, h, "④ 같은 행동을\n월드모델로 재생", "blue")
    q4 = box(ax, 3.78, yb, 1.3, h, "상상\n최종 프레임", "blue")
    j = box(ax, 5.36, yt, 1.16, h, "⑤ 같은 판정기", "aqua", weight="semibold")
    o = box(ax, 5.36, 0.22, 1.16, 1.02, "⑥ 집계\n성공 판정 곡선\nAUROC\n상상 몫", "white", fs=7.2)
    for p, q in ((p1, p2), (p2, p3), (p3, p4), (q3, q4), (p4, j)):
        arrow(ax, right(p), left(q))
    xm = bottom(p2)[0]
    line(ax, [bottom(p2), (xm, left(q3)[1])])
    arrow(ax, (xm, left(q3)[1]), left(q3))
    ax.text(xm + 0.08, 1.05, "같은 초기 관측\n같은 행동열", fontsize=6.6, color=INK2, va="center", linespacing=1.35)
    arrow(ax, right(q4), (5.36, yt + 0.12))
    arrow(ax, bottom(j), top(o))
    ax.text(5.22, yt + h / 2 + 0.07, "실제", fontsize=6.4, color=INK2, ha="center", va="bottom")
    ax.text(5.08, 1.2, "상상", fontsize=6.4, color=INK2, ha="right", va="center")
    return save(fig, "fig04_pipeline.png")


# ---------------------------------------------------------------- 5. M1 boundary resolution
def fig05_m1():
    e1 = load("E1_server/m1/e1_m1_analysis.json")["cells"]
    e4 = load("E4_server/e4_analysis.json")["4b_M1"]
    hs = [3, 5, 7, 10]
    real_sn = [e1[f"val/H{h}/r"]["a"]["auc_sn_real"] for h in hs]
    imag_sn = [e1[f"val/H{h}/r"]["a"]["auc_sn_imag"] for h in hs]
    imag_sf = [e1[f"val/H{h}/r"]["a"]["auc_sf_imag"] for h in hs]
    ceil_sn = [e4[f"val_H{h}_r"]["imag_ceiling"] for h in hs]
    curve = e1["val/H5/r"]["a"]["curve"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"wspace": 0.3, "width_ratios": [1.05, 1]})
    style_axes(ax)
    series = [
        (imag_sf, ORANGE, "-", "D", True, "far-miss, 상상"),
        (real_sn, BLUE, "--", "o", False, "near-miss, 실제 프레임"),
        (ceil_sn, AQUA, ":", "s", True, "near-miss, 상상 특징 판별기"),
        (imag_sn, BLUE, "-", "o", True, "near-miss, 상상"),
    ]
    for ys, c, ls, mk, filled, lab in series:
        ax.plot(hs, ys, color=c, ls=ls, lw=1.6, marker=mk, ms=4.6, mfc=(c if filled else "white"), mec=c,
                mew=1.1, label=lab, zorder=3)
    ax.axvline(5, ymin=0.6, ymax=1.0, color=GRAY_DARK, lw=0.8, zorder=1)
    ax.set_xticks(hs)
    ax.set_xlim(2.5, 10.5)
    ax.set_ylim(0.64, 1.01)
    ax.set_xlabel("상상 길이 H (계획 단계)", color=INK)
    ax.set_ylabel("AUROC", color=INK)
    panel_title(ax, "(a) 성공 대 실패를 가르는 AUROC", pad=14)
    leg = ax.legend(loc="lower left", frameon=False, title="성공 대 …", title_fontsize=6.8, fontsize=6.6,
                    handlelength=2.4, labelspacing=0.3, borderaxespad=0.1)
    leg._legend_box.align = "left"

    style_axes(bx)
    bx.axvspan(1.0, 2.0, color=BAND_NEAR, lw=0, zorder=0)
    bx.text(1.41, 1.04, "near-miss", ha="center", fontsize=6.8, color=INK2)
    xs = [c["r_med"] for c in curve]
    bx.plot(xs, [c["p_real"] for c in curve], color=INK2, ls="--", lw=1.3, marker="o", ms=4, mfc="white",
            mec=INK2, mew=1.0, label="실제 프레임", zorder=3)
    bx.plot(xs, [c["p_imag"] for c in curve], color=BLUE, lw=1.8, marker="o", ms=4.2, mfc=BLUE, mec=BLUE,
            label="상상 (H = 5)", zorder=4)
    bx.set_xscale("log")
    bx.set_xticks([0.3, 0.5, 1, 2, 3, 5])
    bx.set_xticklabels(["0.3", "0.5", "1", "2", "3", "5"])
    bx.minorticks_off()
    bx.set_xlim(0.26, 5.6)
    bx.set_ylim(-0.03, 1.08)
    bx.set_xlabel("실패 거리 r (로그 눈금)", color=INK)
    bx.set_ylabel("성공 판정 확률", color=INK)
    panel_title(bx, "(b) 실패 거리별 성공 판정 (H = 5)", pad=14)
    bx.legend(loc="upper right", frameon=False, fontsize=6.8)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.87)
    return save(fig, "fig05_m1_boundary.png")


# ---------------------------------------------------------------- 6. M3 dose-response
def fig06_m3():
    pooled = load("E1_server/m3/e1_m3_analysis.json")["pooled"]
    e4 = load("E4_server/e4_analysis.json")["4a_M3"]
    ps = ["0.3", "0.1", "0.03"]
    fig = plt.figure(figsize=(W, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.15, 1, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    ax.axvspan(1.0, 2.0, color=BAND_NEAR, lw=0, zorder=0)
    ax.text(1.41, 1.05, "near-miss (학습 실패 구간)", ha="center", fontsize=6.6, color=INK2)
    ax.text(2.45, 1.05, "외삽", ha="center", fontsize=6.6, color=INK2)
    real = pooled["0.03"]["curve"]
    ax.plot(real["r"], real["real"], color=INK2, ls="--", lw=1.2, label="실제 프레임 (모든 p)", zorder=4)
    for p, c in zip(ps, RAMP3):
        cv = pooled[p]["curve"]
        ax.plot(cv["r"], cv["imag"], color=c, lw=1.9, label=f"상상, p = {p}", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1, 1.5, 2, 3])
    ax.set_xticklabels(["0.6", "0.8", "1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_xlim(0.6, 3.05)
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("실패 거리 r (로그 눈금)", color=INK)
    ax.set_ylabel("성공 판정 확률", color=INK)
    panel_title(ax, "(a) 학습 실패율 p별 성공 판정", pad=14)
    ax.legend(loc="center right", frameon=False, fontsize=6.6, bbox_to_anchor=(1.0, 0.38), labelspacing=0.3)

    nm = [e4[f"p{p}"]["NM"] for p in ps]
    sm = [e4[f"p{p}"]["SM"] for p in ps]
    for k, (vals, title, ymax, fmt) in enumerate(((nm, "(b) near-miss → 성공 (NM)", 0.62, "{:.2f}"),
                                                   (sm, "(c) 성공 → 실패 (SM)", 0.026, "{:.3f}"))):
        bx = fig.add_subplot(gs[k + 1])
        style_axes(bx)
        xs = np.arange(3)
        bx.bar(xs, vals, width=0.5, color=RAMP3, zorder=3)
        for x, v in zip(xs, vals):
            bx.text(x, v + ymax * 0.025, fmt.format(v), ha="center", va="bottom", fontsize=6.8, color=INK)
        bx.set_xticks(xs)
        bx.set_xticklabels(["0.3", "0.1", "0.03"])
        bx.set_xlabel("학습 실패율 p", color=INK)
        bx.set_ylim(0, ymax)
        bx.tick_params(axis="x", length=0)
        panel_title(bx, title, pad=14)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.86)
    return save(fig, "fig06_m3_dose.png")


# ---------------------------------------------------------------- 7. M2 curves + example frames
def fig07_m2():
    m2 = load("E1_m2/analysis/e1_analysis.json")
    pt = m2["cells"]["a|pooled6"]["point"]
    edges = m2["bins"]
    xs = [0.7] + [(lo + hi) / 2 for lo, hi in edges[1:-1]] + [3.3]

    fig = plt.figure(figsize=(W, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.32], wspace=0.16)
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    ax.axvspan(1.0, 2.0, color=BAND_NEAR, lw=0, zorder=0)
    ax.text(1.5, 1.05, "near-miss", ha="center", fontsize=6.6, color=INK2)
    ax.plot(xs, pt["real"]["bins"], color=INK2, ls="--", lw=1.2, marker="o", ms=3.6, mfc="white", mec=INK2,
            label="실제 프레임", zorder=4)
    ax.plot(xs, pt["k1"]["bins"], color=BLUE, lw=1.8, marker="o", ms=3.8, label="상상, 청크 1개", zorder=5)
    ax.plot(xs, pt["k4"]["bins"], color=RAMP3[0], lw=1.6, marker="o", ms=3.4, label="상상, 청크 4개", zorder=3)
    ax.set_xlim(0.55, 3.5)
    ax.set_xticks([0.7, 1, 1.5, 2, 2.5, 3, 3.3])
    ax.set_xticklabels(["<0.8", "1", "1.5", "2", "2.5", "3", "≥3"])
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlabel("실패 거리 r", color=INK)
    ax.set_ylabel("성공 판정 확률", color=INK)
    panel_title(ax, "(a) 실패 거리별 성공 판정 (판정기 (a))", pad=14)
    ax.legend(loc="upper right", frameon=False, fontsize=6.6, bbox_to_anchor=(1.0, 0.93), labelspacing=0.3)

    im = Image.open(f"{RES}/E1_m2/analysis/e1_frames_real_vs_imag_k1.png").convert("RGB")
    cw, rh = im.size[0] // 4, im.size[1] // 12
    cut = 16  # drop the red caption strip burned into the top of each source row
    rows = [(10, "성공 · r = 0.30"), (11, "near-miss · r = 1.07")]
    # source columns: 0 real third-person, 1 imagined third-person, 2 real wrist, 3 imagined wrist.
    # The judge reads both views together, so show each pair side by side: real pair, then imagined pair.
    order = [0, 2, 1, 3]
    heads = ["실제 · 3인칭", "실제 · 손목", "상상 · 3인칭", "상상 · 손목"]
    verdict = {(10, 0): "판정: 성공", (10, 2): "판정: 실패 (오판)", (11, 0): "판정: 실패", (11, 2): "판정: 실패"}
    sub = gs[1].subgridspec(2, 4, wspace=0.04, hspace=0.34)
    for ri, (row, lab) in enumerate(rows):
        for ci, src in enumerate(order):
            a = fig.add_subplot(sub[ri, ci])
            tile = im.crop((src * cw, row * rh + cut, (src + 1) * cw, (row + 1) * rh))
            a.imshow(tile)
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
                a.text(1.02, -0.12, verdict[(row, ci)], transform=a.transAxes, ha="center", va="top",
                       fontsize=6.4, color=INK2)
    fig.text(0.535, 0.94, "(b) 실제 대 상상 프레임 (t8 그릇 → 접시, 청크 1개, 들어 올린 뒤)", fontsize=8.5,
             color=INK, fontweight="semibold")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.86)
    return save(fig, "fig07_m2.png")


# ---------------------------------------------------------------- 8. judge vs imagination decomposition
def fig08_h3():
    e3 = load("E3_server/m1/e3_m1_analysis.json")
    h3m2 = load("E1_m2/analysis/h3_m2.json")["paired"]
    cells = [
        ("M1 · 원 정의 r (주 칸)", e3["e1_cells"]["val_H5_r"]["a"]),
        ("M1 · 물체 자세 정의 (보조)", e3["e1_cells"]["val_H5_r_obj"]["a"]),
        ("M1 · 학습 정책 episode (보조)", e3["dp"]["H5"]["a_std"]),
        ("M2 · Cosmos Policy (주 칸)", {"p_near_real": h3m2["p_real_near"], "p_near_imag": h3m2["p_imag_near"],
                                         "share": h3m2["share"]}),
        ("M3 · p = 0.03 (주 칸)", e3["m3"]["0.03"]),
    ]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    style_axes(ax, grid_axis="x")
    gap = 0.003
    for i, (lab, c) in enumerate(cells):
        y = len(cells) - 1 - i
        floor, imag = c["p_near_real"], c["p_near_imag"]
        ax.barh(y, floor, height=0.46, color=GRAY, zorder=3, label="판정기 자체 (지각 바닥)" if i == 0 else None)
        ax.barh(y, imag - floor - gap if imag > floor else 0, left=floor + gap, height=0.46, color=BLUE, zorder=3,
                label="상상이 더한 몫" if i == 0 else None)
        ax.text(imag + 0.008, y, f"{imag:.3f}  ·  상상 몫 {c['share']:.2f}", va="center", fontsize=6.9, color=INK)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([lab for lab, _ in cells][::-1], fontsize=7.2, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.72)
    ax.set_ylim(-0.6, len(cells) - 0.4)
    ax.set_xlabel("near-miss를 성공으로 판정한 비율 (상상 프레임)", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=6.8, bbox_to_anchor=(1.0, 1.0))
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.2, top=0.97)
    return save(fig, "fig08_h3_decomp.png")


# ---------------------------------------------------------------- 9. policy level
def fig09_policy():
    m3 = load("E2_server/m3/e2_m3_analysis.json")["0.03"]["r29"]["table"]
    m1 = load("E2_server/m1/e2_m1_analysis.json")["dp"]["H5"]
    names = {"pi1_precise": "π1 정확", "pi2_good": "π2 양호", "pi3_medium": "π3 보통", "pi4_sloppy": "π4 엉성",
             "pi5_far_failer": "π5 far-failer", "pi6_near_failer": "π6 near-failer"}
    colors = {"pi5_far_failer": ORANGE, "pi6_near_failer": BLUE}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw={"wspace": 0.42, "width_ratios": [1.3, 1]})
    style_axes(ax, grid_axis="x")
    for i, row in enumerate(m3):
        y = len(m3) - 1 - i
        c = colors.get(row["policy"], GRAY_DARK)
        ax.plot([row["s"], row["sh"]], [y, y], color=GRAY_LIGHT, lw=2.0, zorder=2, solid_capstyle="round")
        ax.plot(row["s"], y, "o", ms=6, mfc="white", mec=c, mew=1.4, zorder=4)
        ax.plot(row["sh"], y, "o", ms=6, mfc=c, mec="white", mew=1.0, zorder=5)
        ax.text(1.075, y, f"{row['n']:.2f}", va="center", ha="right", fontsize=6.9, color=INK2)
    ax.text(1.075, len(m3) - 0.35, "near 비중", va="bottom", ha="right", fontsize=6.6, color=INK2)
    ax.set_yticks(range(len(m3)))
    ax.set_yticklabels([names[r["policy"]] for r in m3][::-1], fontsize=7.2, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.42, 1.08)
    ax.set_ylim(-0.6, len(m3) - 0.1)
    ax.set_xlabel("성공률", color=INK)
    ax.plot([], [], "o", ms=6, mfc="white", mec=GRAY_DARK, mew=1.4, label="실제")
    ax.plot([], [], "o", ms=6, mfc=GRAY_DARK, mec="white", label="상상")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, bbox_to_anchor=(0.0, 1.0), ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    panel_title(ax, "(a) M3 정책 6개 (p = 0.03)")

    style_axes(bx)
    tab = m1["table"]
    bx.axhline(0, color=AXIS, lw=0.8, zorder=1)
    bx.plot([r["n"] for r in tab], [r["o_imag"] for r in tab], "o", ms=5, mfc=BLUE, mec="white", mew=0.8,
            zorder=3)
    rho = m1["point"]["spearman_n_oimag"]
    bx.text(0.02, 0.95, f"Spearman {rho:.2f}", transform=bx.transAxes, fontsize=6.9, color=INK2, va="top")
    bx.set_xlim(-0.01, 0.31)
    bx.set_ylim(-0.03, 0.09)
    bx.set_xlabel("near-miss 비중 n", color=INK)
    bx.set_ylabel("상상 과대 추정", color=INK)
    panel_title(bx, "(b) M1 Diffusion Policy 24개")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.19, top=0.9)
    return save(fig, "fig09_policy.png")


# ---------------------------------------------------------------- 10. data-side prescriptions (H5a)
def fig10_h5a():
    e5 = load("E5_server/e5_analysis.json")["H5a"]
    keys = ["ref", "rw", "near", "cp", "full", "bw"]
    labels = ["(0)", "(i)", "(ii)", "(iii)", "(iv)", "(v)"]
    cols = [GRAY_DARK, BLUE, ORANGE, GRAY_LIGHT, GRAY_LIGHT, GRAY_LIGHT]
    sets = [("M1 (원 정의, H = 5)", "M1"), ("M3 (p = 0.03)", "M3_p0.03"), ("M3 (p = 0.1)", "M3_p0.1")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (title, k) in zip(axes, sets):
        style_axes(ax)
        vals = [e5[k][p]["NM"] for p in keys]
        xs = np.arange(len(keys))
        ax.bar(xs, vals, width=0.58, color=cols, zorder=3)
        for x in (0, 1, 2):
            ax.text(x, vals[x] + 0.012, f"{vals[x]:.2f}", ha="center", va="bottom", fontsize=6.6, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 0.62)
        panel_title(ax, title)
    axes[0].set_ylabel("NM (near-miss → 성공)", color=INK)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.13, top=0.87)
    return save(fig, "fig10_h5a.png")


# ---------------------------------------------------------------- 11. M3 predictor capacity sweep
def fig11_capacity():
    sw = load("E4_server/e4_analysis.json")["4b_sweep"]
    rows = [("R3", "작은 예측기 (폭 64 · 2층)", RAMP3[0]), ("base", "기본 (폭 128 · 4층)", RAMP3[1]),
            ("R4", "큰 예측기 (폭 256 · 8층)", RAMP3[2]), ("R1", "토큰 256개", GRAY), ("R2", "PCA 384차원", GRAY)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.15), sharey=True, gridspec_kw={"wspace": 0.08})
    ys = np.arange(len(rows))[::-1]
    for a in (ax, bx):
        style_axes(a, grid_axis="x")
        a.tick_params(axis="y", length=0)
    ax.barh(ys, [sw[k]["NM"] for k, _, _ in rows], height=0.5, color=[c for _, _, c in rows], zorder=3)
    for y, (k, _, _) in zip(ys, rows):
        ax.text(sw[k]["NM"] + 0.02, y, f"{sw[k]['NM']:.3f}", va="center", fontsize=6.9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _ in rows], fontsize=7.2, color=INK)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("NM (near-miss → 성공)", color=INK)
    panel_title(ax, "(a) near-miss 오판")
    bx.barh(ys, [sw[k]["auc_sn_imag"] for k, _, _ in rows], height=0.5, color=[c for _, _, c in rows], zorder=3,
            label="고정 판별기")
    bx.plot([sw[k]["auc_sn_ceiling"] for k, _, _ in rows], ys, "D", ms=4.6, mfc="white", mec=INK2, mew=1.0,
            zorder=4, label="상한 판별기", ls="none")
    bx.set_xlim(0.5, 1.02)
    bx.set_xlabel("AUROC 성공 대 near-miss", color=INK)
    panel_title(bx, "(b) 경계 해상도")
    bx.legend(loc="lower right", frameon=False, fontsize=6.4, bbox_to_anchor=(1.0, 1.0), ncol=2,
              handlelength=1.4, columnspacing=0.8, borderaxespad=0.1)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.2, top=0.86)
    return save(fig, "fig11_capacity.png")


# ---------------------------------------------------------------- 12. mechanism chain with status
def fig12_chain():
    fig, ax = canvas(W, 3.2)
    h = 0.55
    ys = (2.42, 1.47, 0.52)
    c1 = box(ax, 0.08, ys[0], 1.5, h, "성공에 치우친\n학습 데이터", "orange")
    c2 = box(ax, 0.08, ys[1], 1.5, h, "예측기 용량과\n표현의 한계", "orange")
    c3 = box(ax, 0.08, ys[2], 1.5, h, "판정기 자체의 흐림\n(지각 바닥)")
    m1 = box(ax, 1.95, ys[0], 1.45, h, "상상 분포가\n성공 쪽으로 밀림", "blue")
    m2 = box(ax, 1.95, ys[1], 1.45, h, "경계 근처의\n정보 손실", "blue")
    n = box(ax, 3.78, ys[1], 1.25, h, "near-miss를\n성공으로 오판", "blue", weight="semibold")
    q1 = box(ax, 5.35, ys[0], 1.2, h, "정책의 상상\n과대 추정", "blue")
    q2 = box(ax, 5.35, ys[1], 1.2, h, "비슷한 정책의\n순위 왜곡")
    q3 = box(ax, 5.35, ys[2], 1.2, h, "checkpoint\n선택의 손해")
    arrow(ax, right(c1), left(m1))
    arrow(ax, right(c2), left(m2))
    arrow(ax, right(m1), (3.95, ys[1] + h))
    arrow(ax, (m2[0] + m2[2], ys[1] + 0.36), (n[0], ys[1] + 0.36))
    # the judge's own blur joins below the mechanism row, clear of the status labels
    line(ax, [right(c3), (3.6, right(c3)[1]), (3.6, ys[1] + 0.17)])
    arrow(ax, (3.6, ys[1] + 0.17), (n[0], ys[1] + 0.17))
    arrow(ax, (4.85, ys[1] + h), left(q1))
    arrow(ax, bottom(q1), top(q2))
    arrow(ax, bottom(q2), top(q3))
    ax.text(4.62, 2.62, "+ 정책의\nnear-miss 비중", fontsize=6.4, color=INK2, ha="center", va="center",
            linespacing=1.3)
    off = 0.15
    pill(ax, top(m1)[0], ys[0] - off, "확인 · M3 (H4a, H1a)", "aqua")
    pill(ax, top(m2)[0], ys[1] - off, "확인 · M1 (H1b, H4b)", "aqua")
    pill(ax, top(c2)[0], ys[1] - off, "M3 용량 실험 [서술]", "yellow")
    pill(ax, top(c3)[0], ys[2] - off, "M1 원 정의 · VLM에서 큼", "gray")
    pill(ax, top(n)[0], ys[1] - off, "상상 몫 M2 0.83 · M3 1.00", "aqua")
    pill(ax, top(q1)[0], ys[0] - off, "M3 확인 · M1은 작음", "yellow")
    pill(ax, top(q2)[0], ys[1] - off, "역전 없음 (두 모델)", "gray")
    pill(ax, top(q3)[0], ys[2] - off, "시험하지 않음 (E6)", "gray")
    lx = 0.08
    for tint, lab in (("aqua", "확인"), ("yellow", "부분 또는 서술 결과"), ("gray", "확인 안 됨 · 시험 안 함 · 참고")):
        w = pill_width(lab, fs=6.2)
        pill(ax, lx + w / 2, 0.1, lab, tint, fs=6.2)
        lx += w + 0.1
    return save(fig, "fig12_chain.png")


FIGS = [fig01_world_models, fig02_policy_eval, fig03_concept, fig04_pipeline, fig05_m1, fig06_m3, fig07_m2,
        fig08_h3, fig09_policy, fig10_h5a, fig11_capacity, fig12_chain]


def main():
    register_fonts()
    for f in FIGS:
        print("wrote", f())


if __name__ == "__main__":
    main()
