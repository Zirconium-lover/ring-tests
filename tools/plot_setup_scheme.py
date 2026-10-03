#!/usr/bin/env python3
"""Схема расчётной постановки испытания кольца на сегментной оправке.

    python3 tools/plot_setup_scheme.py [-o notes/figs/setup_scheme.png]

Три панели: вид сверху с расчётным сектором, осевой разрез с плоскостью
симметрии по высоте, развёртка внутренней поверхности для двух высот кольца.
Размеры кольца — направляющий канал 11.0 x 12.6 мм; оправка схематическая
(8 сегментов, конус 20 градусов при вершине).
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge, Rectangle, FancyArrowPatch

RI, RO = 5.5, 6.3          # кольцо, мм
T = RO - RI
NSEG = 8
GAP_DEG = 4.0              # видимый зазор между сегментами (схема)
RC = 2.4                   # отверстие под конус в сегментах (схема)
HALF_CONE = np.radians(10.0)

C_RING = "#9aa5b1"
C_RING_SECTOR = "#d1495b"
C_SEG = "#c9d6e3"
C_SEG_SECTOR = "#5b8fc7"
C_STRIP = "#f2b134"
C_BAND = "#222222"


def panel_top(ax):
    ax.set_title("а) Вид сверху: расчётный сектор 22.5°", fontsize=11)
    pitch = 360.0 / NSEG
    for k in range(NSEG):
        c = k * pitch
        a0, a1 = c - pitch / 2 + GAP_DEG / 2, c + pitch / 2 - GAP_DEG / 2
        ax.add_patch(Wedge((0, 0), RI, a0, a1, width=RI - RC,
                           fc=C_SEG, ec="white", lw=0.8))
    ax.add_patch(Wedge((0, 0), RO, 0, 360, width=T, fc=C_RING, ec="none"))
    # расчётный сектор: от середины сегмента (0°) до середины зазора (22.5°)
    ax.add_patch(Wedge((0, 0), RI, 0, pitch / 2 - GAP_DEG / 2,
                       width=RI - RC, fc=C_SEG_SECTOR, ec="none"))
    ax.add_patch(Wedge((0, 0), RO, 0, pitch / 2, width=T,
                       fc=C_RING_SECTOR, ec="none"))
    ax.add_patch(plt.Circle((0, 0), RC - 0.15, fc="#eeeeee", ec="#999999"))
    ax.text(-0.2, -1.0, "конус", ha="center", va="center", fontsize=9)
    for ang, lab, va in ((0.0, "φ = 0\nсередина сегмента", "top"),
                         (pitch / 2, "φ = 22.5°\nсередина зазора", "bottom")):
        a = np.radians(ang)
        ax.plot([0, 7.4 * np.cos(a)], [0, 7.4 * np.sin(a)], "k--", lw=1)
        ax.text(7.5 * np.cos(a), 7.5 * np.sin(a) + (-0.2 if va == "top" else 0.2),
                lab, fontsize=9, ha="left", va=va)
    a = np.radians(pitch / 4)
    ax.add_patch(FancyArrowPatch((3.2 * np.cos(a), 3.2 * np.sin(a)),
                                 (4.9 * np.cos(a), 4.9 * np.sin(a)),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color="k", lw=1.5))
    ax.text(3.0, 1.35, "$u_r$", fontsize=11)
    ax.annotate("Ø 11.0", xy=(-RI * 0.707, -RI * 0.707),
                xytext=(-8.4, -6.9), fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.annotate("Ø 12.6", xy=(-RO * 0.94, RO * 0.34),
                xytext=(-8.6, 5.0), fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.text(1.5, -7.6, "8 сегментов — 16 одинаковых секторов;\n"
            "считаем один, на гранях — условия симметрии",
            ha="center", va="top", fontsize=9)
    ax.set_xlim(-9, 13)
    ax.set_ylim(-9.2, 7.6)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_section(ax, H=6.0):
    ax.set_title("б) Осевой разрез: половина высоты", fontsize=11)
    over = 1.5                      # сегмент выше кольца на столько с каждой стороны
    zb, zt = -H / 2 - over, H / 2 + over
    rin_b, rin_t = RC, RC + (zt - zb) * np.tan(HALF_CONE)
    # конус (в модели не нужен)
    tc = np.tan(HALF_CONE)
    ax.add_patch(Polygon([(0, zb - 0.8), (rin_b - 0.8 * tc, zb - 0.8),
                          (rin_t + 2.0 * tc, zt + 2.0), (0, zt + 2.0)],
                         fc="#eeeeee", ec="#999999", lw=0.8))
    ax.text(1.0, zt + 1.2, "конус", fontsize=9, ha="center")
    # сегмент
    ax.add_patch(Polygon([(rin_b, zb), (RI, zb), (RI, zt), (rin_t, zt)],
                         fc=C_SEG_SECTOR, ec="k", lw=0.8))
    ax.text((RI + rin_b) / 2 + 0.25, zb + 0.6, "сегмент\n(жёсткий)",
            fontsize=8.5, ha="center", color="white")
    # кольцо: считаемая половина z >= 0 и отброшенная z < 0
    ax.add_patch(Rectangle((RI, 0), T, H / 2, fc=C_RING_SECTOR, ec="k", lw=0.8))
    ax.add_patch(Rectangle((RI, -H / 2), T, H / 2, fc="white", ec="k",
                           lw=0.8, hatch="///"))
    ax.plot([-0.3, RO + 1.4], [0, 0], "k--", lw=1)
    ax.text(RO + 1.5, 0, "плоскость\nсимметрии z = 0", fontsize=8.5,
            va="center")
    ax.plot([0, 0], [zb - 1.0, zt + 2.5], "k-.", lw=0.8)
    ax.text(0.1, zb - 1.0, "ось", fontsize=8, va="bottom")
    ax.annotate("", xy=(RO + 0.45, H / 2), xytext=(RO + 0.45, -H / 2),
                arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(RO + 0.6, -H / 4, "H", fontsize=11, va="center")
    ax.annotate("свободный торец", xy=(RI + T / 2, H / 2),
                xytext=(RO + 0.9, H / 2 + 1.6), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.annotate("контакт с трением μ", xy=(RI, H / 4),
                xytext=(RO + 0.9, H / 4 + 0.6), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.annotate("", xy=(RI - 0.2, zt - 0.8), xytext=(RI - 1.7, zt - 0.8),
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.text(RI - 1.6, zt - 0.6, "$u_r$", fontsize=11)
    ax.annotate("", xy=(0.9, zt + 0.3), xytext=(0.9, zt + 2.3 + 0.9),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#777777"))
    ax.text(1.1, zt + 2.6, "$u_z$", fontsize=10, color="#777777")
    ax.text(-0.3, zb - 1.9,
            "конус не моделируем: каждый сегмент смещается\n"
            "радиально на $u_r = u_z\\,\\tan(θ/2)$",
            fontsize=8.5, va="top")
    ax.set_xlim(-0.5, RO + 4.6)
    ax.set_ylim(zb - 3.6, zt + 3.4)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_unrolled(ax, H, note):
    pitch = 2 * np.pi * RI / NSEG           # 4.32 мм по внутренней поверхности
    gap = 2 * np.pi * RI * GAP_DEG / 360
    s0, s1 = -1.0, 2.6 * pitch
    ax.add_patch(Rectangle((s0, 0), s1 - s0, H, fc="#f4f4f4", ec="k", lw=0.8))
    for k in range(-1, 4):
        g = (k + 0.5) * pitch
        a, b = g - pitch + gap / 2, g - gap / 2
        a, b = max(a, s0), min(b, s1)
        if b > a:
            ax.add_patch(Rectangle((a, 0), b - a, H, fc=C_SEG, ec="none"))
        for e in (g - gap / 2, g + gap / 2):
            if s0 < e < s1:
                w = 0.45
                x = e - w if e < g else e
                ax.add_patch(Rectangle((x, 0), w, H, fc=C_STRIP, ec="none",
                                       alpha=0.85))
    # полоса среза ~45° от торца до торца, начинается у края сегмента
    g = 0.5 * pitch
    xs = g - gap / 2 - 0.2
    ax.plot([xs, xs + H], [0, H], color=C_BAND, lw=2.6)
    ax.set_xlim(s0, s1)
    ax.set_ylim(-0.2, H + 0.2)
    ax.set_aspect("equal")
    ax.set_yticks([0, H])
    ax.set_yticklabels(["0", f"H = {H:g}"], fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.text(s1 + 0.25, H / 2, note, fontsize=8.5, va="center", ha="left")
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default="notes/figs/setup_scheme.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.o) or ".", exist_ok=True)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.12,
                          wspace=0.05)
    panel_top(fig.add_subplot(gs[0, 0]))
    panel_section(fig.add_subplot(gs[0, 1]))
    sub = gs[1, :].subgridspec(1, 2, wspace=0.12)
    ax1 = fig.add_subplot(sub[0, 0])
    ax2 = fig.add_subplot(sub[0, 1])
    fig.text(0.5, 0.49, "в) Развёртка внутренней поверхности кольца "
             "(схема, один и тот же масштаб)", ha="center", fontsize=11)
    panel_unrolled(ax1, 3.0, "")
    panel_unrolled(ax2, 10.0, "")
    ax1.set_title("H = 3 мм, H/t ≈ 4: полоса короткая,\n"
                  "торцы рядом по всей её длине", fontsize=9.5)
    ax2.set_title("H = 10 мм, H/t ≈ 12: середина высоты — как в длинной трубе;\n"
                  "полоса длиной ≈ H идёт и над сегментом, и над зазором",
                  fontsize=9.5)
    for ax in (ax1, ax2):
        ax.set_xlabel("дуга по внутренней поверхности, мм", fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, fc=C_SEG),
               Rectangle((0, 0), 1, 1, fc="#f4f4f4", ec="k", lw=0.5),
               Rectangle((0, 0), 1, 1, fc=C_STRIP),
               plt.Line2D([], [], color=C_BAND, lw=2.6)]
    fig.legend(handles, ["над сегментом (контакт)", "над зазором",
                         "зона максимальной деформации у края сегмента",
                         "полоса среза ~45° (как в опытах 2008 г.)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2,
               fontsize=9, frameon=False)
    fig.savefig(args.o, dpi=150, bbox_inches="tight")
    print("wrote", args.o)


if __name__ == "__main__":
    main()
