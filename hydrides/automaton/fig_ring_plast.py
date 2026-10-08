"""Рисунок: кольца Э635 (Плясов и др. 2023, образец 1, 200 Н, 152 ppm) — модель с таблицами ореолов против модели
с коллективной пластичностью и опыт (табл. 1: F_l по третям стенки). Модель — полоса 850 × 120 мкм, наружная
поверхность сверху, фора 5 °C.

python fig_ring_plast.py папка_ореолы_90 папка_ореолы_0 папка_пластика [--photo снимок_рис3.png] [--out файл]
--photo — вырезка рис. 3 статьи (только для показа; снимки статьи в репозиторий не кладём)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
C_EXP, C_HALO, C_PL = "#0b0b0b", "#eb6834", "#2a78d6"
EXP = {"S1_90": (0.10, 0.69, 0.89), "S1_0": (0.39, 0.10, 0.01)}
TITLE = {"S1_90": "участок 90°: внутренняя поверхность растянута (до 215 МПа)",
         "S1_0": "участок 0°: наружная поверхность растянута (до 63 МПа)"}
PHOTO_BOX = {"S1_90": (622, 195, 1032, 503), "S1_0": (1068, 195, 1478, 503)}    # рис. 3а, «после»


def pick(folder, site, plast):
    for f in glob.glob(os.path.join(folder, f"*ring{site}_seed1.json")):
        m = json.load(open(f))
        extra = any(m.get(k) for k in ("halo_cap", "halo_smax", "app_center"))
        if bool(m.get("plast")) == plast and not extra and (plast or m.get("halo", True) is not False):
            return m, np.load(f[:-5] + ".npz")
    raise FileNotFoundError(site)


def show(ax, z, title):
    h = np.asarray(z["hyd"], float); dx = float(z["dx"])
    k = 3
    h = h[: h.shape[0] // k * k, : h.shape[1] // k * k].reshape(h.shape[0] // k, k, h.shape[1] // k, k).max((1, 3))
    ax.imshow(1 - 0.85 * np.clip(h * 1.5, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="antialiased",
              extent=(0, h.shape[1] * k * dx, h.shape[0] * k * dx, 0))
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("дуга, мкм", color=MUTED, fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("halo90"); ap.add_argument("halo0"); ap.add_argument("plast")
    ap.add_argument("--photo"); ap.add_argument("--out", default=os.path.join(HERE, "figs", "fig27_ring_plast.png"))
    a = ap.parse_args()
    ph = a.photo is not None
    ncol = 4 if ph else 3
    fig = plt.figure(figsize=(15 if ph else 11, 9.5))
    gs = fig.add_gridspec(2, ncol, width_ratios=([2.6] if ph else []) + [1, 1, 2.2], wspace=0.35, hspace=0.38)
    for r, site in enumerate(("S1_90", "S1_0")):
        mh, zh = pick(a.halo90 if site == "S1_90" else a.halo0, site, False)
        mp, zp = pick(a.plast, site, True)
        c = 0
        if ph:
            ax = fig.add_subplot(gs[r, 0])
            ax.imshow(Image.open(a.photo).crop(PHOTO_BOX[site]))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("опыт: рис. 3а Плясова (после, " + ("90°" if site == "S1_90" else "0°") + ")\nмасштаб ≈ 1.3 × 0.85 мм",
                         loc="left", fontsize=9)
            c = 1
        ax = fig.add_subplot(gs[r, c]); show(ax, zh, "модель: таблицы\nореолов")
        ax.set_ylabel("от наружной поверхности, мкм", color=MUTED, fontsize=8)
        ax = fig.add_subplot(gs[r, c + 1]); show(ax, zp, "модель: коллективная\nпластичность")
        ax = fig.add_subplot(gs[r, c + 2])
        x = np.arange(3); w = 0.26
        for i, (lab, vals, col) in enumerate((("опыт (табл. 1)", EXP[site], C_EXP),
                                              ("ореолы", [mh[f"F_l_5_{s}"] for s in ("out", "mid", "in")], C_HALO),
                                              ("коллективная пластичность", [mp[f"F_l_5_{s}"] for s in ("out", "mid", "in")], C_PL))):
            ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab)
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(x); ax.set_xticklabels(["наружная", "средняя", "внутренняя"]); ax.set_xlabel("треть стенки", color=MUTED, fontsize=8)
        ax.set_ylim(0, 1.12); ax.set_ylabel("F_l — доля радиальных (> 5 мкм)", color=MUTED, fontsize=9)
        ax.set_title(TITLE[site] + f"\nобъектов: ореолы {mh['n_obj_5']}, пластичность {mp['n_obj_5']}", loc="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
        if r == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Кольцо Э635, 200 Н, 152 ppm, фора 5 °C: таблицы ореолов против коллективной пластичности",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(a.out, dpi=120, facecolor="white", bbox_inches="tight")
    print("→", a.out)


if __name__ == "__main__":
    main()
