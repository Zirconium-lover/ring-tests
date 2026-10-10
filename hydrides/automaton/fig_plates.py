"""Пластинки модели как есть — каждая чёрточка одна пластинка (центр, угол, длина), без растра и без съёмки.

python fig_plates.py расчёт.npz [--exp снимок_до.png снимок_1.png ...] [--h 0.32] [--out рисунок.png]

Берёт списки пластинок P_<стадия>, которые bench.py сохраняет рядом с полями. Справа — увеличенный фрагмент
60 × 60 мкм (самый густой под последней нагрузкой) с толщиной пластинки в масштабе. --exp — снимки опыта
(пиксель 3.5 мкм), по одному на стадию: слева вырез того же размера из центра снимка.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

BLUE, GREY, ORANGE = "#2b6cb0", "#8a8f98", "#dd6b20"
PANEL_IN = 4.6


def segs(P, H, W):
    out, cols = [], []
    for cy, cx, psi, half, _ in P:
        if half <= 0:
            continue
        dy, dx = -np.sin(psi) * half, np.cos(psi) * half
        dev = np.degrees(abs(np.arctan(np.tan(psi))))
        col = BLUE if dev <= 40 else (ORANGE if dev >= 65 else GREY)
        for oy in (-H, 0, H):                        # поле периодическое: копии у краёв
            for ox in (-W, 0, W):
                out.append([(cx - dx + ox, cy - dy + oy), (cx + dx + ox, cy + dy + oy)]); cols.append(col)
    return out, cols


def draw(ax, P, H, W, lim, lw):
    s, c = segs(P, H, W)
    ax.add_collection(LineCollection(s, colors=c, linewidths=lw, capstyle="butt"))
    ax.set_xlim(lim[1], lim[1] + lim[2]); ax.set_ylim(lim[0] + lim[2], lim[0]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def best_zoom(P, H, W, side=60.0, step=10.0):
    best = (-1.0, 0.0, 0.0)
    for y0 in np.arange(0, H - side + 1, step):
        for x0 in np.arange(0, W - side + 1, step):
            m = (P[:, 0] >= y0) & (P[:, 0] < y0 + side) & (P[:, 1] >= x0) & (P[:, 1] < x0 + side)
            L = float(2 * P[m, 3].sum())
            if L > best[0]:
                best = (L, y0, x0)
    return best[1], best[2], side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("--exp", nargs="*", default=[]); ap.add_argument("--h", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.npz)
    stages = [k[2:] for k in z.files if k.startswith("P_")]
    H, W = (np.array(z[stages[0]].shape) * float(z["dx"])).tolist()
    zoom = best_zoom(z["P_" + stages[-1]], H, W)
    nc = 3 if a.exp else 2
    fig, axs = plt.subplots(len(stages), nc, figsize=(nc * PANEL_IN + 0.6, len(stages) * PANEL_IN + 1.7), squeeze=False)
    for i, st in enumerate(stages):
        P = z["P_" + st]; j = 0
        if a.exp and i < len(a.exp):
            from PIL import Image
            im = np.asarray(Image.open(a.exp[i]).convert("L"), float) / 255
            w = int(round(min(H, W) / 3.5)); y0 = im.shape[0] // 2 - w // 2; x0 = im.shape[1] // 2 - w // 2
            axs[i, 0].imshow(im[y0:y0 + w, x0:x0 + w], cmap="gray", vmin=0, vmax=1, extent=(0, min(H, W), min(H, W), 0))
            axs[i, 0].set_title(f"опыт: {os.path.basename(a.exp[i])}", fontsize=10, loc="left")
            axs[i, 0].set_xticks([]); axs[i, 0].set_yticks([]); j = 1
        draw(axs[i, j], P, H, W, (0.0, 0.0, max(H, W)), 0.9)
        axs[i, j].add_patch(Rectangle((zoom[1], zoom[0]), zoom[2], zoom[2], fill=False, ec="black", lw=0.8, ls="--"))
        n0 = int(P[:, 4].sum())
        axs[i, j].set_title(f"модель, {st}: {len(P)} пластинок" + (f" ({n0} нерастворившихся)" if n0 else ""), fontsize=10, loc="left")
        draw(axs[i, j + 1], P, H, W, zoom, a.h * PANEL_IN * 72 / zoom[2])
        axs[i, j + 1].set_title(f"увеличено {zoom[2]:.0f} × {zoom[2]:.0f} мкм, толщина в масштабе", fontsize=10, loc="left")
    fig.suptitle(f"Пластинки модели как есть: каждая чёрточка — одна пластинка, без растра и без съёмки ({os.path.basename(a.npz)}).\n"
                 f"Толщина {a.h:.2f} мкм в масштабе на увеличенном фрагменте; на всём поле линии утолщены для видимости.\n"
                 "Цвет — угол к окружному: синие ≤ 40°, серые 40–65°, оранжевые ≥ 65°.", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out or a.npz[:-4] + "_plates.png", dpi=110, facecolor="white")


if __name__ == "__main__":
    main()
