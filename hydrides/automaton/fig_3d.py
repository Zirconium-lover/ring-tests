"""Рисунок пилота 3D: диски гидридов в кубе 48 мкм (без напряжения и при 250 МПа) и сечение r–θ.
python fig_3d.py папка_пилота"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection

D = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, MUTED, BG, GRID, GRAY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc", "#a9a8a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
L = 48.0


def trace_dev(nv):
    d = np.cross(nv, [0, 0, 1.0]); nrm = np.linalg.norm(d, axis=1)
    d = d / np.maximum(nrm, 1e-9)[:, None]
    return np.degrees(np.arccos(np.clip(np.abs(d[:, 0]), 0, 1)))


def col(dev):
    return [BLUE if v <= 40 else (ORANGE if v >= 65 else GRAY) for v in dev]


def discs(ax, z):
    c, nv, R = z["c"], z["n"], z["R"]
    polys = []
    for ci, ni, ri in zip(c, nv, R):
        u = np.cross(ni, [0, 0, 1.0] if abs(ni[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u)
        v = np.cross(ni, u)
        a = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        pts = ci + ri * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
        polys.append(pts[:, [0, 2, 1]])                    # оси рисунка: TD, L, ND (ND вверх)
    pc = Poly3DCollection(polys, facecolors=col(trace_dev(nv)), edgecolors="none", alpha=0.85)
    ax.add_collection3d(pc)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_xlabel("TD, мкм"); ax.set_ylabel("ось трубы, мкм"); ax.set_zlabel("ND (r), мкм")
    ax.view_init(elev=18, azim=-60)


def section(ax, z, zc):
    c, nv, R = z["c"], z["n"], z["R"]
    segs, cols = [], []
    dev = trace_dev(nv)
    for ci, ni, ri, dv in zip(c, nv, R, dev):
        sin_t = np.sqrt(max(1e-12, 1 - ni[2] ** 2))
        dz = (zc - ci[2] + L / 2) % L - L / 2
        rr = abs(dz) / sin_t
        if rr >= ri:
            continue
        half = np.sqrt(ri * ri - rr * rr)
        d = np.cross(ni, [0, 0, 1.0]); d /= np.linalg.norm(d)
        # центр хорды: точка плоскости диска на высоте zc, ближайшая к центру
        w = np.array([0, 0, 1.0]) - ni[2] * ni; w /= max(np.linalg.norm(w), 1e-9)
        p0 = ci + w * (dz / max(w[2], 1e-9))
        segs.append([[p0[0] - half * d[0], p0[1] - half * d[1]], [p0[0] + half * d[0], p0[1] + half * d[1]]])
        cols.append(col([dv])[0])
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.2))
    ax.set_xlim(0, L); ax.set_ylim(L, 0); ax.set_aspect("equal")
    ax.set_xlabel("TD, мкм"); ax.set_ylabel("ND (r), мкм")


fig = plt.figure(figsize=(16, 9.5), facecolor=BG)
fig.subplots_adjust(wspace=0.45)
for row, s in enumerate((0, 250)):
    fn = os.path.join(D, f"d3_sL21.0_s{s}_seed1")
    z = np.load(fn + ".npz"); m = json.load(open(fn + ".json"))
    ax = fig.add_subplot(2, 3, 3 * row + 1, projection="3d")
    discs(ax, z)
    ax.set_title(f"{'без напряжения' if s == 0 else f'{s} МПа по окружности'}: {m['n_plates']} дисков", loc="left", color=INK)
    for k, zc in enumerate((12.0, 36.0)):
        ax2 = fig.add_subplot(2, 3, 3 * row + 2 + k)
        section(ax2, z, zc)
        ax2.set_title(f"сечение r–θ, z = {zc:.0f} мкм" + (f"   (RHF по сечениям {m['RHF_trace']:.2f})" if k == 0 else ""), loc="left", color=INK, fontsize=10)
fig.suptitle("Пилот 3D: диски гидрида в кубе 48 мкм (текстура с осевой составляющей, f_L ≈ 0.11). "
             "Синие — окружные следы (≤ 40° к TD), оранжевые — радиальные (≥ 65°)", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig9_pilot3d.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
