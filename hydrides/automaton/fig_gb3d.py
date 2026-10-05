"""Рис. 13: межзёренные отрезки и переход через границу в 3D (куб 64 мкм, параметры 2D без подгонки).
python fig_gb3d.py папка_межзёренных папка_отрезков"""
import os
import sys
import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sec3d import hyd_from_cells  # noqa: E402

D, D0 = sys.argv[1], sys.argv[2]
BLUE, ORANGE, GREEN, PURPLE, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#3b9a5a", "#8a5cc2", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}
N, DX, L = 128, 0.5, 64.0


def variant(m):
    if not m.get("gb", False):
        return "cross" if m.get("cross_deg", 0) > 0 else "base"
    return f"gb_{m.get('gb_eps', 'facet')}" + ("_cross" if m.get("cross_deg", 0) > 0 else "")


runs = defaultdict(lambda: defaultdict(list))
for d in (D, D0):
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        if m.get("shape") != "segment" or m.get("img_method") != "voxel" or m.get("beta", 0.12) != 0.12:
            continue
        runs[variant(m)][int(m["sigma_app"])].append((m, f[:-5]))

VAR = [("base", MUTED, "отрезки в теле зерна"),
       ("cross", GREEN, "+ переход через границу (15°)"),
       ("gb_grain", PURPLE, "+ межзёренные, несоответствие в осях зерна"),
       ("gb_facet", BLUE, "+ межзёренные, несоответствие по нормали к грани"),
       ("gb_facet_cross", ORANGE, "+ межзёренные (по грани) + переход")]


def curve(ax, key, title, ylab="RHF"):
    for v, c, lab in VAR:
        if v not in runs:
            continue
        s = sorted(runs[v])
        y = [np.mean([m.get(key, 0.0) for m, _ in runs[v][q]]) for q in s]
        e = [np.std([m.get(key, 0.0) for m, _ in runs[v][q]]) for q in s]
        ax.errorbar(s, y, yerr=e, color=c, marker="o", ms=4, lw=1.8, capsize=2, label=lab)
    ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel(ylab); ax.grid(color=GRID)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)


fig = plt.figure(figsize=(17, 11.5), facecolor=BG)
gs = fig.add_gridspec(2, 3, hspace=0.33, wspace=0.27)
ax = fig.add_subplot(gs[0, 0])
curve(ax, "RHF_image", "RHF по изображению сечений")
for q, (a, b) in LEP.items():
    ax.plot([q], [a], "ks", ms=7, mfc="none"); ax.plot([q], [b], "k^", ms=7, mfc="none")
ax.plot([], [], "ks", mfc="none", label="Lepine, MATLAB"); ax.plot([], [], "k^", mfc="none", label="Lepine, HAPPy")
ax.set_ylim(0, 0.95); ax.legend(frameon=False, fontsize=7.8, loc="upper left")
ax = fig.add_subplot(gs[0, 1])
curve(ax, "gb_frac", "Доля гидрида в межзёренных отрезках (цвета как слева)", "доля площади пластинок")
ax.set_ylim(0, 1)

ax = fig.add_subplot(gs[0, 2])
bins = np.arange(0, 91, 5)
for v, c, lab in VAR:
    if v not in runs or 250 not in runs[v]:
        continue
    dv_all, w_all = [], []
    for m, fn in runs[v][250]:
        z = np.load(fn + ".npz")
        d = np.cross(z["n"], [0, 0, 1.0]); sn = np.linalg.norm(d, axis=1)
        dv_all.append(np.degrees(np.arccos(np.clip(np.abs(d[:, 0]) / np.maximum(sn, 1e-9), 0, 1)))); w_all.append(z["area"] * sn)
    dv, w = np.concatenate(dv_all), np.concatenate(w_all)
    ax.hist(dv, bins=bins, weights=w / w.sum(), histtype="step", lw=1.8, color=c, label=f"средний {np.average(dv, weights=w):.0f}° (цвета как слева)" if v == "base" else f"средний {np.average(dv, weights=w):.0f}°")
ax.axvline(47.7, color=INK, lw=1.2, ls="--")
ax.text(46.5, 0.005, "FIB-SEM Lepine:\nрадиальный\nпакет 47.7°", fontsize=8.5, color=INK, ha="right")
ax.set_xlabel("угол следа пластинки к TD, град"); ax.set_ylabel("доля площади пластинок"); ax.grid(color=GRID)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Наклон пластинок при 250 МПа", loc="left", fontsize=10.5, color=INK)


def section(ax, v, s, title):
    m, fn = runs[v][s][0]
    z = np.load(fn + ".npz")
    ptr = z["cptr"]; gb = z["gb"]
    is_gb = np.repeat(gb, np.diff(ptr))
    h_in = hyd_from_cells(N, z["cells"][~is_gb], z["cfr"][~is_gb])[:, :, 40].T
    h_gb = hyd_from_cells(N, z["cells"][is_gb], z["cfr"][is_gb])[:, :, 40].T
    img = np.ones(h_in.shape + (3,))
    for h, col in ((h_in, (0.32, 0.32, 0.31)), (h_gb, (0.92, 0.41, 0.20))):
        a = np.clip(h, 0, 1)[..., None]
        img = img * (1 - a) + a * np.array(col)
    ax.imshow(img, extent=(0, L, L, 0))
    ax.set_xlabel("TD, мкм"); ax.set_ylabel("ND (r), мкм")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)


section(fig.add_subplot(gs[1, 0]), "gb_facet_cross", 0, "межзёренные (оранжевые) + в зерне (серые), 0 МПа")
section(fig.add_subplot(gs[1, 1]), "gb_facet_cross", 250, "то же, 250 МПа: сечение r–θ")
section(fig.add_subplot(gs[1, 2]), "gb_grain", 250, "межзёренные в осях зерна, 250 МПа")
fig.suptitle("3D: межзёренные отрезки на гранях зёрен и переход отрезка через границу (β 0.12, потолок 90 — как в 2D)",
             x=0.01, y=0.96, ha="left", fontsize=12.5, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig13_gb3d.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
