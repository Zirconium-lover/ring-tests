"""Рис. 12: 3D с кристаллографией — отрезки макрогидрида на {10-17} против дисков (те же β 0.12 и потолок 90 МПа,
что в 2D; куб 64 мкм). python fig_seg3d.py папка"""
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
from sec3d import hyd_from_cells, hyd_of, packets3d  # noqa: E402

D = sys.argv[1]
BLUE, ORANGE, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}
L, N, DX = 64.0, 128, 0.5


def trace_dev(nv):
    d = np.cross(nv, [0, 0, 1.0]); s = np.linalg.norm(d, axis=1)
    return np.degrees(np.arccos(np.clip(np.abs(d[:, 0]) / np.maximum(s, 1e-9), 0, 1))), s


runs = defaultdict(lambda: defaultdict(list))
for f in glob.glob(D + "/*.json"):
    m = json.load(open(f))
    if m.get("img_method") != "voxel":
        continue
    runs[m["shape"]][int(m["sigma_app"])].append((m, f[:-5]))

fig = plt.figure(figsize=(17, 11), facecolor=BG)
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.27)
ax = fig.add_subplot(gs[0, 0])
for shape, c, lab in (("disc", MUTED, "диски в базисе"), ("segment", BLUE, "отрезки {10-17}")):
    s = sorted(runs[shape])
    for key, ls, nm in (("RHF_trace", "-", "по следам"), ("RHF_image", "--", "по изображению")):
        v = [np.mean([m[key] for m, _ in runs[shape][q]]) for q in s]
        e = [np.std([m[key] for m, _ in runs[shape][q]]) for q in s]
        ax.errorbar(s, v, yerr=e, color=c, ls=ls, marker="o", ms=4, lw=1.8, capsize=2, label=f"{lab}, {nm}")
for q, (a, b) in LEP.items():
    ax.plot([q], [a], "ks", ms=7, mfc="none"); ax.plot([q], [b], "k^", ms=7, mfc="none")
ax.plot([], [], "ks", mfc="none", label="Lepine, MATLAB"); ax.plot([], [], "k^", mfc="none", label="Lepine, HAPPy")
ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("RHF"); ax.set_ylim(0, 0.9); ax.grid(color=GRID)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("RHF (β 0.12, потолок 90 — как в 2D, без подгонки)", loc="left", fontsize=10.5, color=INK)

ax = fig.add_subplot(gs[0, 1])
bins = np.arange(0, 91, 5)
for shape, c, lab in (("disc", MUTED, "диски"), ("segment", BLUE, "отрезки {10-17}")):
    devs, ws = [], []
    for m, fn in runs[shape][250]:
        z = np.load(fn + ".npz")
        dv, sn = trace_dev(z["n"])
        devs.append(dv); ws.append((z["area"] if "area" in z else np.pi * z["R"] ** 2) * sn)
    dv, w = np.concatenate(devs), np.concatenate(ws)
    ax.hist(dv, bins=bins, weights=w / w.sum(), histtype="step", lw=2, color=c,
            label=f"{lab}: средний {np.average(dv, weights=w):.0f}°")
ax.axvline(47.7, color=ORANGE, lw=1.5, ls="--"); ax.text(46.5, ax.get_ylim()[1] * 0.04, "FIB-SEM Lepine:\nрадиальный\nпакет 47.7°", fontsize=8.5, color=INK, ha="right")
ax.set_xlabel("угол следа пластинки к TD, град"); ax.set_ylabel("доля площади пластинок"); ax.grid(color=GRID)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Наклон пластинок при 250 МПа", loc="left", fontsize=10.5, color=INK)

ax = fig.add_subplot(gs[0, 2])
for shape, c, lab in (("disc", MUTED, "диски"), ("segment", BLUE, "отрезки {10-17}")):
    pts = []
    for m, fn in runs[shape][0]:
        z = np.load(fn + ".npz")
        hyd = hyd_of(z, L, DX)
        pk = packets3d(hyd, DX)
        tot = hyd.sum() * DX ** 3
        pts += [(p["ext_L"], p["ext_ND"]) for p in pk if p["vol"] < 0.25 * tot and p["ext_TD"] > p["ext_ND"]]
    if pts:
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=c, alpha=0.7, label=f"{lab}: медианы {np.median(pts[:, 0]):.0f} × {np.median(pts[:, 1]):.1f} мкм")
ax.add_patch(plt.Rectangle((7, 3.8), 26, 1.1, color=ORANGE, alpha=0.25, lw=0))
ax.text(20, 1.0, "оранжевая полоса — FIB-SEM Lepine:\nпо оси ≥ 7–12.6 мкм (обрезано объёмом),\nтолщина по ND 3.8–4.9 мкм", fontsize=8.5, color=INK)
ax.set_xlabel("протяжённость пакета вдоль оси трубы, мкм"); ax.set_ylabel("толщина пакета по ND, мкм")
ax.set_xlim(0, 40); ax.set_ylim(0, 17); ax.grid(color=GRID); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.set_title("Окружные пакеты в 3D при 0 МПа (зазор 1 мкм)", loc="left", fontsize=10.5, color=INK)

for k, (s, sec) in enumerate(((0, "rt"), (250, "rt"), (250, "tl"))):
    ax = fig.add_subplot(gs[1, k])
    m, fn = runs["segment"][s][0]
    z = np.load(fn + ".npz")
    hyd = hyd_from_cells(N, z["cells"], z["cfr"])
    if sec == "rt":
        ax.imshow(hyd[:, :, 40].T, cmap="gray_r", extent=(0, L, L, 0), vmin=0, vmax=1)
        ax.set_xlabel("TD, мкм"); ax.set_ylabel("ND (r), мкм")
        ax.set_title(f"отрезки {{10-17}}, {s} МПа: сечение r–θ", loc="left", fontsize=10.5, color=INK)
    else:
        ax.imshow(hyd[:, 64, :].T, cmap="gray_r", extent=(0, L, L, 0), vmin=0, vmax=1)
        ax.set_xlabel("TD, мкм"); ax.set_ylabel("ось трубы, мкм")
        ax.set_title(f"отрезки {{10-17}}, {s} МПа: сечение TD – ось трубы", loc="left", fontsize=10.5, color=INK)
fig.suptitle("3D с кристаллографией: единица — отрезок макрогидрида в зерне на {10-17} (несоответствие δ в осях кристалла)",
             x=0.01, y=0.96, ha="left", fontsize=12.5, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig12_seg3d.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
