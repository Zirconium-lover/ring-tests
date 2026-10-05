"""Калибровка 3D (быстрый движок, куб 64 мкм, две затравки): RHF по следам и по изображению сечений
против Lepine; вид сечений r–θ «как на шлифе» в 3D и в 2D.
python fig_calib3d.py папка_калибровки папка_2D(cal7)"""
import os
import sys
import json
import glob
import warnings
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sec3d import section_plates, render_plates, hs  # noqa: E402

D, D2 = sys.argv[1], sys.argv[2]
BLUE, ORANGE, GREEN, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#3b9a5a", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}
L = 64.0

g = defaultdict(lambda: defaultdict(list))
for f in glob.glob(D + "/*.json"):
    m = json.load(open(f))
    if m.get("size_um") != L or "R_min" in m:
        continue
    g[(m["beta"], m["sigma_cap"])][int(m["sigma_app"])].append((m["RHF_trace"], m["RHF_image"]))

SETS = [((0.12, 90.0), MUTED, "β 0.12, потолок 90 (параметры 2D)"),
        ((0.25, 65.0), BLUE, "β 0.25, потолок 65"),
        ((0.25, 45.0), ORANGE, "β 0.25, потолок 45")]

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.05), hspace=0.3, wspace=0.28)
for k, (key, title) in enumerate(((0, "RHF по следам пластинок в сечениях r–θ"), (1, "RHF по изображению сечений (как на шлифе)"))):
    ax = fig.add_subplot(gs[0, k])
    for pk, c, lab in SETS:
        s = sorted(g[pk]); v = [np.mean([x[key] for x in g[pk][q]]) for q in s]
        e = [np.std([x[key] for x in g[pk][q]]) for q in s]
        ax.errorbar(s, v, yerr=e, color=c, marker="o", ms=4, lw=1.8, capsize=2, label=lab)
    for q, (a, b) in LEP.items():
        ax.plot([q], [a], "ks", ms=7, mfc="none"); ax.plot([q], [b], "k^", ms=7, mfc="none")
    ax.plot([], [], "ks", mfc="none", label="Lepine, MATLAB"); ax.plot([], [], "k^", mfc="none", label="Lepine, HAPPy")
    ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("RHF"); ax.set_ylim(0, 0.9)
    ax.grid(color=GRID); ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    if k == 0:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
# карта ошибки по следам
ax = fig.add_subplot(gs[0, 2])
bs = sorted({k[0] for k in g}); cs = sorted({k[1] for k in g})
E = np.full((len(cs), len(bs)), np.nan)
for i, c in enumerate(cs):
    for j, b in enumerate(bs):
        d = g.get((b, c))
        if d and all(q in d for q in LEP):
            E[i, j] = np.sqrt(np.mean([(np.mean([x[0] for x in d[q]]) - np.mean(LEP[q])) ** 2 for q in LEP]))
im = ax.imshow(E, origin="lower", cmap="viridis_r", aspect="auto", vmin=0.05, vmax=0.3)
for i in range(len(cs)):
    for j in range(len(bs)):
        if np.isfinite(E[i, j]):
            ax.text(j, i, f"{E[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if E[i, j] > 0.17 else INK)
ax.set_xticks(range(len(bs)), [f"{b:g}" for b in bs]); ax.set_yticks(range(len(cs)), [f"{c:g}" for c in cs])
ax.set_xlabel("β, 1/МПа"); ax.set_ylabel("потолок ближнего поля, МПа")
ax.set_title("ошибка RHF по следам против Lepine (0/200/250)", loc="left", color=INK, fontsize=10.5)
fig.colorbar(im, ax=ax, shrink=0.8)


def show(ax, P, title, Lf):
    tile = int(np.ceil(128 / Lf))
    PP = np.concatenate([P + np.array([a * Lf, b * Lf, 0, 0]) for a in range(tile) for b in range(tile)]) if tile > 1 else P
    im = render_plates(PP, (Lf * tile, Lf * tile), um=0.65, h_um=0.6, blur_um=1.0, noise=0.03) / 255.0
    n = int(128 / 0.65)
    im = im[:n, :n]
    r, _ = hs.analyse("ca", um=0.65, g=im, field=True, scales_um=(3.0,), line_um=5.0, n_layers=3)
    ax.imshow(im, cmap="gray", extent=(0, 128, 128, 0), vmin=0, vmax=1)
    ax.set_title(f"{title}\nRHF этого изображения {r['scales'][0]['RHF_simon']:.2f}", loc="left", color=INK, fontsize=10)
    ax.set_xlabel("TD, мкм"); ax.set_ylabel("ND (r), мкм")


for k, s in enumerate((0, 250)):
    z = np.load(os.path.join(D, f"beta0.25_seed1_sigma_app{s}_sigma_cap45.0_size_um64.0.npz"))
    show(fig.add_subplot(gs[1, k]), section_plates(z["c"], z["n"], z["R"], 20.0, L), f"3D (β 0.25, потолок 45), {s} МПа: сечение z = 20 мкм", L)
z2 = np.load(glob.glob(os.path.join(D2, "beta0.12_capture_um35.0_seed1_sigma_app250_sigma_cap90.npz"))[0])
show(fig.add_subplot(gs[1, 2]), z2["plates"], "2D (калибровка), 250 МПа: участок поля 240 мкм", 240.0)
fig.suptitle("Калибровка 3D по Lepine: ориентацию отдельных пластинок подогнать можно, вид пакетов на сечении — нет",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig10_calib3d.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
