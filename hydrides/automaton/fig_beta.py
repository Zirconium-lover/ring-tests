"""Рис. 15: перекалибровка β и потолка ближнего поля (2D) — RHF по изображению против Lepine и доля длины
под 45–135° против ступеньки Cinbiz 2016. python fig_beta.py папка [папка ...]"""
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
BLUE, ORANGE, GREEN, VIOLET, INK, MUTED, BG, GRID = ("#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc7",
                                                     "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
runs = defaultdict(lambda: defaultdict(list))
for d in sys.argv[1:]:
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        runs[(m["beta"], m["sigma_cap"], bool(m.get("screen", False)))][int(m["sigma_app"])].append(m)
VAR = [((0.12, 90.0, False), BLUE, "β 0.12, потолок 90 (остаётся)"), ((0.6, 90.0, False), ORANGE, "β 0.6, потолок 90"),
       ((0.6, 220.0, False), VIOLET, "β 0.6, потолок 220"), ((0.3, 90.0, True), GREEN, "β 0.3, потолок 90, экранирование по МКЭ")]
cinbiz = lambda s: 1.0 / (1.0 + np.exp(-np.log(19.0) / 15.0 * (s - 160.0)))
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=BG)
for ax, key, title in ((axs[0], "RHF_image", "RHF по изображению (как у Lepine)"),
                       (axs[1], "Fn45_image", "доля длины на снимке под 45–135°, масштаб 3 мкм (как у Cinbiz)")):
    for k, c, lab in VAR:
        if k not in runs:
            continue
        s = sorted(runs[k])
        y = [np.nanmean([m[key] for m in runs[k][q]]) for q in s]
        e = [np.nanstd([m[key] for m in runs[k][q]]) for q in s]
        ax.errorbar(s, y, yerr=e, color=c, marker="o", ms=4, lw=1.8, capsize=2, label=lab)
    if key == "RHF_image":
        for q, (a, b) in {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}.items():
            ax.plot([q], [a], "ks", ms=7, mfc="none"); ax.plot([q], [b], "k^", ms=7, mfc="none")
        ax.plot([], [], "ks", mfc="none", label="Lepine, MATLAB"); ax.plot([], [], "k^", mfc="none", label="Lepine, HAPPy")
    else:
        x = np.linspace(0, 260, 300)
        ax.plot(x, cinbiz(x), color=INK, lw=1.2, ls="--", label="Cinbiz 2016, одноосно (0 при 145, ~1 при 177)")
    ax.axvspan(145, 177, color=ORANGE, alpha=0.10, lw=0)
    ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("доля"); ax.set_ylim(0, 1.02)
    ax.set_xlim(-5, 260); ax.grid(color=GRID)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Перекалибровка β и потолка ближнего поля (2D, ℓ 35 мкм, две затравки): у модели переход пологий и начинается с 25–50 МПа, порога нет",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig15_beta.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
