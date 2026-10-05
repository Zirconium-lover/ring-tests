"""Рис. 14: экранирование приложенного напряжения у гидридов (по МКЭ) в 2D-автомате.
python fig_screen.py папка"""
import os
import sys
import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
runs = defaultdict(lambda: defaultdict(list))
for f in glob.glob(D + "/*.json"):
    m = json.load(open(f))
    runs[m["var"]][m["sigma"]].append(m)
VAR = [("off", MUTED, "без экранирования"), ("half", ORANGE, "экранирование, зона ×0.5"), ("on", BLUE, "экранирование по МКЭ")]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=BG)
for ax, key, title in ((axs[0], "RHF_image", "RHF по изображению (как у Lepine)"),
                       (axs[1], "RHF45", "RHF по следам пластинок, доля длины под 45–135° (как у Cinbiz)")):
    for v, c, lab in VAR:
        s = sorted(runs[v])
        y = [np.mean([m[key] for m in runs[v][q]]) for q in s]
        e = [np.std([m[key] for m in runs[v][q]]) for q in s]
        ax.errorbar(s, y, yerr=e, color=c, marker="o", ms=4, lw=1.8, capsize=2, label=lab)
    ax.axvspan(145, 177, color=ORANGE, alpha=0.12, lw=0)
    ax.text(147, 0.03, "Cinbiz, одноосно:\nRHF от 0 до ~1\nмежду 145 и 177 МПа", fontsize=8.5, color=INK)
    if key == "RHF_image":
        for q, (a, b) in {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}.items():
            ax.plot([q], [a], "ks", ms=7, mfc="none"); ax.plot([q], [b], "k^", ms=7, mfc="none")
        ax.plot([], [], "ks", mfc="none", label="Lepine, MATLAB"); ax.plot([], [], "k^", mfc="none", label="Lepine, HAPPy")
    ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("RHF"); ax.set_ylim(0, 1); ax.grid(color=GRID)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Экранирование приложенного напряжения в пластической зоне у гидридов (2D, β 0.12, потолок 90, ℓ 35 мкм, две затравки)",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig14_screen.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
