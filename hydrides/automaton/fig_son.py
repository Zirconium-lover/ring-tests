"""Рис. 19: расчёт под условия Son и др. 2026 (Zr-Nb, 187 ppm, 400 °C, 0.3 °C/мин) против их опыта —
RHF по снимку и доля межзёренных гидридов. python fig_son.py папка [папка ...] (son_run.py)"""
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
R = defaultdict(lambda: defaultdict(list))
for d in sys.argv[1:]:                      # вариант — из первой папки, где он есть
    got = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        got[m["variant"]][int(m["sigma_app"])].append(m)
    for v, by in got.items():
        if v not in R:
            R[v] = by
VAR = [("viz5", BLUE, "фора зёрен 5 °C (Vizcaíno, Zr-2.5Nb) — без подгонки"),
       ("viz5_s2", GREEN, "то же + разброс по зёрнам 2 °C (старые остатки)"),
       ("viz5_s3", VIOLET, "то же + разброс по зёрнам 3 °C (старые остатки)"),
       ("zry10.9", MUTED, "фора 10.9 °C (калибровка на Zry-4, Cinbiz)")]
SON = {"CWSR": {0: 0.0, 83: 0.038, 96: 0.107, 146: 0.454}, "PRXA": {0: 0.0, 83: 0.115, 96: 0.263, 146: 0.540}}
fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), facecolor=BG)


def curve(ax, v, key, **kw):
    s = sorted(R[v])
    y = [np.nanmean([m[key] for m in R[v][q]]) for q in s]
    e = [np.nanstd([m[key] for m in R[v][q]]) for q in s]
    ax.errorbar(s, y, yerr=e, marker="o", ms=4, lw=1.8, capsize=2, **kw)


for v, c, lab in VAR:
    if v in R:
        curve(axs[0], v, "RHF_image", color=c, label=lab)
        curve(axs[1], v, "GB_frac", color=c, label=lab)
        curve(axs[2], v, "GB_frac_rad", color=c, label=lab)
for name, mk in (("CWSR", "s"), ("PRXA", "^")):
    q = sorted(SON[name])
    axs[0].plot(q, [SON[name][k] for k in q], color=INK, marker=mk, ms=8, mfc="none", lw=1.0, ls=":",
                label=f"Son 2026, {name} (опыт, PROPHET)")
axs[1].axhspan(0.90, 0.94, color=VIOLET, alpha=0.15, lw=0); axs[1].text(3, 0.95, "Son, CWSR: 92 %", fontsize=8.5, color=VIOLET)
axs[1].axhspan(0.65, 0.71, color=MUTED, alpha=0.12, lw=0); axs[1].text(3, 0.61, "Son, PRXA: 68 %", fontsize=8.5, color=MUTED)
axs[2].axhspan(0.59, 0.65, color=MUTED, alpha=0.12, lw=0); axs[2].text(3, 0.555, "Son, PRXA, радиальные: 62 %", fontsize=8.5, color=MUTED)
titles = ("RHF по снимку", "доля длины межзёренных — все", "доля межзёренных среди радиальных")
for ax, t in zip(axs, titles):
    ax.set_title(t, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylim(0, 1.02); ax.set_xlim(-5, 185)
    ax.grid(color=GRID); ax.set_facecolor(BG)
axs[0].legend(frameon=False, fontsize=8.3, loc="upper left")
fig.suptitle("Условия опыта Son и др. 2026: Zr-Nb, 187 ppm, 400 °C, 0.3 °C/мин; модель — 2D, только окружное "
             "напряжение (в опыте осевое ≈ σθ/2), зерно 2.5 × 4.5 мкм (в опыте 0.9 мкм)", x=0.01, ha="left",
             fontsize=11.5, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig19_son.png"), dpi=100, facecolor=BG, bbox_inches="tight")
for v, _, lab in VAR:
    if v in R:
        s = sorted(R[v])
        print(lab)
        for k in ("RHF_image", "Fn45_image", "RHF45_plates", "GB_frac", "GB_frac_rad"):
            print(f"  {k:13s}" + " ".join(f"{q}:{np.nanmean([m[k] for m in R[v][q]]):.2f}" for q in s))
