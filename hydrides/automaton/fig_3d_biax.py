"""Рис. 21: кинетический 3D-движок под двухосной нагрузкой — RHF в сечениях как на шлифах и порог против
Cinbiz и др. 2016 (155 / 110 / 75 МПа при σz/σθ = 0 / 0.57 / 0.83). python fig_3d_biax.py папка [подпись]"""
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
D = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
R = defaultdict(lambda: defaultdict(list))
for f in glob.glob(D + "/*.json"):
    m = json.load(open(f))
    R[round(m.get("sigma_axial", 0.0), 2)][int(m["sigma_app"])].append(m)
COL = {0.0: BLUE, 0.5: GREEN, 0.57: GREEN, 0.83: ORANGE, 1.0: VIOLET}
CIN = {0.0: 155.0, 0.57: 110.0, 0.83: 75.0}


def mean(b, s, k):
    return np.nanmean([m.get(k, np.nan) for m in R[b][s]])


def crossing(b, k, lvl):
    s = sorted(R[b]); y = [mean(b, q, k) for q in s]
    for (s0, y0), (s1, y1) in zip(zip(s, y), zip(s[1:], y[1:])):
        if y0 < lvl <= y1:
            return s0 + (lvl - y0) / (y1 - y0) * (s1 - s0)
    return np.nan


fig, axs = plt.subplots(1, 4, figsize=(21, 5.0), facecolor=BG)
for b in sorted(R):
    s = sorted(R[b]); c = COL.get(b, MUTED); lab = f"σz/σθ = {b:g}"
    axs[0].plot(s, [mean(b, q, "RHF_rt") for q in s], "o-", color=c, lw=1.8, ms=4, label=lab)
    axs[1].plot(s, [mean(b, q, "RHF_surf") for q in s], "o-", color=c, lw=1.8, ms=4, label=lab)
    axs[2].plot(s, [mean(b, q, "frac_nTD") for q in s], "o-", color=c, lw=1.8, ms=4, label=lab + ": нормаль по θ")
    axs[2].plot(s, [mean(b, q, "frac_nL") for q in s], "s--", color=c, lw=1.4, ms=4, label=lab + ": нормаль по оси")
for ax, t in zip(axs[:3], ("RHF в поперечном сечении r–θ (веса 0/0.5/1)",
                           "на плоскости поверхности: доля следов круче 45°",
                           "доля объёма гидрида по направлению нормали")):
    ax.set_title(t, loc="left", fontsize=10.2, color=INK); ax.set_ylim(0, 1.02)
    ax.set_xlabel("σθ (наибольшее главное), МПа"); ax.grid(color=GRID); ax.set_facecolor(BG)
axs[0].legend(frameon=False, fontsize=8.5, loc="upper left"); axs[2].legend(frameon=False, fontsize=7.5, loc="center left")
bb = sorted(R)
for k, c, lab in (("RHF_rt", BLUE, "модель: RHF r–θ = 0.5"), ("RHF_surf", ORANGE, "модель: поверхность, 0.1 (начало)")):
    lvl = 0.5 if k == "RHF_rt" else 0.1
    axs[3].plot(bb, [crossing(b, k, lvl) for b in bb], "o-", color=c, lw=1.8, ms=5, label=lab)
axs[3].plot(list(CIN), list(CIN.values()), "ks", ms=8, mfc="none", label="Cinbiz 2016 (начало выхода из плоскости)")
x = np.linspace(0, 1, 50)
axs[3].plot(x, 155.0 / (1 + x), color=MUTED, ls=":", lw=1.2, label="постоянное σ_h = (σθ + σz)/3 ≈ 52 МПа")
axs[3].set_xlabel("σz/σθ"); axs[3].set_ylabel("порог σθ, МПа"); axs[3].set_ylim(0, 260); axs[3].grid(color=GRID)
axs[3].set_title("порог от двухосности", loc="left", fontsize=10.2, color=INK); axs[3].legend(frameon=False, fontsize=8)
axs[3].set_facecolor(BG)
fig.suptitle("Кинетический 3D-движок: двухосная нагрузка и сечения как на шлифах " + TAG, x=0.01, ha="left",
             fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig21_3d_biax.png"), dpi=100, facecolor=BG, bbox_inches="tight")
for b in bb:
    s = sorted(R[b])
    print(f"σz/σθ = {b}")
    for k in ("RHF_rt", "RHF_rz", "RHF_surf", "frac_nTD", "frac_nND", "frac_nL", "T_first", "n_plates", "time_s"):
        print(f"  {k:9s}" + " ".join(f"{q}:{mean(b, q, k):.2f}" for q in s))
    print("  порог RHF_rt=0.5:", round(crossing(b, "RHF_rt", 0.5)), " поверхность 0.1:", round(crossing(b, "RHF_surf", 0.1)))
