"""Рисунок: Э635 против Плясова и др. (2023). (а) кольца, постоянная нагрузка — подбор форы по порогу 45 ± 9 МПа;
(б) трубы под давлением — предсказание F_l(σmax) при разном водороде против рис. 6 статьи (точки сняты с графика)."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# рис. 6 Плясова и др. 2023 (сняты с графика, ±0.01)
EXP = {0: [0.09, 0.115, 0.12], 50: [0.165, 0.17, 0.18, 0.185],
       70: [0.26, 0.275, 0.29, 0.30, 0.33, 0.34, 0.35, 0.36, 0.37], 90: [0.36, 0.365, 0.37, 0.375, 0.38, 0.42],
       110: [0.28, 0.30, 0.305, 0.315, 0.325, 0.345, 0.41, 0.44],
       140: [0.30, 0.305, 0.31, 0.325, 0.33, 0.34, 0.35, 0.365, 0.375, 0.385, 0.395, 0.415, 0.42, 0.43, 0.445]}

rows = [json.load(open(f)) for f in glob.glob(os.path.join(D, "*.json"))]


def curve(sel, key="F_l_5"):
    by = {}
    for m in rows:
        if sel(m):
            by.setdefault(int(m["sigma_app"]), []).append(m[key])
    s = sorted(by)
    return np.array(s), np.array([np.nanmean(by[k]) for k in s])


fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# (а) кольца
ax[0].axvspan(36, 54, color=GRID, zorder=0)
ax[0].text(45, 1.02, "порог Плясова\n45 ± 9 МПа", ha="center", va="bottom", fontsize=8, color=MUTED)
for i, b in enumerate((1.0, 3.0, 5.0, 8.0)):
    x, y = curve(lambda m, b=b: m.get("sigma_T0", 0) == 0 and m["bias_dT"] == b)
    ax[0].plot(x, y, "-o", color=CAT[i], lw=2, ms=5, label=f"фора {b:g} °C")
ax[0].set_xlabel("окружное напряжение (постоянное), МПа", color=MUTED)
ax[0].set_ylabel("F_l — доля длины ±45° от радиуса (гидриды > 5 мкм)", color=MUTED)
ax[0].set_title("а) кольца, 160 ppm: подбор форы", loc="left", color=INK)
ax[0].set_ylim(0, 1.12)
ax[0].legend(frameon=False, fontsize=8, loc="lower right")

# (б) трубы под давлением
for s, v in EXP.items():
    ax[1].plot([s] * len(v), v, "^", mfc="none", mec=INK, ms=6, ls="none",
               label="Плясов 2023, рис. 6 (150–450 ppm)" if s == 0 else None)
for i, h in enumerate((150.0, 210.0, 300.0, 400.0)):
    x, y = curve(lambda m, h=h: m.get("sigma_T0", 0) == 400 and m["bias_dT"] == 8.0 and m["H_ppm"] == h)
    ax[1].plot(x, y, "-o", color=CAT[i], lw=2, ms=5, label=f"модель, фора 8 °C, {h:g} ppm")
x, y = curve(lambda m: m.get("sigma_T0", 0) == 400 and m["bias_dT"] == 17.0)
ax[1].plot(x, y, ":s", color=MUTED, lw=1.5, ms=4, label="фора Zry-4 (17 °C), 210 ppm")
ax[1].set_xlabel("σmax при 400 °C (дальше спадает ∝ T), МПа", color=MUTED)
ax[1].set_title("б) трубы под давлением: предсказание", loc="left", color=INK)
ax[1].set_ylim(0, 1.12)
ax[1].legend(frameon=False, fontsize=8, loc="upper left")
for a in ax:
    a.grid(color=GRID, lw=0.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "figs", "fig25_e635_plyasov.png")
fig.savefig(out, dpi=130, facecolor="white")
print("→", out)
