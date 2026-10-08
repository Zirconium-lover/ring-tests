"""Рисунок: кольца Э635 целиком против табл. 1 и 2 Плясова и др. (2023).
Сверху — F_l по толщине стенки (модель: 12 слоёв и трети; опыт: трети, табл. 1) и окружное напряжение при 400 °C;
полоса — граница зон по табл. 2. Снизу — модельный шлиф (внутренняя поверхность слева).

python fig_e635_ring.py папка_с_расчётами
"""
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
sys.path.insert(0, HERE)
from e635_plyasov import cases  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
CAT = {5.0: "#2a78d6", 8.0: "#eb6834"}
T_MM = 0.85
# табл. 1: F_l по третям (наружная, средняя, внутренняя); табл. 2: граница от внутренней поверхности, мм
EXP = {"S1_0": (0.39, 0.10, 0.01), "S1_90": (0.10, 0.69, 0.89),
       "S2_0": (0.35, 0.11, 0.04), "S2_90p": (0.15, 0.47, 0.71)}
BEFORE = {"S1": (0.125, 0.097, 0.097), "S2": (0.145, 0.089, 0.078)}
BOUND = {"S1_0": (0.57, 0.04), "S1_90": (0.50, 0.04), "S2_0": (0.51, 0.04), "S2_90p": (0.70, 0.04)}
TITLE = {"S1_0": "200 Н, 152 ppm, участок 0°", "S1_90": "200 Н, 152 ppm, участок 90°",
         "S2_0": "350 Н, 168 ppm, участок 0°", "S2_90p": "350 Н, 168 ppm, участок 90°"}
PROF = {c["ring"]: c["sigma_prof"] for c in cases("ring")}

rows = {}
for f in glob.glob(os.path.join(D, "*.json")):
    m = json.load(open(f))
    if "ring" in m:
        rows[(m["ring"], float(m["bias_dT"]))] = (m, f[:-5] + ".npz")

sites = list(EXP)
fig = plt.figure(figsize=(16, 6.4))
gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1.0], hspace=0.35, wspace=0.32)
x3 = np.array([5, 3, 1]) / 6 * T_MM                       # центры третей от внутренней поверхности: нар, ср, вн
for j, site in enumerate(sites):
    ax = fig.add_subplot(gs[0, j])
    b0, e = BOUND[site]
    ax.axvspan(b0 - e, b0 + e, color=GRID, zorder=0)
    ax.text(b0, 1.04, "граница\n(табл. 2)", ha="center", va="bottom", fontsize=7, color=MUTED)
    for k in range(3):                                    # трети опыта — ступеньки
        lo, hi = (2 - k) / 3 * T_MM, (3 - k) / 3 * T_MM
        ax.plot([lo, hi], [EXP[site][k]] * 2, color=INK, lw=2.5, solid_capstyle="butt",
                label="опыт, трети (табл. 1)" if k == 0 else None)
        ax.plot([lo, hi], [BEFORE[site[:2]][k]] * 2, color=MUTED, lw=1, ls=":",
                label="до опыта" if k == 0 else None)
    for b, col in CAT.items():
        if (site, b) not in rows:
            continue
        m = rows[(site, b)][0]
        pr = np.array(m["F_l_5_prof"], float)
        n = len(pr)
        xc = (n - 0.5 - np.arange(n)) / n * T_MM            # слой i от наружной → расстояние от внутренней
        ax.plot(xc, pr, "-", color=col, lw=1.3, alpha=0.8)
        ax.plot(x3, [m[f"F_l_5_{s}"] for s in ("out", "mid", "in")], "o", color=col, ms=7, mec="white",
                label=f"модель, фора {b:g} °C")
    ax.set_xlim(0, T_MM); ax.set_ylim(0, 1.15)
    ax.set_title(TITLE[site], loc="left", fontsize=10, color=INK)
    ax.set_xlabel("расстояние от внутренней поверхности, мм", color=MUTED, fontsize=8)
    if j == 0:
        ax.set_ylabel("F_l (гидриды > 5 мкм)", color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
    a2 = ax.twinx()
    yp, sp_ = np.array(PROF[site], float).T
    xx = np.linspace(0, T_MM, 200)
    a2.plot(xx, np.interp(1 - xx / T_MM, yp, sp_), color="#9a9893", lw=1, ls="--")
    a2.axhline(45, color="#9a9893", lw=0.6, ls=":")
    a2.set_ylim(-250, 250)
    a2.tick_params(labelsize=7, colors="#9a9893")
    if j == 3:
        a2.set_ylabel("σθ при 400 °C, МПа (штрих)", color="#9a9893", fontsize=8)
    if j == 0:
        ax.legend(frameon=False, fontsize=7, loc="center left")
    # модельный шлиф: фора 5 °C, внутренняя поверхность слева
    axi = fig.add_subplot(gs[1, j])
    key = (site, 5.0) if (site, 5.0) in rows else next((k for k in rows if k[0] == site), None)
    if key is not None:
        z = np.load(rows[key][1])
        if "hyd" in z.files:
            h = np.asarray(z["hyd"], float)[::-1].T            # строки — окружность, столбцы — толщина от внутренней
            k = 3                                              # максимум по блокам 3×3: тонкие пластинки не пропадают
            h = h[: h.shape[0] // k * k, : h.shape[1] // k * k].reshape(h.shape[0] // k, k, h.shape[1] // k, k).max((1, 3))
            axi.imshow(1 - 0.85 * np.clip(h / max(h.max(), 1e-9) * 1.5, 0, 1), cmap="gray", vmin=0, vmax=1,
                       extent=(0, T_MM, 0, h.shape[0] * k * float(z["dx"]) / 1000), aspect="auto",
                       interpolation="antialiased")
        axi.set_title(f"модель, фора {key[1]:g} °C", loc="left", fontsize=8, color=MUTED)
    axi.set_xlim(0, T_MM)
    axi.set_yticks([]); axi.tick_params(labelsize=7)
fig.suptitle("Кольца Э635 (Плясов и др. 2023, метод 1): доля радиальных гидридов по толщине стенки",
             x=0.01, ha="left", fontsize=11, color=INK)
out = os.path.join(HERE, "figs", "fig26_e635_ring.png")
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print("→", out)
for (site, b), (m, _) in sorted(rows.items()):
    print(site, b, "трети:", " ".join(f"{m[f'F_l_5_{s}']:.2f}" for s in ("out", "mid", "in")),
          " опыт:", " ".join(f"{v:.2f}" for v in EXP[site]), " весь:", f"{m['F_l_5']:.2f}",
          " T_first", m.get("T_first"), " GB", f"{m.get('GB_frac', np.nan):.2f}")
