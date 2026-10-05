"""Рисунки по МКЭ одной пластинки: сверка с Фурье, влияние пластичности, карты выгоды.
Запуск: python fig_fe.py папка_с_fields*.npz"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Ellipse
from scipy import ndimage as ndi
from gmaps import gmap

D = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
RAMP = {150: "#6da7ec", 250: "#2a78d6", 400: "#104281"}
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#9cc3ef", "#efeeea", "#f4b49a", ORANGE])
DIV.set_bad("#b9b6ad")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
z = np.load(os.path.join(D, "fields.npz"))
zs = np.load(os.path.join(D, "fields_sens.npz"))
zc = np.load(os.path.join(D, "fields_check.npz"))
g = z["x"]; X, Y = np.meshgrid(g, g)
i0 = np.argmin(np.abs(g - 0.05)); j0 = np.argmin(np.abs(g - 0.05))
A = 2.5


def fe(src, name):
    return {k: src[f"{name}_{k}"] for k in ("sxx", "syy", "sxy", "szz")}


def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)


# ---------------------------------------------------------------- 1. сверка упругого решения
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=BG)
fx, fy = z["fft0.1_x"], z["fft0.1_y"]
jx = fx > A; iy0 = np.argmin(np.abs(fy - 0.05))
x4, y4 = z["fft0.4_x"], z["fft0.4_y"]; iy4 = np.argmin(np.abs(y4 - 0.2)); jx4 = x4 > A + 0.2
ax = axs[0]
ax.plot(fx[jx] - A, z["fft0.1_syy"][iy0, jx], color=BLUE, lw=2, label="Фурье, клетка 0.1 мкм")
ax.plot(x4[jx4] - A, z["fft0.4_syy"][iy4, jx4], "o", color=ORANGE, ms=5, label="Фурье, клетка 0.4 мкм (как в автомате)")
sel = g > A
ax.plot(g[sel] - A, z["el_syy"][i0, sel], "--", color=INK, lw=1.6, label="МКЭ (CalculiX), упруго")
ax.set_xlim(0, 9); ax.set_ylim(-100, 1500)
ax.set_xlabel("расстояние от кончика по оси пластинки, мкм"); ax.set_ylabel("σ_yy (по нормали к пластинке), МПа")
ax.set_title("а) перед кончиком", loc="left", color=INK); style(ax)
ax.legend(frameon=False, fontsize=9)
ax = axs[1]
jc = np.argmin(np.abs(fx - 0.05)); iy = fy > 0.3
ax.plot(fy[iy], z["fft0.1_syy"][iy, jc], color=BLUE, lw=2)
ax.plot(fy[iy], z["fft0.1_sxx"][iy, jc], color=BLUE, lw=2)
jc4 = np.argmin(np.abs(x4 - 0.2)); iy4b = y4 > 0.4
ax.plot(y4[iy4b], z["fft0.4_syy"][iy4b, jc4], "o", color=ORANGE, ms=5)
ax.plot(y4[iy4b], z["fft0.4_sxx"][iy4b, jc4], "o", color=ORANGE, ms=5)
s2 = g > 0.3
ax.plot(g[s2], z["el_syy"][s2, j0], "--", color=INK, lw=1.6); ax.plot(g[s2], z["el_sxx"][s2, j0], "--", color=INK, lw=1.6)
ax.text(6.0, 140, "σ_xx", color=INK); ax.text(6.0, -200, "σ_yy", color=INK)
ax.set_xlim(0, 9); ax.set_xlabel("расстояние от середины пластинки по нормали, мкм"); ax.set_ylabel("напряжение, МПа")
ax.set_title("б) над серединой пластинки", loc="left", color=INK); style(ax)
fig.suptitle("Упругое поле одной пластинки (5 × 0.6 мкм, несоответствие 7.2 / 4.6 %): Фурье-ядро автомата совпадает с МКЭ",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig_fe1_check.png"), dpi=130, facecolor=BG)

# ---------------------------------------------------------------- 2. пластичность перед кончиком и максимум выгоды
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), facecolor=BG, gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]
sel = g > A
ax.plot(g[sel] - A, z["el_syy"][i0, sel], "--", color=ORANGE, lw=1.6)
ax.text(0.55, 1050, "упруго", color=INK, fontsize=9)
for sy in (150, 250, 400):
    yv = z[f"pl{sy}_syy"][i0, sel]
    ax.plot(g[sel] - A, yv, color=RAMP[sy], lw=2)
    k = np.argmax(yv); ax.plot(g[sel][k] - A, yv[k], "o", color=RAMP[sy], ms=6)
    ax.text(g[sel][k] - A + 0.15, yv[k] + 25, f"σ_y = {sy}", color=INK, fontsize=9)
ax.set_xlim(0, 9.5); ax.set_ylim(-700, 1200)
ax.set_xlabel("расстояние от кончика по оси пластинки, мкм"); ax.set_ylabel("σ_yy, МПа")
ax.set_title("а) σ_yy перед кончиком", loc="left", color=INK); style(ax)
ax.text(2.6, -480, "у кончика — сжатие,\nрастяжение уходит вперёд", fontsize=9, color=MUTED)


def gmax(src, name, psi=0):
    gm = gmap(**fe(src, name), psi_deg=psi)
    forbid = ndi.binary_dilation(src[f"{name}_plate"], iterations=4)
    gm = np.where(forbid, np.nan, gm)
    k = np.nanargmax(gm); i, j = np.unravel_index(k, gm.shape)
    return gm[i, j], abs(X[i, j]) - A


ax = axs[1]
sys_ = np.array([150, 250, 400])
vals = [gmax(z, f"pl{s}") for s in sys_]
ax.plot(sys_, [v for v, _ in vals], "o-", color=BLUE, lw=2, ms=7)
for s, (v, d), (dx_, dy_) in zip(sys_, vals, ((4, -62), (8, -22), (8, -22))):
    ax.text(s + dx_, v + dy_, f"{d:.1f} мкм\nза кончиком", fontsize=8.5, color=MUTED)
vp, dp = gmax(zc, "perf_pl250")
ax.plot([250], [vp], "s", color=BLUE, ms=7, mfc=BG, mew=1.8)
ax.text(258, vp + 4, "без упрочнения", fontsize=8.5, color=MUTED)
for v, lab in ((gmax(zs, "lens_pl250")[0], "линза"), (gmax(zs, "thin_pl250")[0], "толщина 0.3")):
    ax.plot([250], [v], "D", color=AQUA, ms=5)
ax.text(186, 168, "линза,\nтолщина 0.3", fontsize=8.5, color=MUTED, ha="right")
xx = np.array([100, 450])
ax.plot(xx, 0.73 * xx, color=MUTED, lw=0.9, ls=":")
ax.text(420, 0.73 * 420 + 12, "0.73 σ_y", fontsize=9, color=MUTED, ha="right")
ax.axhline(90, color=ORANGE, lw=1.4)
ax.text(445, 74, "σ_cap автомата = 90 МПа", fontsize=9, color=INK, ha="right")
ax.set_xlim(100, 450); ax.set_ylim(0, 340)
ax.set_xlabel("предел текучести матрицы σ_y, МПа"); ax.set_ylabel("max g соосной пластинки, МПа")
ax.set_title("б) потолок выгоды ≈ 0.7–0.9 σ_y", loc="left", color=INK); style(ax)
fig.suptitle("Пластичность матрицы вокруг пластинки (МКЭ, плоская деформация, 7.2 / 4.6 %)", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig_fe2_plastic.png"), dpi=130, facecolor=BG)

# ---------------------------------------------------------------- 3. карты выгоды
W = 9.0
cols = [("автомат: Фурье, клетка 0.4 мкм,\nобрезка ±90 МПа", "auto"), ("МКЭ, упруго", "el"),
        ("МКЭ, σ_y = 250 МПа", "pl250"), ("МКЭ, σ_y = 250 МПа,\nпластинка-линза", "lens_pl250")]
psis = (0, 60, 90)
fig, axs = plt.subplots(3, 4, figsize=(13.2, 9.6), facecolor=BG)
for r, psi in enumerate(psis):
    for c, (title, key) in enumerate(cols):
        ax = axs[r, c]
        if key == "auto":
            S = {k: z[f"fft0.4_{k}"] for k in ("sxx", "syy", "sxy", "szz")}
            gm = np.clip(gmap(**S, psi_deg=psi), -90, 90)
            occ = ndi.binary_dilation(z["fft0.4_hyd"] > 0.2, iterations=1)
            gm = np.where(occ, np.nan, gm)
            x4, y4 = z["fft0.4_x"], z["fft0.4_y"]
            ext = (x4[0] - 0.2, x4[-1] + 0.2, y4[0] - 0.2, y4[-1] + 0.2)
        else:
            src = z if key in ("el", "pl250") else zs
            gm = gmap(**fe(src, key), psi_deg=psi)
            gm = np.where(ndi.binary_dilation(src[f"{key}_plate"], iterations=1), np.nan, gm)
            ext = (g[0] - 0.05, g[-1] + 0.05, g[0] - 0.05, g[-1] + 0.05)
        im = ax.imshow(gm, origin="lower", extent=ext, cmap=DIV, vmin=-200, vmax=200, interpolation="nearest")
        if key == "lens_pl250":
            ax.add_patch(Ellipse((0, 0), 2 * A, 0.6, fc=INK, ec=INK))
        else:
            ax.add_patch(Rectangle((-A, -0.3), 2 * A, 0.6, fc=INK, ec=INK))
        # маленький значок ориентации новой пластинки
        p = np.radians(psi); L = 1.3
        ax.plot([-7.2 - L / 2 * np.cos(p), -7.2 + L / 2 * np.cos(p)], [7.0 - L / 2 * np.sin(p), 7.0 + L / 2 * np.sin(p)],
                color=INK, lw=2.4, solid_capstyle="round")
        ax.set_xlim(-W, W); ax.set_ylim(-W, W); ax.set_aspect("equal")
        ax.set_xticks([-8, -4, 0, 4, 8]); ax.set_yticks([-8, -4, 0, 4, 8])
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
        if r == 0:
            ax.set_title(title, loc="left", fontsize=10, color=INK)
        if c == 0:
            ax.set_ylabel(f"новая пластинка ψ = {psi}°\n\ny, мкм")
        if r == 2:
            ax.set_xlabel("x, мкм")
cb = fig.colorbar(im, ax=axs, shrink=0.55, pad=0.02)
cb.set_label("выгода зарождения g = σ:ε*/ε_n, МПа (оранжевое — выгодно)")
cb.outline.set_edgecolor(MUTED)
fig.suptitle("Где выгодно зародиться следующей пластинке рядом с уже выросшей (чёрная, 5 мкм)", x=0.01, ha="left",
             fontsize=12, color=INK)
fig.text(0.01, 0.01, "Серое — сама пластинка и запретная полоса вокруг неё. Значок в углу — ориентация новой пластинки. "
         "Шкала обрезана на ±200 МПа; упругий максимум у кончика — до 870 МПа.", fontsize=9, color=MUTED)
fig.savefig(os.path.join(FIG, "fig_fe3_gmaps.png"), dpi=110, facecolor=BG, bbox_inches="tight")
print("ok")
