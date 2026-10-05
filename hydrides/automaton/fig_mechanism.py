"""Рис. 1: поле одной пластинки и выгодность зарождения следующей (по ориентациям)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from ca_hydride import Elastic, plate_eigen, Params, eigen_components_for, EPS_N, EPS_T

BLUE, ORANGE, INK, MUTED, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#9cc3ef", "#efeeea", "#f4b49a", ORANGE])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
p = Params(); dx = 0.2; shape = (300, 300)
el = Elastic(shape, dx, p.E, p.nu)
c = np.array([30.0, 30.0]); half = 2.5
ys, xs, fr, comps = plate_eigen(shape, dx, c, 0.0, half, p.h_um)
E = [np.zeros(shape) for _ in range(4)]
for arr, v in zip(E, comps): arr[np.ix_(ys, xs)] = fr * v
S11, S22, S12, S33 = el.stress(*E)
win = slice(int(18 / dx), int(42 / dx))
ext = (-12, 12, -12, 12)
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=BG, gridspec_kw=dict(wspace=0.12))
for ax, psi in zip(axs, (0, 45, 90)):
    e11, e22, e12 = eigen_components_for(np.radians(psi))
    g = (e11 * S11 + e22 * S22 + 2 * e12 * S12 + EPS_T * S33) / EPS_N
    g = np.where(E[0] > 0, np.nan, g)
    im = ax.imshow(np.clip(g[win, win], -200, 200), cmap=DIV, norm=TwoSlopeNorm(0, -200, 200), extent=ext)
    ax.plot([-half, half], [0, 0], color=INK, lw=3, solid_capstyle="butt")
    L = 2.2
    ax.plot([-L * np.cos(np.radians(psi)) + 8, L * np.cos(np.radians(psi)) + 8],
            [-L * np.sin(np.radians(psi)) + 8, L * np.sin(np.radians(psi)) + 8], color=INK, lw=2.4)
    ax.text(8, 11, "новая", ha="center", va="center", fontsize=9, color=INK)
    ax.set_title(f"{'абв'[psi // 45]}) новая пластинка под {psi}° к TD", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("TD (окружное), мкм")
    if psi == 0: ax.set_ylabel("ND (радиальное), мкм")
cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.015)
cb.set_label("выгода зарождения g / ε*ₙ, МПа (оранжевый — выгодно)")
fig.text(0.01, -0.02, "Чёрная полоса в центре — уже выросшая пластинка 5 мкм вдоль TD. Поле — микроупругость в пространстве Фурье, "
         "несоответствие δ-гидрида 7.2 % по нормали и 4.6 % в плоскости.", fontsize=9, color=MUTED)
fig.savefig("figs/fig1_mechanism.png", dpi=120, bbox_inches="tight", facecolor=BG)
# числа для текста
for psi in (0, 30, 45, 60, 90):
    e11, e22, e12 = eigen_components_for(np.radians(psi))
    g = (e11 * S11 + e22 * S22 + 2 * e12 * S12 + EPS_T * S33) / EPS_N
    g = np.where(E[0] > 0, -1e9, g)
    k = np.unravel_index(np.argmax(g[win, win]), g[win, win].shape)
    print(psi, "макс", round(g[win, win][k]), "МПа в точке (ND, TD) мкм:", round(-(k[0] * dx - 12), 1), round(k[1] * dx - 12, 1))
