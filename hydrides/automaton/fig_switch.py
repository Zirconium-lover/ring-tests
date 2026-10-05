"""Рис. 3: куда «тянет» следующую пластинку противоположного наклона — переключение при 45°."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ca_hydride import Elastic, plate_eigen, Params, eigen_components_for, EPS_N, EPS_T

BLUE, ORANGE, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
p = Params(); dx = 0.2; shape = (400, 400)
el = Elastic(shape, dx, p.E, p.nu)
c = np.array([40.0, 40.0]); half = 2.0
Y, X = np.mgrid[0:shape[0], 0:shape[1]] * dx
dyy, dxx = -(Y - c[0]), X - c[1]
R = np.hypot(dyy, dxx); A = np.degrees(np.arctan2(dyy, dxx)) % 180
psis = np.arange(5, 90, 5)
dirs, stren = [], []
for psi0 in psis:
    ys, xs, fr, comps = plate_eigen(shape, dx, c, np.radians(psi0), half, p.h_um)
    E = [np.zeros(shape) for _ in range(4)]
    for arr, v in zip(E, comps): arr[np.ix_(ys, xs)] = fr * v
    S = el.stress(*E)
    e11, e22, e12 = eigen_components_for(np.radians(-psi0))
    g = (e11 * S[0] + e22 * S[1] + 2 * e12 * S[2] + EPS_T * S[3]) / EPS_N
    m = (R > 4.0) & (R < 6.0) & (E[0] == 0)
    k = np.argmax(np.where(m, g, -1e9))
    dirs.append(A.flat[k] if A.flat[k] <= 90 else 180 - A.flat[k]); stren.append(g.flat[k])
dirs, stren = np.array(dirs), np.array(stren)
np.savetxt("figs/switch_curve.csv", np.c_[psis, dirs, stren], delimiter=",", header="psi0_deg,stack_dir_deg,g_max_MPa", comments="")
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=BG, gridspec_kw=dict(wspace=0.3, width_ratios=[1.25, 1]))
ax = axs[0]
ax.plot(psis, dirs, "o-", color=INK, lw=2, ms=5)
ax.axvline(45, color=GRID, lw=1.2, ls=(0, (4, 3)))
ax.fill_between([0, 45], 0, 90, color=BLUE, alpha=0.07, lw=0); ax.fill_between([45, 90], 0, 90, color=ORANGE, alpha=0.07, lw=0)
ax.text(20, 75, "цепочка\nвдоль дуги", color=BLUE, ha="center", fontsize=10)
ax.text(68, 15, "цепочка\nпо радиусу", color=ORANGE, ha="center", fontsize=10)
ax.set_xlim(0, 90); ax.set_ylim(0, 90); ax.set_xticks([0, 15, 30, 45, 60, 75, 90]); ax.set_yticks([0, 30, 45, 60, 90])
ax.set_xlabel("наклон пластинок ±ψ к дуге, °"); ax.set_ylabel("куда выгоднее следующая пластинка, ° от дуги")
ax.set_title("а) Направление цепочки из пластинок ±ψ (поле упругости)", loc="left", fontsize=10.5, color=INK)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax = axs[1]
for k, (psi, col, lab) in enumerate(((25, BLUE, "±25°: зигзаг вдоль дуги"), (65, ORANGE, "±65°: зигзаг по радиусу"))):
    t = np.radians(psi)
    pts = [np.array([0.0, 0.0])]
    for i in range(6):
        s = 1 if i % 2 == 0 else -1
        if psi < 45:
            d = np.array([np.cos(t), s * np.sin(t)])
        else:
            d = np.array([s * np.cos(t), np.sin(t)])
        pts.append(pts[-1] + d)
    pts = np.array(pts)
    off = np.array([0.3, 1.6]) if psi < 45 else np.array([6.6, 0.4])
    for i in range(6):
        ax.plot(*(np.vstack([pts[i], pts[i + 1]]) * 0.92 + off).T, color=col, lw=4, solid_capstyle="round")
    ax.text(*(off + (np.array([0, -0.9]) if psi < 45 else np.array([-1.6, -0.9]))), lab, color=INK, fontsize=9.5)
ax.set_xlim(0, 8.6); ax.set_ylim(-0.8, 6.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("б) Как это выглядит", loc="left", fontsize=10.5, color=INK)
fig.savefig("figs/fig3_switch.png", dpi=120, bbox_inches="tight", facecolor=BG)
print(np.c_[psis, dirs.round(0), stren.round(0)])
