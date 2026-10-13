"""Рисунки главы 4: ячейка ГПУ и гидридная пластинка, типы межфазных границ, краевая дислокация и её поле,
энергия границы зерна по Риду–Шокли."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from style import *


def f_hcp():
    """а) ячейка ГПУ α-Zr: базисная плоскость (0001), призматическая {10-10}, оси a1–a3 и c;
    б) пластинка δ-гидрида на плоскости, близкой к базисной: несоответствие в плоскости ε_a ≈ 4.6 %, по нормали ε_c ≈ 7.2 %."""
    fig = plt.figure(figsize=(W1, 2.9))
    a = fig.add_subplot(1, 2, 1, projection="3d")
    A, C = 1.0, 1.593
    ang = np.radians(np.arange(0, 360, 60) + 30)
    hx, hy = A * np.cos(ang), A * np.sin(ang)
    for z in (0, C):
        a.plot(np.r_[hx, hx[0]], np.r_[hy, hy[0]], z, color=SUB, lw=0.8)
    for x, y in zip(hx, hy):
        a.plot([x, x], [y, y], [0, C], color=SUB, lw=0.8)
    a.add_collection3d(Poly3DCollection([list(zip(hx, hy, np.zeros(6)))], facecolor=CIRC, alpha=0.25, edgecolor=CIRC))
    k = 0
    prism = [(hx[k], hy[k], 0), (hx[k + 1], hy[k + 1], 0), (hx[k + 1], hy[k + 1], C), (hx[k], hy[k], C)]
    a.add_collection3d(Poly3DCollection([prism], facecolor=RAD, alpha=0.22, edgecolor=RAD))
    atoms = [(x, y, z) for z in (0, C) for x, y in zip(np.r_[hx, 0], np.r_[hy, 0])]
    for t in (90, 210, 330):
        r = A / np.sqrt(3)
        atoms.append((r * np.cos(np.radians(t)), r * np.sin(np.radians(t)), C / 2))
    P = np.array(atoms)
    a.scatter(P[:, 0], P[:, 1], P[:, 2], s=38, color=METAL_E, edgecolor=INK, linewidth=0.5, depthshade=False)
    a.quiver(0, 0, C, 0, 0, 0.7, color=INK, arrow_length_ratio=0.2, lw=1.0)
    a.text(0.05, 0, C + 0.8, r"$c$", color=INK)
    for t, name in ((0, r"$a_1$"), (120, r"$a_2$"), (240, r"$a_3$")):
        v = np.array([np.cos(np.radians(t)), np.sin(np.radians(t))]) * 1.35
        a.quiver(0, 0, 0, v[0], v[1], 0, color=INK, arrow_length_ratio=0.12, lw=0.9)
        a.text(v[0] * 1.12, v[1] * 1.12, 0, name, color=INK)
    a.text(-1.9, 0.6, -0.2, r"$(0001)$", color=CIRC, fontsize=8.5)
    a.text(0.75, -0.95, C * 0.55, r"$\{10\bar{1}0\}$", color=RAD, fontsize=8.5)
    a.set_box_aspect((1, 1, 0.95)); a.view_init(elev=17, azim=-62); a.set_axis_off()
    a.set_title("а", loc="left", fontweight="bold")
    b = fig.add_subplot(1, 2, 2)
    b.add_patch(Rectangle((-3, -1.8), 6, 3.6, facecolor=METAL, edgecolor=METAL_E))
    for yy in np.linspace(-1.6, 1.6, 9):
        b.plot([-2.9, 2.9], [yy, yy], color=METAL_E, lw=0.5)
    b.add_patch(Rectangle((-1.8, -0.09), 3.6, 0.18, facecolor=HYD_F, edgecolor=CIRC, lw=1.0))
    b.text(0, 0.28, r"гидрид $\delta$ (ГЦК)", ha="center", color=CIRC, fontsize=8.5)
    b.annotate("", xy=(2.45, 0), xytext=(1.85, 0), arrowprops=dict(arrowstyle="->", color=LOAD, lw=1.4))
    b.annotate("", xy=(-2.45, 0), xytext=(-1.85, 0), arrowprops=dict(arrowstyle="->", color=LOAD, lw=1.4))
    b.text(2.5, -0.38, r"$\varepsilon_a \approx 4.6\,\%$", color=LOAD, fontsize=8.5, ha="center")
    b.annotate("", xy=(0.9, 0.85), xytext=(0.9, 0.12), arrowprops=dict(arrowstyle="->", color=RAD, lw=1.4))
    b.annotate("", xy=(0.9, -0.85), xytext=(0.9, -0.12), arrowprops=dict(arrowstyle="->", color=RAD, lw=1.4))
    b.text(1.05, 0.75, r"$\varepsilon_c \approx 7.2\,\%$", color=RAD, fontsize=8.5)
    b.annotate("", xy=(-2.7, 1.55), xytext=(-2.7, 0.75), arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    b.text(-2.55, 1.2, r"$c$", color=INK)
    b.text(0, -1.55, r"плоскости $(0001)$ $\alpha$-Zr", ha="center", color=SUB, fontsize=8)
    b.set_xlim(-3.1, 3.1); b.set_ylim(-1.9, 1.9); b.set_aspect("equal"); b.axis("off")
    b.set_title("б", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=0.5)
    save(fig, "f4_hcp")


def lattice(ax, xs, ys, **kw):
    X, Y = np.meshgrid(xs, ys)
    ax.plot(X.ravel(), Y.ravel(), "o", ms=3.6, **kw)


def f_interfaces():
    """Когерентная, полукогерентная (дислокации несоответствия) и некогерентная межфазные границы."""
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.2))
    d1, d2 = 1.0, 1.12
    for ax, kind in zip(axs, ("coh", "semi", "inc")):
        lattice(ax, np.arange(-6, 6.01, d1), np.arange(-5, -0.4, d1), color=METAL_E, mec=INK, mew=0.4)
        if kind == "coh":
            for j, y in enumerate(np.arange(0.0, 4.6, d2)):
                xs = np.arange(-6, 6.01, d1)
                ax.plot(xs, np.full_like(xs, y), "o", ms=3.6, color=HYD_F, mec=CIRC, mew=0.5)
            ax.text(0, -6.0, r"когерентная" "\n" r"$\gamma \sim 0.05$--$0.2$ Дж/м$^2$", fontsize=8, color=SUB, ha="center", va="top")
        elif kind == "semi":
            for y in np.arange(0.0, 4.6, d2):
                xs = np.arange(-5.6, 5.61, d2)
                ax.plot(xs, np.full_like(xs, y), "o", ms=3.6, color=HYD_F, mec=CIRC, mew=0.5)
            for xd in (-4.5, 4.5, 0.0):
                ax.text(xd, -0.55, r"$\perp$", color=RAD, fontsize=11, ha="center", va="center")
            ax.text(0, -6.0, r"полукогерентная" "\n" r"$0.2$--$0.5$ Дж/м$^2$", fontsize=8, color=SUB, ha="center", va="top")
        else:
            th = np.radians(27)
            g = np.arange(-8, 8.01, d2)
            X, Y = np.meshgrid(g, g)
            Xr = X * np.cos(th) - Y * np.sin(th); Yr = X * np.sin(th) + Y * np.cos(th)
            m = (Yr >= -0.1) & (Yr <= 4.6) & (np.abs(Xr) <= 6)
            ax.plot(Xr[m], Yr[m], "o", ms=3.6, color=HYD_F, mec=CIRC, mew=0.5)
            ax.text(0, -6.0, r"некогерентная" "\n" r"$0.5$--$1$ Дж/м$^2$", fontsize=8, color=SUB, ha="center", va="top")
        ax.axhline(-0.27, color=LOAD, lw=0.8, ls="--")
        ax.set_xlim(-6.4, 6.4); ax.set_ylim(-8.2, 5.0); ax.set_aspect("equal"); ax.axis("off")
    for ax, s in zip(axs, "абв"):
        ax.text(-6.3, 4.9, s, fontweight="bold", fontsize=10.5)
    fig.tight_layout(w_pad=0.3)
    save(fig, "f4_interfaces")


def f_dislocation():
    """а) краевая дислокация: решётка, искажённая полем Вольтерры (лишняя полуплоскость сверху);
    б) напряжение σ_xx вокруг неё: сжатие над плоскостью скольжения, растяжение под ней."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.75))
    nu = 0.34
    # построение Вольтерры: над плоскостью скольжения вставлена полуплоскость атомов (x = 0, y > 0); соседние
    # столбцы раздвинуты смещением u_x = b·atan2(x, −y)/2π (разрыв ±b/2 по полуплоскости, ноль глубоко внизу)
    cols = [i + 0.5 for i in range(-7, 7)] + [0.0]
    ys = np.arange(-6, 6.01, 1.0) + 0.5
    for xc in cols:
        yy = ys[ys > 0] if xc == 0.0 else ys
        xx = np.full_like(yy, xc)
        ux = 0.0 if xc == 0.0 else np.arctan2(xx, -yy) / (2 * np.pi)
        px = xx + ux
        a.plot(px, yy, color=METAL_E, lw=0.6, zorder=1)
        a.plot(px, yy, "o", ms=3.6, color=RAD if xc == 0.0 else METAL_E, mec=INK, mew=0.4, zorder=2)
    a.text(0.0, -0.05, r"$\perp$", color=RAD, fontsize=16, ha="center", va="center")
    a.text(0.25, 6.6, "лишняя полуплоскость", color=RAD, fontsize=7.5, ha="center")
    a.axhline(0, color=LOAD, lw=0.7, ls="--")
    a.text(6.8, -0.55, "плоскость скольжения", color=LOAD, fontsize=7.5, ha="right")
    a.set_aspect("equal"); a.axis("off"); a.set_xlim(-7.4, 7.4); a.set_ylim(-6.3, 7.0)
    label(a, "а", 0.02, 1.0)
    g = np.linspace(-5, 5, 401)
    X, Y = np.meshgrid(g, g)
    sxx = -Y * (3 * X ** 2 + Y ** 2) / (X ** 2 + Y ** 2 + 1e-9) ** 2
    lv = np.linspace(-1.5, 1.5, 13)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("d", [CIRC, "#ffffff", RAD])
    cs = b.contourf(X, Y, np.clip(sxx, -1.5, 1.5), levels=lv, cmap=cmap, extend="both")
    b.contour(X, Y, np.clip(sxx, -1.5, 1.5), levels=lv, colors=[SUB], linewidths=0.3)
    b.text(0, 0, r"$\perp$", color=INK, fontsize=16, ha="center", va="center")
    b.text(0, 3.3, "сжатие", color=CIRC, ha="center", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=0.6))
    b.text(0, -3.6, "растяжение", color=RAD, ha="center", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=0.6))
    b.set_aspect("equal"); b.set_xticks([]); b.set_yticks([])
    cb = fig.colorbar(cs, ax=b, shrink=0.85, ticks=[-1.5, 0, 1.5])
    cb.set_label(r"$\sigma_{xx}$ в единицах $\mu b/2\pi(1-\nu) r_0$", fontsize=8)
    label(b, "б", 0.02, 1.0)
    fig.tight_layout(w_pad=1.0)
    save(fig, "f4_dislocation")


def f_gb():
    """Энергия границы зерна от угла разориентации: формула Рида–Шокли для малых углов и плато для больших."""
    fig, ax = plt.subplots(figsize=(W2 + 0.4, 2.3))
    thm = 15.0
    th = np.linspace(0.05, 60, 600)
    t = np.minimum(th / thm, 1.0)
    g = t * (1 - np.log(t))
    ax.plot(th, g, color=INK)
    for cusp in (38.9, 53.0):        # «особые» границы с малой энергией (условно — положения провалов)
        m = np.abs(th - cusp) < 3
        ax.plot(th[m], g[m] - 0.25 * np.exp(-((th[m] - cusp) / 0.9) ** 2), color=PLAST, lw=1.2)
    ax.axvline(thm, color=DIM, lw=0.6, ls=":")
    ax.text(thm + 1, 0.15, r"$\theta_m \approx 15^\circ$", color=SUB, fontsize=8)
    ax.text(0.8, 1.13, "малоугловые\n(стенки дислокаций)", color=SUB, fontsize=7.6, va="bottom")
    ax.text(22, 1.06, "высокоугловые", color=SUB, fontsize=7.6)
    ax.text(41, 0.62, "специальные\n(совпадающие узлы)", color=PLAST, fontsize=7.4)
    ax.set_xlabel(r"угол разориентации $\theta$, \textdegree"); ax.set_ylabel(r"$\gamma/\gamma_m$")
    ax.set_ylim(0, 1.38); ax.set_xlim(0, 60)
    save(fig, "f4_gb")


if __name__ == "__main__":
    f_hcp(); f_interfaces(); f_dislocation(); f_gb()
