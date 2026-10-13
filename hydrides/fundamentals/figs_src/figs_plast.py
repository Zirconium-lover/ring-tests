"""Рисунки главы 7 и введения: диаграмма растяжения, поверхности текучести и путь нагружения трубы, пластическая
зона вокруг несоразмерного включения, текстура трубы и факторы Кернса, схема переориентации гидридов."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Ellipse, FancyArrowPatch, Rectangle

from style import *


def f_ss():
    """σ–ε: упругий участок (наклон E), предел текучести, упрочнение, разгрузка и остаточная пластическая деформация."""
    fig, ax = plt.subplots(figsize=(W2 + 0.5, 2.5))
    E, sy, K, n = 90.0, 0.3, 0.45, 0.35         # ГПа, ГПа, ГПа
    ep = np.linspace(0, 0.05, 300)
    s_pl = sy + K * ep ** n
    eps_el = np.linspace(0, sy / E, 50)
    ax.plot(eps_el * 100, eps_el * E * 1000, color=CIRC)
    ax.plot((s_pl / E + ep) * 100, s_pl * 1000, color=INK)
    k = 180
    e_u = (s_pl[k] / E + ep[k]) * 100
    ax.plot([e_u, ep[k] * 100], [s_pl[k] * 1000, 0], color=PLAST, ls="--", lw=1.1)
    ax.annotate(r"$\varepsilon_p$", xy=(ep[k] * 100, 8), xytext=(ep[k] * 100 - 0.6, 90), color=PLAST,
                arrowprops=dict(arrowstyle="->", color=PLAST, lw=0.8), fontsize=9)
    ax.plot([sy / E * 100], [sy * 1000], "o", color=RAD, ms=4)
    ax.text(sy / E * 100 + 0.15, sy * 1000 - 40, r"$\sigma_y$", color=RAD)
    ax.text(0.05, 120, r"наклон $E$", color=CIRC, fontsize=8.5, rotation=0)
    ax.text(3.0, 520, "упрочнение", color=INK, fontsize=8.5)
    ax.text(e_u - 0.4, 250, "разгрузка\nупругая", color=PLAST, fontsize=7.8, ha="right")
    ax.set_xlabel(r"деформация $\varepsilon$, \%"); ax.set_ylabel(r"напряжение $\sigma$, МПа")
    ax.set_xlim(0, 5.5); ax.set_ylim(0, 650)
    save(fig, "f7_ss")


def f_yield():
    """Плоское напряжённое состояние: эллипс Мизеса и шестиугольник Треска; пути нагружения — одноосный,
    труба с заглушками (σ_z = σ_θ/2), равное двухосное."""
    fig, ax = plt.subplots(figsize=(W2 + 0.4, 3.0))
    t = np.linspace(0, 2 * np.pi, 400)
    # σ1² − σ1σ2 + σ2² = 1: эллипс с полуосями √2 вдоль (1,1) и √(2/3) вдоль (1,−1)
    a_ = np.sqrt(2.0); b_ = np.sqrt(2.0 / 3.0)
    s1 = (a_ * np.cos(t) + b_ * np.sin(t)) / np.sqrt(2); s2 = (a_ * np.cos(t) - b_ * np.sin(t)) / np.sqrt(2)
    ax.plot(s1, s2, color=CIRC, label="Мизес")
    tr = np.array([[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1], [1, 0]])
    ax.plot(tr[:, 0], tr[:, 1], color=RAD, ls="--", lw=1.1, label="Треска")
    ax.axhline(0, color=DIM, lw=0.5); ax.axvline(0, color=DIM, lw=0.5)
    for (d, lab_, c) in (((1, 0), "одноосное", INK), ((1, 0.5), r"труба: $\sigma_z=\sigma_\theta/2$", LOAD), ((1, 1), "двухосное", PLAST)):
        d = np.array(d, float)
        k = 1 / np.sqrt(d[0] ** 2 - d[0] * d[1] + d[1] ** 2)
        ax.annotate("", xy=d * k, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        ax.plot(*(d * k), "o", color=c, ms=3.5)
    ax.text(1.17, 0.66, r"$\sigma_\theta = 1.155\,\sigma_y$", color=LOAD, fontsize=7.8)
    ax.text(1.05, -0.12, "одноосное", color=INK, fontsize=7.6)
    ax.text(0.35, 1.2, "равное двухосное", color=PLAST, fontsize=7.6, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.75); ax.set_ylim(-1.4, 1.45)
    ax.set_xlabel(r"$\sigma_\theta/\sigma_y$"); ax.set_ylabel(r"$\sigma_z/\sigma_y$"); ax.legend(loc="lower right", fontsize=7.8)
    save(fig, "f7_yield")


def f_zone():
    """а) радиус пластической зоны вокруг несоразмерного сферического включения: (c/a)³ = 6μδ/σ_y
    (жёсткое включение, несжимаемая пластическая зона); б) схема: гидрид, ореол пластической деформации, петли."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45), gridspec_kw=dict(width_ratios=[1.1, 1]))
    mu = 33e3                      # МПа
    sy = np.linspace(100, 700, 300)
    for d, c, lab_ in ((0.02, HYDR, r"$\delta = 0.02$"), (0.04, CIRC, r"$\delta = 0.04$"), (0.057, RAD, r"$\delta = 0.057$ (гидрид, среднее)")):
        a.plot(sy, (6 * mu * d / sy) ** (1 / 3), color=c, label=lab_)
    a.set_xlabel(r"предел текучести матрицы $\sigma_y$, МПа"); a.set_ylabel(r"$c/a$"); a.legend(fontsize=7.6); label(a, "а")
    a.set_ylim(1, 4.8)
    b.add_patch(Rectangle((-3, -1.6), 6, 3.2, facecolor=METAL, edgecolor=METAL_E))
    for rr, al in ((2.6, 0.06), (2.0, 0.1), (1.5, 0.16)):
        b.add_patch(Ellipse((0, 0), 2 * rr, rr * 1.05, facecolor=PLAST, alpha=al, edgecolor="none"))
    b.add_patch(Ellipse((0, 0), 2.6, 0.22, facecolor=HYD_F, edgecolor=CIRC, lw=1.0))
    for x, y in ((-1.9, 0.55), (-1.5, -0.65), (1.7, 0.6), (2.05, -0.55), (0.3, 0.8), (-0.4, -0.85)):
        b.add_patch(Circle((x, y), 0.12, fill=False, ec=PLAST, lw=0.9))
    b.text(0, 0.0, "гидрид", color=CIRC, fontsize=7.6, ha="center", va="center")
    b.text(0, 1.25, "пластический ореол", color=PLAST, fontsize=8, ha="center")
    b.text(2.05, 1.0, "петли", color=PLAST, fontsize=7.6, ha="center")
    b.set_xlim(-3.05, 3.05); b.set_ylim(-1.65, 1.65); b.set_aspect("equal"); b.axis("off"); label(b, "б", 0.04, 0.98)
    fig.tight_layout(w_pad=1.5)
    save(fig, "f7_zone")
    return (6 * mu * 0.057 / 300) ** (1 / 3)


def f_texture():
    """Текстура трубы (схема): оси c наклонены от радиального направления к окружному на ±χ0 с разбросом;
    а) полюсная фигура (0002) в равноплощадной проекции, центр — радиальное направление; б) факторы Кернса."""
    rng = np.random.default_rng(5)
    n = 4000
    chi = rng.choice([-1, 1], n) * rng.normal(35, 12, n)          # наклон от r к θ, °
    eta = rng.normal(0, 10, n)                                    # отклонение к оси трубы, °
    c = np.radians(chi); e = np.radians(eta)
    v = np.stack([np.cos(c) * np.cos(e), np.sin(c) * np.cos(e), np.sin(e)], 1)    # (r, θ, z)
    v[v[:, 0] < 0] *= -1
    fr, ft, fa = (v ** 2).mean(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.6), gridspec_kw=dict(width_ratios=[1, 1.05]))
    # равноплощадная проекция (Ламберт) с центром в r: радиус ∝ √2·sin(угол/2)
    ang = np.arccos(np.clip(v[:, 0], -1, 1))
    rho = np.sqrt(2) * np.sin(ang / 2)
    az = np.arctan2(v[:, 2], v[:, 1])
    X, Y = rho * np.cos(az), rho * np.sin(az)
    a.hexbin(X, Y, gridsize=28, extent=(-1.42, 1.42, -1.42, 1.42), cmap="Blues", mincnt=1, linewidths=0)
    a.add_patch(Circle((0, 0), np.sqrt(2), fill=False, ec=SUB, lw=0.8))
    a.annotate("", xy=(1.62, 0), xytext=(1.42, 0), arrowprops=dict(arrowstyle="->", color=INK))
    a.text(1.66, 0.08, r"$\theta$", color=INK); a.text(0.06, 1.5, r"$z$ (ось трубы)", color=INK, fontsize=8)
    a.text(-0.08, -0.12, r"$r$", color=INK, ha="right")
    a.plot(0, 0, "+", color=INK, ms=6)
    a.set_aspect("equal"); a.set_xlim(-1.75, 1.8); a.set_ylim(-1.6, 1.65); a.axis("off"); label(a, "а", 0.08, 0.97)
    names = [r"$f_r$ (радиальное)", r"$f_\theta$ (окружное)", r"$f_z$ (осевое)"]
    vals = [fr, ft, fa]
    b.bar(range(3), vals, color=[RAD, CIRC, DIM], width=0.6)
    for i, val in enumerate(vals):
        b.text(i, val + 0.015, f"{val:.2f}", ha="center", fontsize=8.5)
    b.set_xticks(range(3)); b.set_xticklabels(names, fontsize=8); b.set_ylabel(r"$f = \langle\cos^2\rangle$"); b.set_ylim(0, 0.8)
    label(b, "б")
    fig.tight_layout(w_pad=1.0)
    save(fig, "f7_texture")
    return fr, ft, fa


def f_tube():
    """Схема: сечение трубы до опыта — окружные гидриды; после охлаждения под окружным напряжением — радиальные."""
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.45))
    rng = np.random.default_rng(2)
    for k, ax in enumerate(axs):
        Ro, Ri = 1.0, 0.72
        ax.add_patch(Wedge((0, 0), Ro, 30, 150, width=Ro - Ri, facecolor=METAL, edgecolor=METAL_E))
        for _ in range(60):
            th = np.radians(rng.uniform(38, 142)); r = rng.uniform(Ri + 0.03, Ro - 0.03)
            L = rng.uniform(0.03, 0.07)
            radial = (k == 1) and rng.random() < 0.55
            ang = th if radial else th + np.pi / 2
            x, y = r * np.cos(th), r * np.sin(th)
            dx, dy = L * np.cos(ang), L * np.sin(ang)
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=RAD if radial else CIRC, lw=1.3, solid_capstyle="round")
        if k == 1:
            for th in np.radians([55, 90, 125]):
                x, y = 1.09 * np.cos(th), 1.09 * np.sin(th)
                tx, ty = -np.sin(th), np.cos(th)
                ax.annotate("", xy=(x + 0.13 * tx, y + 0.13 * ty), xytext=(x - 0.13 * tx, y - 0.13 * ty),
                            arrowprops=dict(arrowstyle="<->", color=LOAD, lw=1.2))
            ax.text(0, 1.22, r"окружное напряжение $\sigma_\theta$", color=LOAD, ha="center", fontsize=8.5)
        ax.text(0, 0.5, "до опыта: окружные" if k == 0 else "охлаждение под нагрузкой:\nчасть гидридов радиальные",
                ha="center", va="top", fontsize=8.3, color=INK)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(0.22, 1.32); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(w_pad=0.5)
    save(fig, "f0_tube")


if __name__ == "__main__":
    f_ss(); f_yield(); print("c/a:", round(f_zone(), 2)); print("Кернс:", [round(v, 3) for v in f_texture()]); f_tube()
