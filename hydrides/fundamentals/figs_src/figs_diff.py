"""Рисунки главы 3: случайное блуждание, решения уравнения Фика, длина диффузии водорода в цирконии,
перераспределение водорода в поле напряжений, устойчивость явной схемы."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

from style import *

R = 8.314
D0, QD = 7.9e-7, 44.4e3            # водород в α-Zr (Kearns 1972; как в модели), м²/с, Дж/моль
V_H = 1.7e-6                        # парциальный мольный объём водорода в Zr, м³/моль


def f_walk():
    """Случайное блуждание: траектории частиц и распределение смещений — гауссиана шириной √(2Dt)."""
    rng = np.random.default_rng(3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for k, c in enumerate((CIRC, RAD, HYDR, LOAD, PLAST)):
        steps = rng.choice([[1, 0], [-1, 0], [0, 1], [0, -1]], size=400)
        p = np.vstack([[0, 0], np.cumsum(steps, 0)])
        a.plot(p[:, 0], p[:, 1], color=c, lw=0.7, alpha=0.9)
        a.plot(p[-1, 0], p[-1, 1], "o", color=c, ms=3.5)
    a.plot(0, 0, "o", color=INK, ms=4)
    th = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.sqrt(400) * np.cos(th), np.sqrt(400) * np.sin(th), color=DIM, lw=0.8, ls="--")
    a.text(14.5, 14.5, r"$\sqrt{n}\,a$", color=SUB, fontsize=8.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.spines["left"].set_visible(False); a.spines["bottom"].set_visible(False)
    a.set_xlabel(r"5 атомов, по $n = 400$ прыжков длиной $a$"); label(a, "а")
    N = 20000
    for n, c in ((25, CIRC), (100, PLAST), (400, RAD)):
        x = rng.choice([-1, 1], size=(N, n)).sum(1)
        b.hist(x, bins=np.arange(-70, 71, 2) - 1, density=True, color=c, alpha=0.25)
        xx = np.linspace(-70, 70, 400)
        b.plot(xx, 2 * np.exp(-xx ** 2 / (2 * n)) / np.sqrt(2 * np.pi * n) / 2, color=c, label=fr"$n={n}$")
    b.set_xlabel(r"смещение $x/a$"); b.set_ylabel("доля частиц"); b.legend(); label(b, "б")
    b.text(0.98, 0.55, r"$\langle x^2\rangle = n a^2 = 2Dt$", transform=b.transAxes, ha="right", color=SUB, fontsize=8.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "f3_walk")


def f_profiles():
    """Тонкий слой расплывается гауссианой; постоянная концентрация на поверхности — erfc. Масштаб — √(Dt)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45))
    x = np.linspace(-6, 6, 600)
    for Dt, c in ((0.25, RAD), (1.0, PLAST), (4.0, CIRC)):
        a.plot(x, np.exp(-x ** 2 / (4 * Dt)) / np.sqrt(4 * np.pi * Dt), color=c, label=fr"$Dt = {Dt:g}$")
    a.set_xlabel(r"$x$ (усл. ед.)"); a.set_ylabel(r"$c(x,t)$"); a.legend(); label(a, "а")
    x = np.linspace(0, 6, 400)
    for Dt, c in ((0.25, RAD), (1.0, PLAST), (4.0, CIRC)):
        b.plot(x, erfc(x / (2 * np.sqrt(Dt))), color=c, label=fr"$Dt = {Dt:g}$")
        b.plot([2 * np.sqrt(Dt)], [erfc(1.0)], "o", color=c, ms=3.5)
    b.axhline(erfc(1.0), color=DIM, lw=0.6, ls=":")
    b.text(5.9, erfc(1.0) + 0.03, r"$x = 2\sqrt{Dt}$: $c = 0.16\,c_s$", ha="right", fontsize=8, color=SUB)
    b.set_xlabel(r"$x$"); b.set_ylabel(r"$c/c_s$"); b.legend(); label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "f3_profiles")


def f_dh():
    """Длина диффузии водорода в α-Zr √(Dt) от температуры для разных времён — против масштабов структуры."""
    fig, ax = plt.subplots(figsize=(W1 * 0.62, 2.7))
    T = np.linspace(20, 450, 300)
    D = D0 * np.exp(-QD / (R * (T + 273.15)))
    for t, lab_, c in ((1, "1 с", DIM), (60, "1 мин", HYDR), (3600, "1 ч", CIRC), (86400, "1 сут", PLAST)):
        L = np.sqrt(D * t) * 1e6
        ax.plot(T, L, color=c)
        ax.text(T[-1] + 5, L[-1], lab_, color=c, fontsize=8, va="center")
    for v, lab_ in ((0.3, "толщина пластинки"), (5, "зерно"), (60, "шаг между гидридами"), (600, "стенка трубы")):
        ax.axhline(v, color=LOAD, lw=0.6, ls=":")
        ax.text(22, v * 1.15, lab_, fontsize=7.5, color=LOAD, bbox=dict(fc="white", ec="none", pad=0.4))
    ax.set_yscale("log"); ax.set_ylim(1e-3, 3e4); ax.set_xlim(20, 500)
    ax.set_xlabel(r"$T$, \textdegree C"); ax.set_ylabel(r"$\sqrt{Dt}$, мкм")
    save(fig, "f3_dh")


def f_stress():
    """Равновесное распределение водорода в поле гидростатического напряжения: c/c0 = exp(V_H σ_h / RT).
    а) у вершины трещины (плоская деформация, K = 10 МПа·√м); б) обогащение от σ_h при разных T."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45))
    nu = 0.34
    r = np.logspace(-0.5, 2.5, 300) * 1e-6
    K = 10e6
    sh = 2 * (1 + nu) / 3 * K / np.sqrt(2 * np.pi * r)          # перед трещиной, θ = 0
    for Tc, c in ((150, RAD), (250, PLAST), (350, CIRC)):
        Tk = Tc + 273.15
        a.plot(r * 1e6, np.exp(V_H * np.minimum(sh, 0.9e9) / (R * Tk)), color=c, label=fr"{Tc}\,\textdegree C")
    a.set_xscale("log"); a.set_xlabel(r"расстояние от вершины, мкм"); a.set_ylabel(r"$c/c_0$")
    a.legend(loc="upper right"); label(a, "а")
    a.text(0.97, 0.5, r"$K = 10$ МПа$\sqrt{\text{м}}$", transform=a.transAxes, ha="right", fontsize=8, color=SUB)
    a.axvspan(r[0] * 1e6, 15.7, color=PLAST, alpha=0.07)
    a.text(1.0, 1.08, "пластическая\nзона:\n" + r"$\sigma_h \approx 3\sigma_y$", color=PLAST, fontsize=7.6, ha="center")
    s = np.linspace(-300, 600, 300) * 1e6
    for Tc, c in ((150, RAD), (250, PLAST), (350, CIRC)):
        b.plot(s / 1e6, np.exp(V_H * s / (R * (Tc + 273.15))), color=c)
    b.axhline(1, color=DIM, lw=0.6); b.axvline(0, color=DIM, lw=0.6)
    b.set_xlabel(r"$\sigma_h$, МПа"); b.set_ylabel(r"$c/c_0$"); label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "f3_stress")


def f_numerics():
    """Явная схема для уравнения диффузии: при r = DΔt/Δx² ≤ 1/2 решение гладкое, выше — разнос пилой."""
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.3), sharey=True)
    n = 101; x = np.linspace(-1, 1, n); dx = x[1] - x[0]
    c0 = (np.abs(x) < 0.1).astype(float)
    for ax, rr, c in ((axs[0], 0.45, CIRC), (axs[1], 0.51, RAD)):
        u = c0.copy(); steps = 80
        for _ in range(steps):
            u[1:-1] = u[1:-1] + rr * (u[2:] - 2 * u[1:-1] + u[:-2])
        Dt = rr * dx * dx * steps
        exact = 0.5 * (erfc((x - 0.1) / (2 * np.sqrt(Dt))) - erfc((x + 0.1) / (2 * np.sqrt(Dt))))
        ax.plot(x, exact, color=DIM, lw=2.5, label="точное")
        ax.plot(x, u, color=c, lw=1.1, label=fr"схема, $r = {rr}$")
        ax.set_ylim(-0.65, 0.75); ax.set_xlabel(r"$x$"); ax.legend(loc="upper left", fontsize=7.8)
    axs[0].set_ylabel(r"$c$"); label(axs[0], "а"); label(axs[1], "б")
    fig.tight_layout(w_pad=1.5)
    save(fig, "f3_numerics")


if __name__ == "__main__":
    f_walk(); f_profiles(); f_dh(); f_stress(); f_numerics()
