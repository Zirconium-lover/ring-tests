"""Рисунки глав 1–2: энергия Гиббса, растворы, общая касательная, растворимость водорода в цирконии."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from style import *

R = 8.314


def f_gibbs_T():
    """G(T) двух фаз: пересечение при T_e, движущая сила ΔG ≈ ΔH·ΔT/T_e; рядом — устойчивое, метастабильное, неустойчивое."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    T = np.linspace(300, 1300, 300)
    Te = 900.0
    Ha, Sa = 0.0, 30.0                      # «низкотемпературная» фаза α
    Hb = 9000.0; Sb = Sa + Hb / Te          # «высокотемпературная» фаза β: ΔS = ΔH / T_e
    curv = 0.004
    Ga = Ha - T * Sa - curv * T ** 2 / 2
    Gb = Hb - T * Sb - curv * T ** 2 / 2
    a.plot(T, Ga / 1000, color=CIRC, label=r"$G_\alpha$")
    a.plot(T, Gb / 1000, color=RAD, label=r"$G_\beta$")
    a.axvline(Te, color=DIM, lw=0.8, ls=":")
    a.text(Te + 10, a.get_ylim()[0] if False else (Ga.min() / 1000), "", color=SUB)
    T1 = 640.0
    g1a = Ha - T1 * Sa - curv * T1 ** 2 / 2; g1b = Hb - T1 * Sb - curv * T1 ** 2 / 2
    a.annotate("", xy=(T1, g1a / 1000), xytext=(T1, g1b / 1000), arrowprops=dict(arrowstyle="<->", color=LOAD, lw=1.2))
    a.text(T1 - 30, (g1a + g1b) / 2000, r"$\Delta G$", color=LOAD, ha="right", va="center")
    a.annotate("", xy=(T1, Ga.max() / 1000 + 6), xytext=(Te, Ga.max() / 1000 + 6),
               arrowprops=dict(arrowstyle="<->", color=SUB, lw=0.9))
    a.text((T1 + Te) / 2, Ga.max() / 1000 + 7.2, r"$\Delta T$", color=SUB, ha="center")
    a.text(Te + 15, Ga.min() / 1000 + 1, r"$T_e$", color=SUB)
    a.set_xlabel(r"температура $T$, К"); a.set_ylabel(r"$G$, кДж/моль")
    a.legend(loc="lower left"); a.set_yticks([]); label(a, "а")
    # ландшафт состояний
    x = np.linspace(0, 10, 600)
    y = 1.2 * np.exp(-((x - 2.0) / 0.9) ** 2) * 0 + 0.35 * (x - 1.6) ** 2 * np.exp(-(x - 1.6) ** 2 / 3)
    y = (0.9 - 0.75 * np.exp(-((x - 1.8) / 0.8) ** 2) + 0.35 * np.exp(-((x - 4.2) / 0.75) ** 2)
         - 1.35 * np.exp(-((x - 7.3) / 1.2) ** 2) + 0.4 * np.exp(-((x - 9.6) / 0.5) ** 2))
    b.plot(x, y, color=INK, lw=1.4)
    for xs, c, t in ((1.8, CIRC, "метастабильное"), (4.2, RAD, "неустойчивое"), (7.3, HYDR, "устойчивое")):
        ys = np.interp(xs, x, y)
        b.plot([xs], [ys + 0.09], "o", ms=8, color=c, mec="white", mew=1.0, zorder=5)
        b.text(xs + (0.0 if c != RAD else 0.9), ys - 0.12 if c != RAD else ys + 0.12, t, ha="center" if c != RAD else "left",
               va="top" if c != RAD else "bottom", color=c, fontsize=8.5)
    ym = np.interp(1.8, x, y); yb = np.interp(4.2, x, y)
    b.annotate("", xy=(2.85, yb), xytext=(2.85, ym), arrowprops=dict(arrowstyle="<->", color=LOAD, lw=1.1))
    b.text(2.75, (ym + yb) / 2, r"$\Delta G_a$", color=LOAD, ha="right", va="center")
    b.set_xticks([]); b.set_yticks([]); b.set_xlabel("расположение атомов"); b.set_ylabel(r"$G$")
    b.set_ylim(y.min() - 0.35, y.max() + 0.35)
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "f1_gibbs")


def g_reg(x, W, T):
    """Энергия смешения регулярного раствора на моль, Дж (W = Ω)."""
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return W * x * (1 - x) + R * T * (x * np.log(x) + (1 - x) * np.log(1 - x))


def f_mixing():
    """ΔG_mix регулярного раствора: при Ω/RT > 2 появляется двойная яма; разложение на ΔH и −TΔS."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45))
    x = np.linspace(0.001, 0.999, 500)
    T = 1000.0
    cols = [CIRC, "#5a8fe0", PLAST, "#c45ea0", RAD]
    for w, c in zip((0.0, 1.5, 2.0, 2.5, 3.0), cols):
        a.plot(x, g_reg(x, w * R * T, T) / (R * T), color=c, label=fr"$\Omega/RT={w:g}$")
    a.axhline(0, color=DIM, lw=0.6)
    a.set_xlabel(r"доля $B$, $X$"); a.set_ylabel(r"$\Delta G_{\mathrm{mix}}/RT$")
    a.legend(loc="lower center", ncol=2, fontsize=7.8); label(a, "а")
    a.set_ylim(-0.75, 0.15)
    W = 2.5 * R * T
    b.plot(x, W * x * (1 - x) / (R * T), color=RAD, label=r"$\Delta H_{\mathrm{mix}}=\Omega X(1-X)$")
    b.plot(x, (x * np.log(x) + (1 - x) * np.log(1 - x)), color=CIRC, label=r"$-T\Delta S_{\mathrm{mix}}$")
    b.plot(x, g_reg(x, W, T) / (R * T), color=INK, label=r"$\Delta G_{\mathrm{mix}}$")
    b.axhline(0, color=DIM, lw=0.6)
    b.set_xlabel(r"доля $B$, $X$"); b.set_ylabel(r"в единицах $RT$"); b.legend(loc="lower center", fontsize=7.8)
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "f2_mixing")


def binodal_spinodal(W, Ts):
    xb, xs = [], []
    for T in Ts:
        # бинодаль симметричного раствора: dG/dx = 0 вне центра → ln(x/(1−x)) = W(2x−1)/RT
        f = lambda x: np.log(x / (1 - x)) - W * (2 * x - 1) / (R * T)
        lo, hi = 1e-12, 0.5 - 1e-9
        if W / (R * T) <= 2:
            xb.append(np.nan); xs.append(np.nan); continue
        for _ in range(200):
            m = 0.5 * (lo + hi)
            if f(lo) * f(m) <= 0:
                hi = m
            else:
                lo = m
        xb.append(0.5 * (lo + hi))
        # спинодаль: d²G/dx² = 0 → x(1−x) = RT/(2W)
        xs.append(0.5 - 0.5 * np.sqrt(1 - 2 * R * T / W))
    return np.array(xb), np.array(xs)


def f_gap():
    """Купол расслоения регулярного раствора: бинодаль, спинодаль; G(X) при T ниже критической и общая касательная."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.55))
    W = 20000.0; Tc = W / (2 * R)
    Ts = np.linspace(0.45 * Tc, 0.9999 * Tc, 300)
    xb, xs = binodal_spinodal(W, Ts)
    a.plot(xb, Ts, color=INK); a.plot(1 - xb, Ts, color=INK, label="бинодаль")
    a.plot(xs, Ts, color=RAD, ls="--"); a.plot(1 - xs, Ts, color=RAD, ls="--", label="спинодаль")
    a.fill_betweenx(Ts, xs, 1 - xs, color=RAD, alpha=0.06)
    a.fill_betweenx(Ts, xb, xs, color=LOAD, alpha=0.08); a.fill_betweenx(Ts, 1 - xs, 1 - xb, color=LOAD, alpha=0.08)
    a.text(0.5, 0.62 * Tc, "распад без\nбарьера", ha="center", color=RAD, fontsize=8)
    a.annotate("нужно\nзарождение", xy=(0.17, 0.86 * Tc), xytext=(0.03, 1.02 * Tc), ha="left", color=LOAD, fontsize=8,
               arrowprops=dict(arrowstyle="->", color=LOAD, lw=0.7))
    a.text(0.62, 1.03 * Tc, r"$T_c = \Omega/2R$", ha="left", color=SUB, fontsize=8.5)
    T1 = 0.7 * Tc
    a.axhline(T1, color=CIRC, lw=0.8, ls=":")
    a.set_xlabel(r"$X$"); a.set_ylabel(r"$T$, К"); a.set_xlim(0, 1); a.set_ylim(0.45 * Tc, 1.1 * Tc)
    a.legend(loc="lower center", fontsize=7.8); label(a, "а")
    x = np.linspace(0.001, 0.999, 600)
    g = g_reg(x, W, T1)
    b.plot(x, g / 1000, color=CIRC)
    x1 = binodal_spinodal(W, [T1])[0][0]
    g1 = g_reg(np.array([x1]), W, T1)[0]
    b.plot([x1, 1 - x1], [g1 / 1000, g1 / 1000], color=LOAD, lw=1.2, label="общая касательная")
    b.plot([x1, 1 - x1], [g1 / 1000] * 2, "o", color=LOAD, ms=4)
    xs1 = 0.5 - 0.5 * np.sqrt(1 - 2 * R * T1 / W)
    b.plot([xs1, 1 - xs1], g_reg(np.array([xs1, 1 - xs1]), W, T1) / 1000, "s", color=RAD, ms=3.5, label="точки перегиба")
    b.set_xlabel(r"$X$"); b.set_ylabel(r"$\Delta G_{\mathrm{mix}}$, кДж/моль"); b.legend(loc="upper center", fontsize=7.8)
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "f2_gap")


def f_tangent():
    """Твёрдый раствор и соединение (гидрид): общая касательная задаёт растворимость; если энергия гидрида выше
    (упругая энергия, малый размер), касательная круче и растворимость больше — то, что делает гистерезис."""
    fig, ax = plt.subplots(figsize=(W2 + 0.6, 2.6))
    x = np.linspace(0.0005, 0.75, 800)
    T = 600.0
    Ga = 2000 * x + R * T * (x * np.log(x) + (1 - x) * np.log(1 - x)) + 12000 * x ** 2   # раствор H в α (схема)
    xd = 0.6
    Gd0 = -5000.0
    Gd = Gd0 + 4.0e5 * (x - xd) ** 2
    ax.plot(x, Ga / 1000, color=CIRC, label=r"$\alpha$ (твёрдый раствор)")
    ax.plot(x, Gd / 1000, color=HYDR, label=r"$\delta$ (гидрид)")
    ax.plot(x, (Gd + 1200) / 1000, color=HYDR, ls="--", lw=1.2, label=r"$\delta$ с упругой энергией")

    def tangent(Gd_shift):
        best = None
        for xa in np.linspace(0.001, 0.3, 3000):
            ga = np.interp(xa, x, Ga); dga = np.interp(xa, x, np.gradient(Ga, x))
            line = ga + dga * (x - xa)
            gap = (Gd + Gd_shift) - line
            m = gap.min()
            if best is None or abs(m) < abs(best[0]):
                best = (m, xa, ga, dga, x[np.argmin(gap)])
        return best
    for sh, c, ls in ((0.0, LOAD, "-"), (1200.0, RAD, "--")):
        m, xa, ga, dga, xt = tangent(sh)
        xx = np.array([0, 0.72])
        ax.plot(xx, (ga + dga * (xx - xa)) / 1000, color=c, lw=1.0, ls=ls)
        ax.plot([xa], [ga / 1000], "o", color=c, ms=4)
        ax.plot([xa, xa], [-7.0, ga / 1000], color=c, lw=0.7, ls=":")
        ax.text(xa + (-0.006 if sh == 0 else 0.006), -6.95, r"$X_e$" if sh == 0 else r"$X_e'$", color=c, fontsize=8.5,
                ha="right" if sh == 0 else "left", va="bottom")
    ax.set_ylim(-7, 3); ax.set_xlim(0, 0.75)
    ax.set_xlabel(r"доля водорода $X_{\mathrm H}$"); ax.set_ylabel(r"$G$, кДж/моль")
    ax.legend(loc="upper left", bbox_to_anchor=(0.16, 1.0), fontsize=7.6)
    save(fig, "f2_tangent")


TSS = dict(TSSD=(5.56, 2.24), TSSP=(5.08, 1.68))      # Э635, lg C[ppm] = b − k·1000/T (Плясов и др. 2023, ДСК)


def cl(T_C, bk):
    b, k = bk
    return 10 ** (b - k * 1000 / (np.asarray(T_C, float) + 273.15))


def tl(c, bk):
    b, k = bk
    return 1000 * k / (b - np.log10(c)) - 273.15


def f_tss():
    """Пределы растворимости водорода в Э635: TSSD и TSSP, гистерезис, путь нагрев–охлаждение; аррениусовы координаты."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    T = np.linspace(100, 480, 300)
    a.plot(T, cl(T, TSS["TSSD"]), color=RAD, label="TSSD (растворение)")
    a.plot(T, cl(T, TSS["TSSP"]), color=CIRC, label="TSSP (выпадение)")
    a.fill_between(T, cl(T, TSS["TSSD"]), cl(T, TSS["TSSP"]), color=LOAD, alpha=0.08)
    H = 160.0
    Td, Tp = tl(H, TSS["TSSD"]), tl(H, TSS["TSSP"])
    # путь: нагрев при постоянном C до TSSD, растворение по TSSD до Td; охлаждение — C постоянно до TSSP, далее по TSSP
    Th = np.linspace(100, Td, 100)
    path_h = np.minimum(cl(Th, TSS["TSSD"]), H)
    Tc_ = np.linspace(Td, 100, 200)
    path_c = np.minimum(cl(Tc_, TSS["TSSP"]), H)
    a.plot(Th, path_h, color=INK, lw=1.0, ls="--")
    a.plot(Tc_, path_c, color=INK, lw=1.0, ls=":")
    a.annotate("нагрев", xy=(330, cl(330, TSS["TSSD"])), xytext=(355, 40), color=SUB, fontsize=8,
               arrowprops=dict(arrowstyle="->", color=SUB, lw=0.7))
    a.annotate("охлаждение", xy=(280, cl(280, TSS["TSSP"])), xytext=(160, 150), color=SUB, fontsize=8,
               arrowprops=dict(arrowstyle="->", color=SUB, lw=0.7))
    a.axhline(H, color=DIM, lw=0.6)
    a.plot([Td, Tp], [H, H], "o", color=INK, ms=3.5)
    a.text(Td + 4, H - 14, fr"{Td:.0f}\,\textdegree C", fontsize=7.8, color=RAD)
    a.text(Tp - 4, H + 6, fr"{Tp:.0f}\,\textdegree C", fontsize=7.8, color=CIRC, ha="right")
    a.text(105, H + 5, r"$C = 160$ ppm", fontsize=7.8, color=SUB)
    a.set_ylim(0, 400); a.set_xlim(100, 480)
    a.set_xlabel(r"$T$, \textdegree C"); a.set_ylabel(r"растворённый водород, ppm (масс.)")
    a.legend(loc="upper left", fontsize=7.8); label(a, "а")
    inv = 1000 / (T + 273.15)
    b.plot(inv, np.log(cl(T, TSS["TSSD"])), color=RAD)
    b.plot(inv, np.log(cl(T, TSS["TSSP"])), color=CIRC)
    for key, c, y in (("TSSD", RAD, 0.62), ("TSSP", CIRC, 0.25)):
        Q = TSS[key][1] * 1000 * np.log(10) * R / 1000
        b.text(0.38, y, fr"{key}: $Q \approx {Q:.0f}$ кДж/моль", color=c, transform=b.transAxes, fontsize=8)
    b.set_xlabel(r"$1000/T$, К$^{-1}$"); b.set_ylabel(r"$\ln C$"); label(b, "б")
    fig.tight_layout(w_pad=1.8)
    save(fig, "f2_tss")


def ppm_to_at(c):
    n_h = np.asarray(c) / 1.008; n_zr = 1e6 / 91.22
    return 100 * n_h / (n_h + n_zr)


def f_zrh():
    """Угол диаграммы Zr–H со стороны циркония (схема): α, α+δ, α+β, β; эвтектоид 550 °C."""
    fig, ax = plt.subplots(figsize=(W2 + 0.5, 2.7))
    T = np.linspace(150, 550, 200)
    xs = ppm_to_at(cl(T, TSS["TSSD"]))
    xe = float(ppm_to_at(cl(550, TSS["TSSD"])))
    ax.plot(xs, T, color=RAD)                                          # сольвус α/(α+δ)
    ax.plot([0, xe], [863, 550], color=INK)                            # α/(α+β), схематично прямой
    xb = 37.5
    ax.plot([0, 10], [863, 863 - (863 - 550) * 10 / xb], color=INK)    # (α+β)/β, схематично
    ax.plot([xe, 10], [550, 550], color=INK)                           # эвтектоидная горизонталь
    ax.text(0.35, 400, r"$\alpha$", fontsize=11, color=CIRC)
    ax.text(5.0, 330, r"$\alpha+\delta$", fontsize=10, color=HYDR)
    ax.text(6.0, 690, r"$\alpha+\beta$", fontsize=10, color=INK)
    ax.text(7.6, 840, r"$\beta$", fontsize=11, color=RAD)
    ax.text(6.6, 563, r"эвтектоид $\beta \to \alpha+\delta$", fontsize=7.6, color=SUB)
    ax.plot([xe], [550], "o", color=INK, ms=3.5)
    ax.text(xe + 0.15, 520, fr"$\approx {xe:.1f}$ ат.\,\%", fontsize=7.6, color=SUB)
    ax.text(0.5, 893, r"$863$\,\textdegree C: $\alpha\to\beta$ в чистом Zr", fontsize=7.6, color=SUB)
    for c in (100, 300, 600):
        ax.axvline(float(ppm_to_at(c)), color=DIM, lw=0.5, ls=":")
        ax.text(float(ppm_to_at(c)), 160, f"{c} ppm", rotation=90, fontsize=6.8, color=DIM, ha="right", va="bottom")
    ax.set_xlim(0, 10); ax.set_ylim(150, 930)
    ax.set_xlabel(r"водород, ат.\,\%"); ax.set_ylabel(r"$T$, \textdegree C")
    save(fig, "f2_zrh")


if __name__ == "__main__":
    f_gibbs_T(); f_mixing(); f_gap(); f_tangent(); f_tss(); f_zrh()
