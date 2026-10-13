"""Рисунки главы «Механические свойства и разрушение»: диаграмма растяжения (условная и истинная), условный предел
текучести σ0.2 и критерий Консидера; поле у вершины трещины и критическое напряжение для трещины длины a;
ползучесть (стадии) и релаксация напряжений по степенному закону."""
import numpy as np
import matplotlib.pyplot as plt

from style import *

E = 95e3                 # МПа
K_H, N_H = 750.0, 0.10   # закон Холломона σ = K ε_p^n (модельный сплав циркония, 20 °C)


def tensile(e_end=0.26):
    """Истинные и условные кривые; после ε_u = n — шейка (условное напряжение падает до разрушения)."""
    sig_u = K_H * N_H ** N_H                               # истинное напряжение при ε_p = n
    sig_t = np.linspace(0, sig_u, 3000)
    eps_t = sig_t / E + (sig_t / K_H) ** (1 / N_H)          # упругая часть + закон Холломона
    s_y = K_H * 0.002 ** N_H
    e_eng = np.exp(eps_t) - 1
    s_eng = sig_t / np.exp(eps_t)
    # шейка: условное напряжение плавно спадает
    eu = e_eng[-1]
    e_n = np.linspace(eu, e_end, 200)
    s_n = s_eng[-1] * (1 - 0.9 * ((e_n - eu) / (e_end - eu)) ** 2.2)
    return eps_t, sig_t, e_eng, s_eng, e_n, s_n, s_y


def f_tensile():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W1, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    eps_t, sig_t, e_eng, s_eng, e_n, s_n, s_y = tensile()
    a.plot(np.r_[e_eng, e_n] * 100, np.r_[s_eng, s_n], color=INK, label="условная")
    a.plot(eps_t * 100, sig_t, color=CIRC, ls="--", label="истинная")
    iu = len(s_eng) - 1
    a.plot(e_eng[iu] * 100, s_eng[iu], "o", color=RAD, ms=4)
    a.text(e_eng[iu] * 100 + 0.6, s_eng[iu] + 25, r"$\sigma_{\text{в}}$", color=RAD, fontsize=9)
    a.text(e_eng[iu] * 100 + 4.5, s_eng[iu] + 15, "начало\nшейки", color=RAD, fontsize=7.4)
    a.plot(e_n[-1] * 100, s_n[-1], "x", color=INK, ms=5); a.text(e_n[-1] * 100 - 0.8, s_n[-1] + 25, "разрыв", fontsize=7.4, ha="right")
    a.set_xlabel(r"деформация, \%"); a.set_ylabel(r"напряжение, МПа"); a.set_ylim(0, 700); a.legend(fontsize=7.4, loc="lower center")
    label(a, "а")
    b.plot(e_eng * 100, s_eng, color=INK)
    x = np.linspace(0.2, 0.8, 10)
    b.plot(x, (x / 100 - 0.002) * E, color=LOAD, lw=0.9, ls="--")
    b.plot(0.2 + s_y / E * 100, s_y, "o", color=RAD, ms=4)
    b.text(0.2 + s_y / E * 100 + 0.05, s_y - 70, r"$\sigma_{0.2} \approx %.0f$ МПа" % s_y, color=RAD, fontsize=8)
    b.text(0.25, 60, "0.2\\,\\%", color=LOAD, fontsize=8)
    b.set_xlim(0, 1.5); b.set_ylim(0, 600); b.set_xlabel(r"деформация, \%"); label(b, "б")
    ep = np.linspace(0.005, 0.25, 200)
    sg = K_H * ep ** N_H
    c.plot(ep * 100, sg, color=CIRC, label=r"$\sigma$")
    c.plot(ep * 100, N_H * K_H * ep ** (N_H - 1), color=PLAST, label=r"$\mathrm d\sigma/\mathrm d\varepsilon$")
    c.axvline(N_H * 100, color=RAD, lw=0.6, ls=":")
    c.text(N_H * 100 + 0.5, 120, r"$\varepsilon_u = n$", color=RAD, fontsize=8.5)
    c.set_ylim(0, 1400); c.set_xlabel(r"истинная деформация, \%"); c.set_ylabel(r"МПа"); c.legend(fontsize=7.6, loc="upper right")
    label(c, "в")
    fig.tight_layout(w_pad=1.0)
    save(fig, "k1_tensile")
    return s_y, s_eng[iu], e_eng[iu]


def f_fracture():
    """а) σ_yy перед вершиной трещины K/√(2πr) и зона пластичности Ирвина; б) критическое напряжение
    σ_c = K_c/(1.12√(πa)) для поверхностной трещины глубины a при разных K_c (оценка)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45))
    K, sy = 20.0, 500.0                                  # МПа·√м, МПа
    r = np.logspace(-6, -2.3, 400)                       # м
    s = K / np.sqrt(2 * np.pi * r)
    rp = (K / sy) ** 2 / (2 * np.pi)
    a.plot(r * 1e6, s, color=INK, label=r"$K/\sqrt{2\pi r}$")
    a.plot(r * 1e6, np.minimum(s, sy), color=RAD, lw=1.2, label=r"срез на $\sigma_y$")
    a.axvline(rp * 1e6, color=PLAST, lw=0.7, ls="--")
    a.text(rp * 1e6 * 1.15, 800, r"$r_p = \dfrac{1}{2\pi}\Bigl(\dfrac{K}{\sigma_y}\Bigr)^2$" + "\n" + f"$\\approx {rp * 1e6:.0f}$ мкм",
           color=PLAST, fontsize=8)
    a.set_xscale("log"); a.set_ylim(0, 2500); a.set_xlim(1, 5000)
    a.set_xlabel(r"расстояние от вершины $r$, мкм"); a.set_ylabel(r"$\sigma_{yy}$, МПа"); a.legend(fontsize=7.4, loc="upper right")
    label(a, "а")
    aa = np.logspace(-5.3, -2.5, 300)
    for Kc, c in ((5, RAD), (15, LOAD), (40, CIRC)):
        b.plot(aa * 1e6, Kc / (1.12 * np.sqrt(np.pi * aa)), color=c, label=f"$K_c = {Kc}$ МПа" + r"$\sqrt{\text{м}}$")
    b.axhline(sy, color=DIM, lw=0.6, ls=":"); b.text(6, sy + 40, r"$\sigma_y$", color=SUB, fontsize=8)
    b.axvline(600, color=INK, lw=0.6, ls="--"); b.text(660, 1300, "толщина\nстенки", fontsize=7.4, ha="left")
    b.set_xscale("log"); b.set_ylim(0, 3000); b.set_xlim(5, 3000)
    b.set_xlabel(r"глубина трещины $a$, мкм"); b.set_ylabel(r"$\sigma_c$, МПа"); b.legend(fontsize=7.2, loc="upper right")
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "k2_fracture")
    return rp


def f_creep():
    """а) кривая ползучести при постоянной нагрузке: неустановившаяся, установившаяся, ускоренная стадии;
    б) релаксация напряжения при постоянной полной деформации для степенного закона ε̇ = A σ^n
    (n = 5) при трёх значениях A (условно — три температуры)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.4))
    t = np.linspace(0, 1000, 1000)
    e = 0.3 + 0.6 * (1 - np.exp(-t / 60)) + 0.0012 * t + 0.08 * (np.exp(t / 140) - 1) / np.exp(1000 / 140) * 25
    stop = np.argmax(e > 3.2) if np.any(e > 3.2) else len(t)
    a.plot(t[:stop], e[:stop], color=INK)
    a.plot(t[stop - 1], e[stop - 1], "x", color=INK, ms=6)
    for x0, x1, nm in ((0, 150, "I"), (150, 720, "II"), (720, t[stop - 1], "III")):
        a.axvspan(x0, x1, color=[CIRC, LOAD, RAD]["I II III".split().index(nm)], alpha=0.07)
        a.text((x0 + x1) / 2, 2.95, nm, ha="center", fontsize=9)
    a.text(15, 0.12, r"$\varepsilon_0$ (мгновенная)", fontsize=7.4, color=SUB)
    a.set_xlabel("время (усл.)"); a.set_ylabel(r"деформация, \%"); a.set_ylim(0, 3.3); label(a, "а")
    n = 5.0
    s0 = 300.0
    tt = np.logspace(-1, 4, 300)                                    # ч
    for A, c, nm in ((1e-17, CIRC, "холоднее"), (1e-16, LOAD, ""), (1e-15, RAD, "горячее")):
        st = s0 * (1 + (n - 1) * E * A * s0 ** (n - 1) * tt) ** (-1 / (n - 1))
        b.plot(tt, st, color=c, label=nm if nm else None)
    b.set_xscale("log"); b.set_ylim(0, 320)
    b.set_xlabel("время, ч"); b.set_ylabel(r"$\sigma$, МПа"); b.legend(fontsize=7.6, loc="lower left")
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "k3_creep")


if __name__ == "__main__":
    s_y, s_u, e_u = f_tensile(); print(f"σ0.2 = {s_y:.0f} МПа, σв = {s_u:.0f} МПа при e = {e_u * 100:.1f} %")
    rp = f_fracture(); print(f"r_p = {rp * 1e6:.0f} мкм")
    f_creep()
