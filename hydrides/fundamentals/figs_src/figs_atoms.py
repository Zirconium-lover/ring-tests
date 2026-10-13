"""Рисунки главы «Атомы, связи и твёрдое тело»: потенциал и сила взаимодействия, тепловое расширение из
ангармонизма, теплоёмкость по Дебаю, зоны энергий (сильная связь и модель Кронига–Пенни), распределение Ферми."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from style import *

kB_eV = 8.617e-5


def morse(r, D=1.0, a=1.6, r0=1.0):
    return D * ((1 - np.exp(-a * (r - r0))) ** 2 - 1)


def f_potential():
    """а) потенциал пары атомов (Морзе) и сила; б) уровни энергии и середины колебаний: асимметрия ямы сдвигает
    среднее расстояние с ростом энергии — тепловое расширение."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.6))
    r = np.linspace(0.65, 3.0, 500)
    U = morse(r)
    F = -np.gradient(U, r)
    a.plot(r, U, color=INK, label=r"энергия $U(r)$")
    a.plot(r, F / 4, color=RAD, label=r"сила $F = -\mathrm dU/\mathrm dr$ (масштаб)")
    a.axhline(0, color=DIM, lw=0.6); a.axvline(1.0, color=DIM, lw=0.6, ls=":")
    a.annotate("", xy=(1.0, -1.0), xytext=(1.0, 0), arrowprops=dict(arrowstyle="<->", color=LOAD, lw=1.0))
    a.text(1.06, -0.55, r"$E_{\text{св}}$", color=LOAD)
    a.text(1.03, -1.2, r"$r_0$", color=SUB)
    a.text(0.7, 0.72, "отталкивание", color=SUB, fontsize=7.6)
    a.text(2.0, -0.42, "притяжение", color=SUB, fontsize=7.6)
    a.set_ylim(-1.25, 0.9); a.set_xlim(0.65, 3.0)
    a.set_xlabel(r"расстояние $r/r_0$"); a.set_ylabel(r"$U/E_{\text{св}}$"); a.legend(loc="lower right", fontsize=7.4); label(a, "а")
    r = np.linspace(0.7, 2.6, 600)
    U = morse(r)
    b.plot(r, U, color=INK)
    mids = []
    for E in np.linspace(-0.9, -0.2, 6):
        m = U <= E
        rl, rr = r[m][0], r[m][-1]
        b.plot([rl, rr], [E, E], color=CIRC, lw=0.9)
        mids.append(((rl + rr) / 2, E))
    mids = np.array(mids)
    b.plot(mids[:, 0], mids[:, 1], "o-", color=RAD, ms=3.5, lw=1.2, label="середина колебаний")
    b.axvline(1.0, color=DIM, lw=0.6, ls=":")
    b.set_xlabel(r"$r/r_0$"); b.set_ylabel(r"$U/E_{\text{св}}$"); b.set_ylim(-1.1, 0.15); b.legend(loc="upper left", fontsize=7.4)
    b.text(1.75, -0.95, "чем выше энергия\n(температура),\nтем дальше середина", color=RAD, fontsize=7.6)
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "a1_potential")


def debye_c(T, th):
    x = th / T
    I = quad(lambda y: y ** 4 * np.exp(y) / (np.exp(y) - 1) ** 2, 0, x)[0]
    return 3 * (1 / x) ** 3 * I            # в единицах 3R


def f_thermal():
    """а) среднее расстояние в яме Морзе при классическом распределении Больцмана — тепловое расширение;
    б) теплоёмкость по Дебаю (Θ_D ≈ 290 К для Zr) и предел Дюлонга–Пти 3R."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45))
    r = np.linspace(0.6, 4.0, 4000)
    U = morse(r)
    kT = np.linspace(0.005, 0.25, 60)
    mean = [np.sum(r * np.exp(-U / t)) / np.sum(np.exp(-U / t)) for t in kT]
    a.plot(kT, np.array(mean) - 1, color=CIRC, label="яма Морзе (несимметричная)")
    U2 = 1.6 ** 2 * (r - 1) ** 2 - 1
    mean2 = [np.sum(r * np.exp(-U2 / t)) / np.sum(np.exp(-U2 / t)) for t in kT]
    a.plot(kT, np.array(mean2) - 1, color=DIM, ls="--", label="парабола (гармоническая)")
    a.set_xlabel(r"$k_BT/E_{\text{св}}$"); a.set_ylabel(r"$\langle r\rangle/r_0 - 1$"); a.legend(fontsize=7.6, loc="upper left"); label(a, "а")
    T = np.linspace(5, 900, 300)
    th = 290.0
    c = [debye_c(t, th) for t in T]
    b.plot(T, c, color=INK, label=r"Дебай, $\Theta_D = 290$ К")
    b.plot(T, (T / th) ** 3 * 12 * np.pi ** 4 / 15 / 3, color=DIM, ls="--", lw=1.0, label=r"закон $T^3$")
    b.axhline(1, color=RAD, lw=0.8, ls=":"); b.text(700, 1.03, r"$3R$ (Дюлонг--Пти)", color=RAD, fontsize=8)
    b.set_ylim(0, 1.2); b.set_xlabel(r"$T$, К"); b.set_ylabel(r"$C_V/3R$"); b.legend(fontsize=7.6, loc="lower right"); label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "a2_thermal")


def f_bands():
    """а) N атомов сближаются: каждый уровень расщепляется в зону шириной 4t(r), t ∝ e^(−r/r1);
    б) Кронига–Пенни: свободный электрон (парабола) и разрешённые зоны с щелями на k = nπ/a."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.65))
    rr = np.linspace(0.8, 4.0, 300)
    for E0, t0, c in ((-6.0, 1.2, CIRC), (-2.0, 2.5, RAD)):
        t = t0 * np.exp(-(rr - 1.0) / 0.8)
        for kk in np.linspace(0, np.pi, 9):
            a.plot(rr, E0 - 2 * t * np.cos(kk), color=c, lw=0.5, alpha=0.8)
        a.fill_between(rr, E0 - 2 * t, E0 + 2 * t, color=c, alpha=0.12)
    a.axvline(1.6, color=LOAD, lw=0.8, ls="--"); a.text(1.65, 2.6, "в кристалле", color=LOAD, fontsize=8)
    a.text(3.2, -5.6, "уровни\nатома", color=SUB, fontsize=7.8)
    a.set_xlabel("расстояние между атомами (усл. ед.)"); a.set_ylabel("энергия электрона (усл. ед.)"); label(a, "а")
    a.set_ylim(-11, 4)
    # Кронига–Пенни: cos(ka) = cos(qa) + P sin(qa)/(qa), q = √E (ед. ħ²/2m = 1, a = 1)
    P = 3.0
    E = np.linspace(0.01, 60, 20000)
    q = np.sqrt(E)
    rhs = np.cos(q) + P * np.sin(q) / q
    ok = np.abs(rhs) <= 1
    k = np.arccos(np.clip(rhs, -1, 1))
    # раскладка по зонам (расширенная схема): номер зоны по числу пройденных щелей
    band = np.cumsum(np.r_[0, np.diff(ok.astype(int)) == 1])
    kk = np.where(band % 2 == 1, k + (band - 1) * np.pi, band * np.pi - k)
    kk = np.where(ok, kk, np.nan)
    b.plot(kk / np.pi, E, color=INK, lw=1.4, label="электрон в решётке")
    kf = np.linspace(0, 2.6, 200)
    b.plot(kf, (kf * np.pi) ** 2, color=DIM, ls="--", lw=1.0, label="свободный электрон")
    for n in (1, 2):
        e1 = (n * np.pi) ** 2
        b.axvline(n, color=DIM, lw=0.5, ls=":")
    gaps = []
    edges = np.where(np.diff(ok.astype(int)) != 0)[0]
    for i in range(0, min(len(edges), 4), 2):
        lo, hi = E[edges[i]], E[edges[i + 1]] if i + 1 < len(edges) else E[-1]
        b.axhspan(lo, hi, color=RAD, alpha=0.08)
    b.text(2.05, 33, "запрещённые\nзоны (щели)", color=RAD, fontsize=7.6)
    b.set_xlim(0, 2.6); b.set_ylim(0, 60)
    b.set_xlabel(r"волновое число $k$, ед. $\pi/a$"); b.set_ylabel("энергия (усл. ед.)"); b.legend(fontsize=7.4, loc="upper left")
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "a3_bands")


def f_fermi():
    """Заполнение зон: металл, полупроводник, диэлектрик; распределение Ферми–Дирака при разных T."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for i, (name, gap, fill) in enumerate((("металл", 0.0, 0.5), ("полупроводник", 1.1, 1.0), ("диэлектрик", 6.0, 1.0))):
        x0 = i * 1.4
        lo1, hi1 = 0.0, 3.0
        lo2 = hi1 + gap; hi2 = lo2 + 3.0
        if name == "металл":
            a.add_patch(plt.Rectangle((x0, 0.0), 1.0, 6.0, fc="white", ec=SUB))
            a.add_patch(plt.Rectangle((x0, 0.0), 1.0, 3.5, fc=CIRC, alpha=0.25, ec="none"))
            a.plot([x0 - 0.05, x0 + 1.05], [3.5, 3.5], color=RAD, lw=1.2); a.text(x0 + 1.07, 3.4, r"$E_F$", color=RAD, fontsize=8)
        else:
            a.add_patch(plt.Rectangle((x0, lo1), 1.0, hi1 - lo1, fc=CIRC, alpha=0.25, ec=CIRC))
            a.add_patch(plt.Rectangle((x0, lo2), 1.0, hi2 - lo2, fc="white", ec=SUB))
            a.annotate("", xy=(x0 + 0.5, lo2), xytext=(x0 + 0.5, hi1), arrowprops=dict(arrowstyle="<->", color=LOAD, lw=0.9))
            a.text(x0 + 0.56, (hi1 + lo2) / 2, r"$E_g$", color=LOAD, fontsize=8, va="center")
        a.text(x0 + 0.5, -0.9, name, ha="center", fontsize=8)
    a.set_xlim(-0.2, 4.2); a.set_ylim(-1.3, 12.5); a.axis("off"); label(a, "а", 0.05, 0.98)
    E = np.linspace(-0.5, 0.5, 400)
    for T, c in ((0, INK), (300, CIRC), (1000, RAD)):
        f = (E < 0).astype(float) if T == 0 else 1 / (1 + np.exp(E / (kB_eV * T)))
        b.plot(E, f, color=c, label=f"{T} К")
    b.set_xlabel(r"$E - E_F$, эВ"); b.set_ylabel("заселённость $f(E)$"); b.legend(fontsize=7.6); label(b, "б")
    fig.tight_layout(w_pad=1.5)
    save(fig, "a4_fermi")


if __name__ == "__main__":
    f_potential(); f_thermal(); f_bands(); f_fermi()
