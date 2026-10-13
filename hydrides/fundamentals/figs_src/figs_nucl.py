"""Рисунки главы 6: барьер зарождения, скорость зарождения и С-кривые, зарождение на границе зерна, рост,
кинетика Колмогорова–Джонсона–Мела–Аврами, эффект Гиббса–Томсона, энергетический бюджет гидрида."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon

from style import *

R, kB = 8.314, 1.380649e-23


def f_cnt():
    """ΔG(r) = −(4/3)πr³Δg + 4πr²γ: максимум r*, ΔG*; чем больше движущая сила, тем ниже и ближе барьер."""
    fig, ax = plt.subplots(figsize=(W2 + 0.6, 2.5))
    r = np.linspace(0, 3.2, 400)
    for dg, c in ((1.0, CIRC), (1.5, PLAST), (2.0, RAD)):
        G = -(4 / 3) * np.pi * r ** 3 * dg + 4 * np.pi * r ** 2 * 1.0
        ax.plot(r, G, color=c, label=fr"$\Delta g = {dg:g}$")
        rs = 2 / dg; Gs = 16 * np.pi / (3 * dg ** 2)
        ax.plot([rs], [Gs], "o", color=c, ms=4)
        ax.plot([rs, rs], [0, Gs], color=c, lw=0.6, ls=":")
    ax.plot(r, 4 * np.pi * r ** 2, color=DIM, lw=0.9, ls="--")
    ax.text(1.38, 25.5, r"$4\pi r^2\gamma$", color=SUB, fontsize=8, ha="right")
    ax.axhline(0, color=DIM, lw=0.6)
    ax.text(2.05, 17.5, r"$r^*$, $\Delta G^*$", color=INK, fontsize=8.5)
    ax.set_ylim(-25, 30); ax.set_xlim(0, 3.2)
    ax.set_xlabel(r"радиус зародыша $r$ (ед. $\gamma/\Delta g_1$)"); ax.set_ylabel(r"$\Delta G$ (ед. $\gamma^3/\Delta g_1^2$)")
    ax.legend(loc="lower left", fontsize=7.8)
    save(fig, "f6_cnt")


def f_rate():
    """а) скорость зарождения: множитель барьера растёт с переохлаждением, подвижность падает — максимум;
    б) С-кривые начала (1 %) и конца (99 %) превращения из скоростей зарождения и роста (схема Аврами)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.6))
    Te = 700.0
    T = np.linspace(380, 695, 500)
    dT = Te - T
    Bc = 2.2e6                                  # ΔG*/kT = Bc/(ΔT²·T)
    barrier = np.exp(-Bc / (dT ** 2 * T))
    mob = np.exp(-90e3 / (R * T))
    N = barrier * mob
    a.plot(T, barrier / barrier.max(), color=RAD, label=r"$e^{-\Delta G^*/kT}$")
    a.plot(T, mob / mob.max(), color=CIRC, label=r"$e^{-Q/RT}$ (подвижность)")
    a.plot(T, N / N.max(), color=INK, lw=2.0, label=r"скорость $\dot N$")
    a.axvline(Te, color=DIM, lw=0.6, ls=":"); a.text(Te - 4, 0.55, r"$T_e$", color=SUB, ha="right")
    a.set_xlabel(r"$T$, К"); a.set_ylabel("доля от максимума"); a.legend(loc="upper left", fontsize=7.6); label(a, "а")
    Gr = mob * (1 - np.exp(-dT * 15.0 / T))     # скорость роста: подвижность × (1 − e^(−Δg/RT))
    def tf(f):
        return (-np.log(1 - f) * 3 / (np.pi * N * Gr ** 3)) ** 0.25
    for f, c, ls in ((0.01, CIRC, "-"), (0.99, RAD, "-")):
        t = tf(f)
        b.plot(t / tf(0.01).min(), T, color=c, ls=ls, label=fr"{int(f * 100)} \%")
    b.set_xscale("log"); b.set_xlabel(r"время (ед. минимального времени начала)"); b.set_ylabel(r"$T$, К")
    b.axhline(Te, color=DIM, lw=0.6, ls=":")
    b.set_xlim(0.5, 1e4); b.legend(title="превращено", fontsize=7.6, title_fontsize=7.6, loc="lower right"); label(b, "б")
    b.text(1.2, T[np.argmin(tf(0.01))] + 6, "«нос»", color=SUB, fontsize=8)
    fig.tight_layout(w_pad=2.0)
    save(fig, "f6_rate")


def f_het():
    """а) зародыш-линза на границе зерна: равновесие натяжений γ_αα = 2γ_αβ cos θ;
    б) множитель барьера S(θ) = ½(2 + cos θ)(1 − cos θ)² против γ_αα/γ_αβ."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.35), gridspec_kw=dict(width_ratios=[1, 1.15]))
    th = np.radians(55)
    Rr = 1.0
    h = Rr * (1 - np.cos(th)); w = Rr * np.sin(th)
    t = np.linspace(-th, th, 100)
    xu = Rr * np.sin(t); yu = Rr * np.cos(t) - Rr * np.cos(th)
    a.fill(np.r_[xu, xu[::-1]], np.r_[yu, -yu[::-1]], color=HYD_F, ec=CIRC, lw=1.0)
    a.plot([-1.8, -w], [0, 0], color=INK, lw=1.4); a.plot([w, 1.8], [0, 0], color=INK, lw=1.4)
    a.text(-1.8, 0.1, r"граница зерна" "\n" r"$\gamma_{\alpha\alpha}$", color=INK, fontsize=8, va="bottom")
    a.text(0, 0.0, r"$\beta$", color=CIRC, fontsize=11, ha="center", va="center")
    a.annotate("", xy=(w + 0.6 * np.cos(th), 0.6 * np.sin(th)), xytext=(w, 0), arrowprops=dict(arrowstyle="->", color=RAD, lw=1.1))
    a.text(w + 0.58 * np.cos(th) + 0.05, 0.6 * np.sin(th), r"$\gamma_{\alpha\beta}$", color=RAD, fontsize=8.5)
    a.annotate("", xy=(w + 0.6, 0), xytext=(w, 0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    a.add_patch(Arc((w, 0), 0.5, 0.5, theta1=0, theta2=np.degrees(th), color=LOAD, lw=1.0))
    a.text(w + 0.3, 0.1, r"$\theta$", color=LOAD, fontsize=9)
    a.text(-0.9, -0.62, r"зерно $\alpha_1$", color=SUB, fontsize=8); a.text(0.35, -0.62, r"зерно $\alpha_2$", color=SUB, fontsize=8)
    a.set_xlim(-1.85, 1.85); a.set_ylim(-0.75, 0.75); a.set_aspect("equal"); a.axis("off"); label(a, "а", 0.05, 0.95)
    ratio = np.linspace(0, 2, 300)
    ct = ratio / 2
    S = 0.5 * (2 + ct) * (1 - ct) ** 2
    b.plot(ratio, S, color=INK)
    b.set_xlabel(r"$\gamma_{\alpha\alpha}/\gamma_{\alpha\beta}$"); b.set_ylabel(r"$\Delta G^*_{\text{гр}}/\Delta G^*_{\text{гом}}$")
    b.text(0.1, 0.12, "энергия границы\nидёт в дело", color=SUB, fontsize=7.8)
    b.text(0.95, 0.62, "при отношении 2 граница\n«смачивается»: барьера нет", color=SUB, fontsize=7.8)
    label(b, "б")
    fig.tight_layout(w_pad=1.5)
    save(fig, "f6_het")


def f_growth():
    """а) профиль концентрации у растущей пластинки (рост лимитирован диффузией); б) утолщение ∝ √t, удлинение ∝ t."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.45))
    x = np.linspace(-1, 6, 600)
    cb, ce, c0 = 1.0, 0.12, 0.3
    L = 1.6
    c = np.where(x < 0, cb, c0 - (c0 - ce) * np.exp(-x / L * 1.0))
    a.plot(x, c, color=INK)
    a.fill_between(x[x < 0], 0, cb, color=HYD_F, alpha=0.6)
    a.axhline(c0, color=DIM, lw=0.6, ls=":"); a.axhline(ce, color=DIM, lw=0.6, ls=":")
    a.text(5.9, c0 + 0.03, r"$c_0$ (пересыщение)", ha="right", fontsize=8, color=SUB)
    a.text(5.9, ce - 0.08, r"$c_e$ (равновесие у границы)", ha="right", fontsize=8, color=SUB)
    a.text(-0.5, 0.55, r"$c_\beta$", color=CIRC, ha="center")
    a.annotate("", xy=(L, 0.62), xytext=(0, 0.62), arrowprops=dict(arrowstyle="<->", color=LOAD, lw=1.0))
    a.text(L / 2, 0.66, r"$\sim\!\sqrt{Dt}$", color=LOAD, ha="center", fontsize=8.5)
    a.annotate("", xy=(0.45, 0.45), xytext=(-0.05, 0.45), arrowprops=dict(arrowstyle="->", color=CIRC, lw=1.4))
    a.text(0.5, 0.42, "граница\nдвижется", color=CIRC, fontsize=7.6, va="top")
    a.set_xlabel(r"$x$"); a.set_ylabel(r"$c$"); a.set_ylim(0, 1.05); label(a, "а")
    t = np.logspace(-1, 3, 200)
    b.loglog(t, 0.3 * np.sqrt(t), color=CIRC, label=r"толщина $\propto \sqrt t$")
    b.loglog(t, 0.1 * t, color=RAD, label=r"длина $\propto t$")
    b.set_xlabel(r"время $t$"); b.set_ylabel("размер"); b.legend(loc="upper left"); label(b, "б")
    b.text(0.95, 0.12, "торец всё время встречает\nсвежий пересыщенный раствор", transform=b.transAxes, ha="right", fontsize=7.6, color=SUB)
    fig.tight_layout(w_pad=2.0)
    save(fig, "f6_growth")


def f_jmak():
    """Колмогоров–Джонсон–Мел–Аврами: f = 1 − exp(−(kt)^n) и прямые в координатах Аврами."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.3))
    t = np.logspace(-1.5, 1, 400)
    for n, c in ((1, DIM), (2, CIRC), (3, PLAST), (4, RAD)):
        f = 1 - np.exp(-(t) ** n)
        a.plot(t, f, color=c, label=fr"$n = {n}$")
        m = (f > 1e-3) & (f < 0.999)
        b.plot(np.log(t[m]), np.log(-np.log(1 - f[m])), color=c)
    a.set_xscale("log"); a.set_xlabel(r"$kt$"); a.set_ylabel(r"превращённая доля $f$"); a.legend(fontsize=7.6); label(a, "а")
    b.set_xlabel(r"$\ln t$"); b.set_ylabel(r"$\ln(-\ln(1-f))$"); label(b, "б")
    b.text(0.97, 0.08, r"наклон $= n$", transform=b.transAxes, ha="right", color=SUB, fontsize=8.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "f6_jmak")


def f_gt():
    """Эффект Гиббса–Томсона для гидрида: растворимость у частицы радиуса r выше в exp(2γV/rRT) раз (300 °C)."""
    fig, ax = plt.subplots(figsize=(W2 + 0.4, 2.3))
    r = np.logspace(0, 2.3, 300) * 1e-9
    T = 573.15; V = 1.0e-5
    for g, c in ((0.1, HYDR), (0.2, CIRC), (0.3, RAD)):
        ax.plot(r * 1e9, np.exp(2 * g * V / (r * R * T)), color=c, label=fr"$\gamma = {g}$ Дж/м$^2$")
    ax.axhline(1, color=DIM, lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel(r"радиус частицы $r$, нм"); ax.set_ylabel(r"$c_r/c_\infty$"); ax.legend(fontsize=7.6)
    save(fig, "f6_gt")


def f_budget():
    """Энергетический бюджет на моль водорода при 300 °C (Э635): движущая сила на линии выпадения,
    упругая энергия тонкой пластинки, работа приложенного напряжения 100 МПа."""
    T = 573.15
    tssd = 10 ** (5.56 - 2240 / T); tssp = 10 ** (5.08 - 1680 / T)
    dmu = R * T * np.log(tssp / tssd) / 1000
    V = 1.0e-5; Eph, nu = 90e9, 0.34; EA, EC = 0.0458, 0.0723
    w_bas = Eph * EA ** 2 / (1 - nu) * V / 1000
    w_pri = 0.5 * Eph / (1 - nu ** 2) * (EA ** 2 + EC ** 2 + 2 * nu * EA * EC) * V / 1000
    work = 100e6 * EC * V / 1000
    dwork = 100e6 * (EC - EA) * V / 1000
    items = [("движущая сила на линии TSSP\n" + r"$RT\ln(C_{\mathrm{TSSP}}/C_{\mathrm{TSSD}})$", dmu, HYDR),
             ("упругая энергия тонкой пластинки,\nплоскость $\\parallel$ базису", w_bas, CIRC),
             ("то же, плоскость $\\parallel c$", w_pri, RAD),
             ("работа растяжения 100 МПа:\n$\\sigma\\varepsilon_c V$", work, LOAD),
             ("разница работ для зёрен\n$c \\parallel \\sigma$ и $c \\perp \\sigma$", dwork, PLAST)]
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.5))
    y = np.arange(len(items))[::-1]
    for yy, (lab_, v, c) in zip(y, items):
        ax.barh(yy, v, color=c, height=0.6)
        ax.text(v * 1.15, yy, f"{v:.2f}" if v >= 0.1 else f"{v * 1000:.0f} Дж/моль", va="center", fontsize=8.5, color=c)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=7.8)
    ax.set_xscale("log"); ax.set_xlim(0.01, 30); ax.set_xlabel("кДж на моль водорода (300 \\textdegree C)")
    save(fig, "f6_budget")
    return dict(dmu=dmu, w_bas=w_bas, w_pri=w_pri, work=work, dwork=dwork, S=tssp / tssd, tssd=tssd, tssp=tssp)


if __name__ == "__main__":
    f_cnt(); f_rate(); f_het(); f_growth(); f_jmak(); f_gt()
    print({k: round(v, 3) for k, v in f_budget().items()})
