"""Рисунки главы «Фазовые диаграммы»: энергия Гиббса фаз чистого циркония и схема P–T; изоморфная диаграмма
(идеальные растворы, Cu–Ni) с правилом рычага; эвтектическая диаграмма, построенная из кривых G(x) выпуклой
оболочкой; кривые охлаждения и доли фаз при затвердевании."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from scipy.optimize import brentq

from style import *

R = 8.314


def f_unary():
    """а) G(T) α-, β-Zr и расплава относительно α: пересечения — температуры превращений (863 и 1855 °C);
    б) схема P–T диаграммы циркония (α, β, ω, жидкость; положения линий при высоких давлениях примерные)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.6))
    T = np.linspace(300, 2400, 400)
    Tab, Hab = 1136.0, 4.0e3                      # α→β: 863 °C, теплота ≈ 4 кДж/моль
    Tm, Hm = 2128.0, 21.0e3                       # плавление (β→L)
    Gb = Hab * (1 - T / Tab)                      # G_β − G_α при постоянных ΔH, ΔS
    Gl = Gb + Hm * (1 - T / Tm)
    a.plot(T - 273.15, np.zeros_like(T), color=CIRC, label=r"$\alpha$ (ГПУ)")
    a.plot(T - 273.15, Gb / 1e3, color=RAD, label=r"$\beta$ (ОЦК)")
    a.plot(T - 273.15, Gl / 1e3, color=LOAD, label="жидкость")
    lo = np.minimum(0, np.minimum(Gb, Gl))
    a.plot(T - 273.15, lo / 1e3, color=INK, lw=3.2, alpha=0.18, label="устойчивая фаза")
    for Tt, nm in ((Tab, "863"), (Tm, "1855")):
        a.axvline(Tt - 273.15, color=DIM, lw=0.6, ls=":")
        a.text(Tt - 273.15 + 20, 4.2, nm + r"\,\textdegree C", fontsize=7.6, color=SUB)
    a.set_ylim(-5, 5); a.set_xlabel(r"$T$, \textdegree C"); a.set_ylabel(r"$G - G_\alpha$, кДж/моль")
    a.legend(fontsize=7.2, loc="lower left"); label(a, "а")
    # схема P–T
    P = np.linspace(0, 10, 50)
    Tab_P = 863 - 45 * P                                         # α/β: наклон отрицательный
    b.plot(P[P <= 5.5], Tab_P[P <= 5.5], color=INK)
    tp = (5.5, 863 - 45 * 5.5)
    b.plot([2.2, tp[0]], [20, tp[1]], color=INK)                   # α/ω
    b.plot([tp[0], 10], [tp[1], tp[1] + 60], color=INK)            # ω/β
    b.plot([0, 10], [1855, 1855 + 10 * 25], color=INK)             # β/L
    b.plot(*tp, "o", color=RAD, ms=4)
    b.text(1.0, 350, r"$\alpha$ (ГПУ)", fontsize=10, color=CIRC)
    b.text(6.8, 300, r"$\omega$", fontsize=10, color=PLAST)
    b.text(3.5, 1300, r"$\beta$ (ОЦК)", fontsize=10, color=RAD)
    b.text(3.5, 2020, "жидкость", fontsize=9, color=LOAD)
    b.text(tp[0] + 0.1, tp[1] - 230, "тройная\nточка", fontsize=7.4, color=RAD)
    b.plot([0, 0], [863, 863], "o", color=INK, ms=3)
    b.set_xlim(0, 10); b.set_ylim(0, 2250)
    b.set_xlabel(r"давление, ГПа"); b.set_ylabel(r"$T$, \textdegree C"); label(b, "б")
    b.text(9.8, 90, "схема", fontsize=7.4, color=DIM, ha="right")
    fig.tight_layout(w_pad=2.0)
    save(fig, "d1_unary")


# ----------------------------------------------------------------------------------------------------------------
# изоморфная система: идеальные жидкость и твёрдый раствор (Cu–Ni)
TA, HA = 1358.0, 13.1e3        # Cu
TB, HB = 1728.0, 17.5e3        # Ni


def iso_lines(T):
    kA = np.exp(-HA * (1 - T / TA) / (R * T))
    kB = np.exp(-HB * (1 - T / TB) / (R * T))
    xs = (1 - kA) / (kB - kA)
    return xs, kB * xs                                         # солидус, ликвидус (доля Ni)


def f_isomorph():
    """а) диаграмма Cu–Ni по модели идеальных растворов, конода и правило рычага для сплава 40 % Ni при 1250 °C;
    б) кривые G(x) жидкости и твёрдого раствора при этой температуре и их общая касательная."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.7))
    T = np.linspace(TA + 0.01, TB - 0.01, 300)
    xs, xl = iso_lines(T)
    a.plot(np.r_[0, xl, 1] * 100, np.r_[TA, T, TB] - 273.15, color=LOAD, label="ликвидус")
    a.plot(np.r_[0, xs, 1] * 100, np.r_[TA, T, TB] - 273.15, color=CIRC, label="солидус")
    T1 = 1250 + 273.15
    s1, l1 = iso_lines(T1)
    x0 = 0.43
    a.plot([l1 * 100, s1 * 100], [1250, 1250], color=INK, lw=1.0)
    a.plot([l1 * 100, s1 * 100], [1250, 1250], "o", color=INK, ms=3.5)
    a.plot(x0 * 100, 1250, "s", color=RAD, ms=4)
    a.axvline(x0 * 100, color=RAD, lw=0.6, ls=":")
    a.text(l1 * 100 - 1, 1262, r"$x_L$", ha="right", fontsize=8); a.text(s1 * 100 + 1, 1262, r"$x_S$", fontsize=8)
    a.text(8, 1370, "жидкость L", fontsize=8.5); a.text(52, 1105, r"твёрдый раствор $\alpha$", fontsize=8.5, ha="center")
    sm, lm = iso_lines(1400 + 273.15); xm = 50 * (sm + lm)
    a.annotate(r"$L+\alpha$", xy=(xm, 1400), xytext=(xm + 10, 1290), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=SUB, lw=0.7))
    fl = (s1 - x0) / (s1 - l1)
    a.text(2, 1465, f"доля жидкости $= \\dfrac{{x_S - x_0}}{{x_S - x_L}} = {fl:.2f}$", fontsize=7.6, color=RAD, ha="left")
    a.set_xlim(0, 100); a.set_ylim(1000, 1500)
    a.set_xlabel(r"Ni, ат.\,\%"); a.set_ylabel(r"$T$, \textdegree C"); a.legend(fontsize=7.4, loc="lower right"); label(a, "а")
    x = np.linspace(1e-4, 1 - 1e-4, 800)
    mix = R * T1 * (x * np.log(x) + (1 - x) * np.log(1 - x))
    GS = mix                                                        # твёрдые чистые — отсчёт
    GL = mix + (1 - x) * HA * (1 - T1 / TA) + x * HB * (1 - T1 / TB)
    b.plot(x * 100, GL / 1e3, color=LOAD, label=r"$G_L$")
    b.plot(x * 100, GS / 1e3, color=CIRC, label=r"$G_\alpha$")
    gl = np.interp(l1, x, GL); gs = np.interp(s1, x, GS)
    sl = (gs - gl) / (s1 - l1)
    xx = np.linspace(0, 1, 10)
    b.plot(xx * 100, (gl + sl * (xx - l1)) / 1e3, color=INK, lw=0.8, ls="--", label="общая касательная")
    b.plot([l1 * 100, s1 * 100], [gl / 1e3, gs / 1e3], "o", color=INK, ms=3.5)
    b.axvline(l1 * 100, color=DIM, lw=0.5, ls=":"); b.axvline(s1 * 100, color=DIM, lw=0.5, ls=":")
    b.set_xlabel(r"Ni, ат.\,\%"); b.set_ylabel(r"$G$, кДж/моль"); b.legend(fontsize=7.4, loc="center")
    b.set_ylim(-12, 4); b.text(97, 3.0, r"1250\,\textdegree C", ha="right", fontsize=8); label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "d2_isomorph")
    return l1, s1, fl


# ----------------------------------------------------------------------------------------------------------------
# эвтектика: идеальная жидкость, твёрдый регулярный раствор с сильным расслоением (одна функция → α и β)
EA, EB = 1200.0, 1000.0        # температуры плавления A и B, К
SM = 10.0                      # энтропия плавления, Дж/(моль·К)
OMS, OML = 16e3, 0.0           # параметры взаимодействия твёрдого раствора и жидкости


def G_phases(x, T):
    mix = R * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
    GS = mix + OMS * x * (1 - x)
    GL = mix + OML * x * (1 - x) + (1 - x) * SM * (EA - T) + x * SM * (EB - T)
    return GS, GL


def lower_hull(x, g):
    """Индексы нижней выпуклой оболочки точек (x, g), x по возрастанию."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2 and (g[h[-1]] - g[h[-2]]) * (x[i] - x[h[-2]]) >= (g[i] - g[h[-2]]) * (x[h[-1]] - x[h[-2]]):
            h.pop()
        h.append(i)
    return np.array(h)


def eutectic_map(nx=1201, Ts=None):
    """Для каждой T: интервалы двухфазных областей (концы коноды и фазы) по выпуклой оболочке min(G_S, G_L)."""
    x = np.linspace(1e-5, 1 - 1e-5, nx)
    Ts = np.linspace(500, 1250, 451) if Ts is None else Ts
    res = []
    for T in Ts:
        GS, GL = G_phases(x, T)
        X = np.r_[x, x]; Gall = np.r_[GS, GL]; ph = np.r_[np.zeros(nx, int), np.ones(nx, int)]
        o = np.argsort(X, kind="stable"); X, Gall, ph = X[o], Gall[o], ph[o]
        h = lower_hull(X, Gall)
        ties = []
        dx = x[1] - x[0]
        for i, j in zip(h[:-1], h[1:]):
            if X[j] - X[i] > 2.5 * dx:
                ties.append((X[i], X[j], ph[i], ph[j]))
        res.append((T, ties))
    return res


def f_eutectic():
    """а) кривые G(x) при температуре чуть ниже эвтектической: общая касательная к двум ветвям G_S (α и β);
    б) диаграмма A–B, полученная выпуклой оболочкой при каждой температуре (жидкость — идеальный раствор,
    твёрдые α и β — один регулярный раствор с расслоением)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.75))
    res = eutectic_map()
    # граничные линии
    pts = {"L/L+α": [], "L+α/α": [], "L/L+β": [], "L+β/β": [], "α/α+β": [], "α+β/β": []}
    TE = None
    for T, ties in res:
        for x1, x2, p1, p2 in ties:
            if p1 == 0 and p2 == 0:                       # α + β (обе — твёрдый раствор)
                pts["α/α+β"].append((x1, T)); pts["α+β/β"].append((x2, T))
            elif p1 == 0 and p2 == 1:                     # α + L
                pts["L+α/α"].append((x1, T)); pts["L/L+α"].append((x2, T))
            elif p1 == 1 and p2 == 0:                     # L + β
                pts["L/L+β"].append((x1, T)); pts["L+β/β"].append((x2, T))
        if any(p1 == 0 and p2 == 0 for *_, p1, p2 in ties) and TE is None:
            pass
    sol = [T for T, ties in res if any(p1 == 0 and p2 == 0 for *_, p1, p2 in ties)]
    TE = max(sol)
    col = {"L/L+α": LOAD, "L/L+β": LOAD, "L+α/α": CIRC, "L+β/β": CIRC, "α/α+β": CIRC, "α+β/β": CIRC}
    for k, v in pts.items():
        v = np.array(v)
        if len(v):
            o = np.argsort(v[:, 1])
            b.plot(v[o, 0] * 100, v[o, 1] - 273.15, color=col[k], lw=1.4)
    eut = [t for T, ties in res if abs(T - TE) < 1e-9 for t in ties]
    xa = min(t[0] for t in eut); xb = max(t[1] for t in eut)
    b.plot([xa * 100, xb * 100], [TE - 273.15] * 2, color=INK, lw=1.2)
    # эвтектический состав: где сходятся ликвидусы
    la = np.array(pts["L/L+α"]); lb = np.array(pts["L/L+β"])
    xe = 0.5 * (la[np.argmin(la[:, 1]), 0] + lb[np.argmin(lb[:, 1]), 0])
    b.plot(xe * 100, TE - 273.15, "o", color=RAD, ms=4)
    b.text(xe * 100, TE - 273.15 - 45, "E", color=RAD, ha="center", fontsize=8.5)
    b.text(45, 880, "L", fontsize=10)
    b.text(3, 560, r"$\alpha$", fontsize=10, color=CIRC); b.text(93, 520, r"$\beta$", fontsize=10, color=CIRC)
    b.text(45, 330, r"$\alpha+\beta$", fontsize=10, ha="center")
    b.text(17, 760, r"$L+\alpha$", fontsize=8.5); b.text(72, 650, r"$L+\beta$", fontsize=8.5)
    b.set_xlim(0, 100); b.set_ylim(227, 977)
    b.set_xlabel(r"B, ат.\,\%"); b.set_ylabel(r"$T$, \textdegree C"); label(b, "б")
    # G(x) при T чуть ниже эвтектики
    T1 = TE - 25
    x = np.linspace(1e-4, 1 - 1e-4, 1500)
    GS, GL = G_phases(x, T1)
    a.plot(x * 100, GS / 1e3, color=CIRC, label=r"$G_S$ (ветви $\alpha$ и $\beta$)")
    a.plot(x * 100, GL / 1e3, color=LOAD, label=r"$G_L$")
    X = np.r_[x, x]; Gall = np.r_[GS, GL]; o = np.argsort(X, kind="stable"); X, Gall = X[o], Gall[o]
    h = lower_hull(X, Gall)
    a.plot(X[h] * 100, Gall[h] / 1e3, color=INK, lw=0.8, ls="--", label="нижняя выпуклая оболочка")
    a.set_xlabel(r"B, ат.\,\%"); a.set_ylabel(r"$G$, кДж/моль"); a.legend(fontsize=7.2, loc="upper center")
    a.set_ylim(GS.min() / 1e3 - 0.5, 3.4); label(a, "а")
    a.text(98, GS.min() / 1e3 - 0.45, f"{T1 - 273.15:.0f}" + r"\,\textdegree C", ha="right", fontsize=8)
    fig.tight_layout(w_pad=2.0)
    save(fig, "d3_eutectic")
    return TE, xe, xa, xb, pts


def liquid_fraction(x0, pts, TE, xa, xb, xe):
    """Доля жидкости f_L(T) для сплава x0 < xe (до эвтектики) по правилу рычага по рассчитанным линиям."""
    la = np.array(sorted(pts["L/L+α"], key=lambda p: p[1])); sa = np.array(sorted(pts["L+α/α"], key=lambda p: p[1]))
    def fL(T):
        if T <= TE:
            return 0.0
        xl = np.interp(T, la[:, 1], la[:, 0]); xs = np.interp(T, sa[:, 1], sa[:, 0])
        if x0 <= xs:
            return 0.0
        if x0 >= xl:
            return 1.0
        return (x0 - xs) / (xl - xs)
    return fL


def f_cooling(TE, xe, xa, xb, pts):
    """а) кривые охлаждения: чистый A — площадка; сплав до эвтектики — излом на ликвидусе и площадка при T_E;
    эвтектический сплав — одна площадка; б) доли фаз при затвердевании сплава 20 % B; в) структура этого сплава:
    первичные зёрна α в эвтектической смеси α + β (схема)."""
    fig = plt.figure(figsize=(W1, 2.55))
    a = fig.add_axes([0.08, 0.17, 0.36, 0.78]); b = fig.add_axes([0.53, 0.17, 0.25, 0.78]); c = fig.add_axes([0.80, 0.17, 0.2, 0.78])
    cp, L, k, Tenv = 30.0, SM * 1100.0, 0.05, 300.0     # Дж/(моль·К), Дж/моль, 1/с, К
    x0 = 0.20
    fLa = liquid_fraction(x0, pts, TE, xa, xb, xe)
    for case, c_, nm in (("A", INK, "чистый A"), ("hypo", CIRC, "20\\,\\% B"), ("E", RAD, f"эвтектика ({xe * 100:.0f}\\,\\% B)")):
        T = 1300.0; Q_lat = 0.0; t = 0.0; ts = []; Ts = []
        dt = 0.05
        # затвердевание как функция T: f_L(T); в изотермических точках — расходуем «запас» теплоты
        while T > 650 and t < 120:
            if case == "A":
                fl_now = 1.0 if T > EA else 0.0
                Tstop = EA
            elif case == "E":
                fl_now = 1.0 if T > TE else 0.0
                Tstop = TE
            else:
                fl_now = fLa(T)
                Tstop = TE
            qout = k * cp * (T - Tenv) * dt
            if case in ("A", "E") and abs(T - Tstop) < 1e-9 and Q_lat < L:
                Q_lat += qout                          # площадка: тепло уходит, T стоит
            elif case == "hypo" and abs(T - TE) < 1e-9 and Q_lat < L * fLa(TE + 1e-6):
                Q_lat += qout
            else:
                # теплоёмкость с учётом скрытой теплоты (на наклонном участке)
                dfl = (fLa(T + 0.5) - fLa(T - 0.5)) if case == "hypo" else 0.0
                ceff = cp + L * dfl
                Tn = T - qout / ceff
                if case in ("A", "E") and T > Tstop >= Tn:
                    Tn = Tstop
                if case == "hypo" and T > TE >= Tn:
                    Tn = TE
                T = Tn
            t += dt; ts.append(t); Ts.append(T)
        a.plot(ts, np.array(Ts) - 273.15, color=c_, label=nm)
    a.axhline(TE - 273.15, color=DIM, lw=0.5, ls=":"); a.text(44, TE - 273.15 + 12, r"$T_E$", ha="right", fontsize=8, color=SUB)
    a.set_xlabel(r"время, с"); a.set_ylabel(r"$T$, \textdegree C"); a.legend(fontsize=7.2, loc="upper right")
    a.set_xlim(0, 45); label(a, "а")
    T = np.linspace(TE - 60, 1250, 600)
    fl = np.array([fLa(t) for t in T])
    xeu = (x0 - xa) / (xe - xa)                       # доля эвтектики = доля жидкости у T_E+
    fe = fLa(TE + 1e-6)
    b.plot(T - 273.15, fl, color=LOAD)
    b.plot(T - 273.15, np.where(T > TE, 1 - fl, 1 - fe), color=CIRC)
    b.plot(T - 273.15, np.where(T > TE, 0, fe), color=RAD)
    b.text(800, 0.9, "жидкость", color=LOAD, fontsize=7.6)
    b.text(375, 0.9, r"первичная $\alpha$", color=CIRC, fontsize=7.6)
    b.text(470, 0.24, "эвтектика", color=RAD, fontsize=7.6)
    b.set_xlabel(r"$T$, \textdegree C"); b.set_ylabel("доля")
    b.set_ylim(-0.03, 1.05); label(b, "б")
    # схема структуры
    rng = np.random.default_rng(4)
    c.add_patch(Circle((0, 0), 1, fc="white", ec=SUB, lw=0.8))
    clip = Circle((0, 0), 1, transform=c.transData)
    for y in np.arange(-1.2, 1.2, 0.07):
        ln, = c.plot([-1.2, 1.2], [y - 0.3, y + 0.3], color=RAD, lw=0.9)
        ln.set_clip_path(clip)
    cents = []
    for _ in range(5000):
        p = rng.uniform(-0.8, 0.8, 2)
        if np.hypot(*p) < 0.78 and all(np.hypot(*(p - q)) > 0.5 for q in cents):
            cents.append(p)
        if len(cents) == 8:
            break
    for p in cents:
        ang = np.linspace(0, 2 * np.pi, 30)
        r = 0.2 * (1 + 0.18 * rng.normal(size=30)); r = np.convolve(np.r_[r[-3:], r, r[:3]], np.ones(5) / 5, "same")[3:-3]
        poly = Polygon(np.c_[p[0] + r * np.cos(ang), p[1] + r * np.sin(ang)], fc=HYD_F, ec=CIRC, lw=0.6)
        c.add_patch(poly); poly.set_clip_path(clip)
    c.set_xlim(-1.05, 1.05); c.set_ylim(-1.05, 1.05); c.set_aspect("equal"); c.axis("off"); label(c, "в", 0.12, 0.9)
    save(fig, "d4_cooling")
    return fe


if __name__ == "__main__":
    f_unary()
    l1, s1, fl = f_isomorph(); print(f"Cu–Ni 1250 °C, 40 % Ni: x_L = {l1:.3f}, x_S = {s1:.3f}, доля жидкости {fl:.2f}")
    TE, xe, xa, xb, pts = f_eutectic(); print(f"эвтектика: T_E = {TE - 273.15:.0f} °C, x_E = {xe:.3f}, x_α = {xa:.3f}, x_β = {xb:.3f}")
    fe = f_cooling(TE, xe, xa, xb, pts); print(f"доля эвтектики в сплаве 20 % B: {fe:.2f}")
