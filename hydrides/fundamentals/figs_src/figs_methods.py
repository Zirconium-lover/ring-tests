"""Рисунки главы «Как изучают структуру»: спектр рентгеновской трубки, условие Брэгга, обратная решётка и сфера
Эвальда, расчётные дифрактограммы α-Zr и δ-ZrH (порошок и текстура трубы), уширение пиков и метод sin²ψ,
кривые ДСК из линий растворимости, оптическое изображение тонкой пластинки и травление."""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

from style import *
import texture_model as tm

LAM = 1.5406            # Cu Kα1, Å
A_ZR, C_ZR = 3.232, 5.147
A_DH = 4.78


def f_spectrum():
    """Спектр рентгеновской трубки с медным анодом при 30 кВ: тормозное излучение (закон Крамерса) и линии Kα, Kβ;
    никелевый фильтр гасит Kβ (край поглощения Ni между Kα и Kβ)."""
    fig, ax = plt.subplots(figsize=(W2 + 0.9, 2.45))
    V = 30.0
    lmin = 12.398 / V
    lam = np.linspace(0.3, 2.2, 2000)
    brems = np.where(lam > lmin, (lam / lmin - 1) / lam ** 2, 0.0)
    brems = brems / brems.max()
    g = lambda l0, w, h: h * np.exp(-((lam - l0) / w) ** 2)
    I = brems + g(1.5406, 0.004, 7.0) + g(1.5444, 0.004, 3.5) + g(1.3922, 0.004, 1.5)
    ax.plot(lam, I, color=INK, lw=1.2, label="без фильтра")
    edge = 1.4881                                     # K-край Ni, Å
    mu = np.where(lam < edge, 6.0 * (lam / edge) ** 3, 0.7 * (lam / edge) ** 3)
    ax.plot(lam, I * np.exp(-mu), color=CIRC, lw=1.2, label="с Ni-фильтром")
    ax.axvline(lmin, color=DIM, lw=0.6, ls=":"); ax.text(lmin + 0.02, 6.6, r"$\lambda_{\min} = hc/eV$", color=SUB, fontsize=8)
    ax.text(1.56, 7.2, r"$K\alpha$", color=RAD); ax.text(1.33, 1.9, r"$K\beta$", color=RAD)
    ax.text(0.75, 1.15, "тормозное\nизлучение", color=SUB, fontsize=8)
    ax.set_xlabel(r"длина волны $\lambda$, \AA"); ax.set_ylabel("интенсивность (усл.)"); ax.set_ylim(0, 8); ax.legend(fontsize=7.6)
    save(fig, "m1_spectrum")


def f_bragg():
    """а) условие Брэгга: разность хода 2d sinθ; б) обратная решётка и сфера (окружность) Эвальда."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.55))
    d = 1.0
    for j in range(3):
        a.plot([-3, 3], [-j * d, -j * d], color=METAL_E, lw=1.0)
        for x in np.arange(-3.0, 3.1, 1.0):
            a.plot(x, -j * d, "o", color=METAL_E, mec=INK, mew=0.4, ms=6)
    th = np.radians(30)
    for j, c in ((0, CIRC), (1, RAD)):
        y0 = -j * d; x0 = 0.0
        xin = x0 - 2.6 * np.cos(th); yin = y0 + 2.6 * np.sin(th)
        a.annotate("", xy=(x0, y0), xytext=(xin, yin), arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        a.annotate("", xy=(x0 + 2.6 * np.cos(th), y0 + 2.6 * np.sin(th)), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    for sx in (-1, 1):                     # перпендикуляры из точки отражения верхнего луча на нижний луч
        foot = (sx * d * np.sin(th) * np.cos(th), -d + d * np.sin(th) ** 2)
        a.plot([0, foot[0]], [0, foot[1]], color=SUB, lw=0.6, ls="--")
        a.plot([0, foot[0]], [-d, foot[1]], color=LOAD, lw=2.6, solid_capstyle="butt", zorder=5)
    a.text(0.5, -0.95, r"$d\sin\theta$", color=LOAD, fontsize=8.5)
    a.text(-1.0, -0.95, r"$d\sin\theta$", color=LOAD, fontsize=8.5, ha="right")
    a.add_patch(Arc((0, 0), 1.2, 1.2, theta1=150, theta2=180, color=SUB, lw=0.8)); a.text(-0.85, 0.08, r"$\theta$", color=SUB)
    a.annotate("", xy=(2.75, 0), xytext=(2.75, -d), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8)); a.text(2.82, -0.55, r"$d$")
    a.text(-2.9, 1.65, r"$n\lambda = 2d\sin\theta$", fontsize=10, color=INK)
    a.set_xlim(-3.1, 3.2); a.set_ylim(-2.3, 1.95); a.set_aspect("equal"); a.axis("off"); label(a, "а", 0.05, 1.0)
    # сфера Эвальда через узел 0 и узел g = (0, 2): центр C на серединном перпендикуляре
    gx, gy = np.meshgrid(np.arange(-5, 3), np.arange(-1, 5))
    b.plot(gx.ravel(), gy.ravel(), "o", color=DIM, ms=3)
    O = np.array([0.0, 0.0]); Gp = np.array([0.0, 2.0]); Cc = np.array([-2.5, 1.0])
    k = np.linalg.norm(O - Cc)
    b.add_patch(Circle(Cc, k, fill=False, ec=CIRC, lw=1.0))
    b.plot(*Cc, "+", color=CIRC, ms=6)
    b.annotate("", xy=O, xytext=Cc, arrowprops=dict(arrowstyle="->", color=CIRC, lw=1.3))
    b.annotate("", xy=Gp, xytext=Cc, arrowprops=dict(arrowstyle="->", color=RAD, lw=1.3))
    b.annotate("", xy=Gp, xytext=O, arrowprops=dict(arrowstyle="->", color=LOAD, lw=1.3))
    b.plot(*O, "o", color=INK, ms=4.5); b.plot(*Gp, "o", color=RAD, ms=4.5)
    b.text(-1.35, 0.05, r"$\mathbf k_0$", color=CIRC, fontsize=9)
    b.text(-1.45, 1.75, r"$\mathbf k$", color=RAD, fontsize=9)
    b.text(0.15, 0.85, r"$\mathbf g_{hkl}$", color=LOAD, fontsize=9)
    b.text(0.15, -0.4, "0", color=INK, fontsize=8.5)
    b.text(-5.2, 4.25, r"$\mathbf k - \mathbf k_0 = \mathbf g_{hkl}$", fontsize=9)
    b.text(-4.9, -2.0, "сфера Эвальда, $|\\mathbf k| = 1/\\lambda$", color=CIRC, fontsize=7.6)
    b.set_xlim(-5.4, 2.6); b.set_ylim(-2.3, 4.6); b.set_aspect("equal"); b.axis("off"); label(b, "б", 0.05, 1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "m2_bragg")


def fZr(s):
    return 40 * (0.55 * np.exp(-12 * s ** 2) + 0.45 * np.exp(-0.3 * s ** 2))


def fH(s):
    return np.exp(-20 * s ** 2)


def hcp_lines(hmax=4, lmax=6, with_hkl=False):
    """(d, I) для ГПУ Zr: структурный фактор с атомами (0,0,0), (1/3,2/3,1/2), повторяемость, Лоренц–поляризация."""
    out = {}
    for h, k, l in itertools.product(range(-hmax, hmax + 1), range(-hmax, hmax + 1), range(-lmax, lmax + 1)):
        if (h, k, l) == (0, 0, 0):
            continue
        inv_d2 = 4 / 3 * (h * h + h * k + k * k) / A_ZR ** 2 + l * l / C_ZR ** 2
        d = 1 / np.sqrt(inv_d2)
        if LAM / (2 * d) >= 1:
            continue
        F = 1 + np.exp(2j * np.pi * (h / 3 + 2 * k / 3 + l / 2))
        F2 = abs(F) ** 2
        if F2 < 1e-6:
            continue
        key = round(d, 4)
        # угол нормали к плоскости с осью c
        beta = np.degrees(np.arctan2(np.sqrt(4 / 3 * (h * h + h * k + k * k)) / A_ZR, abs(l) / C_ZR))
        out.setdefault(key, [d, 0.0, (h, k, l), beta])
        out[key][1] += F2
    lines = []
    for d, F2m, hkl, beta in out.values():
        s = 1 / (2 * d)
        th = np.arcsin(LAM / (2 * d))
        lp = (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))
        lines.append((np.degrees(2 * th), F2m * fZr(s) ** 2 * lp * np.exp(-0.5 * s ** 2), beta, d, hkl))
    lines.sort(key=lambda r: r[0])
    if with_hkl:
        return [r[:4] for r in lines], [r[4] for r in lines]
    return [r[:4] for r in lines]


def fcc_hyd_lines(hmax=5, occ=0.83):
    out = {}
    zr = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    hs = list(itertools.product([.25, .75], repeat=3))
    for h, k, l in itertools.product(range(-hmax, hmax + 1), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue
        d = A_DH / np.sqrt(h * h + k * k + l * l)
        if LAM / (2 * d) >= 1:
            continue
        s = 1 / (2 * d)
        F = sum(fZr(s) * np.exp(2j * np.pi * (h * x + k * y + l * z)) for x, y, z in zr) + \
            sum(occ * fH(s) * np.exp(2j * np.pi * (h * x + k * y + l * z)) for x, y, z in hs)
        F2 = abs(F) ** 2
        if F2 < 1e-3:
            continue
        key = round(d, 4)
        out.setdefault(key, [d, 0.0])
        out[key][1] += F2
    lines = []
    for d, F2 in out.values():
        s = 1 / (2 * d)
        th = np.arcsin(LAM / (2 * d))
        lp = (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))
        lines.append((np.degrees(2 * th), F2 * lp * np.exp(-0.5 * s ** 2), d))
    return sorted(lines)


def profile(tt, lines, w=0.12):
    y = np.zeros_like(tt)
    for t0, I in lines:
        y += I / (1 + ((tt - t0) / w) ** 2)
    return y


def hkil_str(h, k, l):
    i = -(h + k)
    return "".join(str(v) if v >= 0 else r"\bar{%d}" % -v for v in (h, k, i, l))


def f_xrd():
    """Расчётные дифрактограммы (Cu Kα): а) α-Zr, порошок; б) α-Zr, грань трубы, нормаль — радиус (текстура);
    в) α-Zr с 5 % δ-гидрида (по объёму): слабые пики гидрида рядом с пиками матрицы."""
    tt = np.linspace(25, 80, 6000)
    zr, hkl_list = hcp_lines(with_hkl=True)
    hy = fcc_hyd_lines()
    fig, axs = plt.subplots(3, 1, figsize=(W1, 4.6), sharex=True)
    I0 = profile(tt, [(t, I) for t, I, b, d in zr])
    norm = I0.max()
    axs[0].plot(tt, I0 / norm * 100, color=INK, lw=1.0)
    lab = {}
    for t, I, b, d in zr:
        if 28 < t < 80 and I / max(i for _, i, _, _ in zr) > 0.04:
            lab[t] = (I, b, d)
    names = {}
    for t, (I, b, d) in lab.items():
        # подбор индексов для подписи по d
        best = None
        for h, k, l in itertools.product(range(0, 4), range(0, 4), range(0, 6)):
            if (h, k, l) == (0, 0, 0) or k > h:
                continue
            dd = 1 / np.sqrt(4 / 3 * (h * h + h * k + k * k) / A_ZR ** 2 + l * l / C_ZR ** 2)
            if abs(dd - d) < 1e-3 and (best is None or h + k + l < sum(best)):
                best = (h, k, l)
        names[t] = best
    for t, (I, b, d) in lab.items():
        y = profile(np.array([t]), [(tt_, ii) for tt_, ii, _, _ in zr])[0] / norm * 100
        if names[t]:
            axs[0].text(t, y + 4, "$" + hkil_str(*names[t]) + "$", ha="center", fontsize=6.6, color=SUB, rotation=90, va="bottom")
    axs[0].set_ylim(0, 135); axs[0].set_ylabel("I, \\%"); axs[0].text(78.5, 105, r"$\alpha$-Zr, порошок", ha="right", fontsize=8.5)
    # плотность полюсов вдоль R для схематичной текстуры трубы (texture_model): её и меряет съёмка с поверхности
    G = tm.sample()
    w = [tm.pole_density(G, tm.family(*hkl), (1, 0, 0)) for hkl in hkl_list]
    It = profile(tt, [(t, I * ww) for (t, I, b, d), ww in zip(zr, w)])
    axs[1].plot(tt, It / It.max() * 100, color=CIRC, lw=1.0)
    axs[1].plot(tt, I0 / norm * 100, color=DIM, lw=0.6, ls="--")
    axs[1].set_ylim(0, 135); axs[1].set_ylabel("I, \\%")
    axs[1].text(78.5, 105, r"$\alpha$-Zr, грань трубы (нормаль --- радиус)", ha="right", fontsize=8.5, color=CIRC)
    Ih = profile(tt, [(t, I) for t, I, d in hy])
    vol = 0.05
    mix = (1 - vol) * I0 / norm + vol * Ih / norm * (A_ZR ** 2 * np.sqrt(3) / 2 * C_ZR / 2) ** 2 / (A_DH ** 3 / 4) ** 2
    axs[2].plot(tt, mix * 100, color=INK, lw=1.0)
    axs[2].plot(tt, vol * Ih / norm * (A_ZR ** 2 * np.sqrt(3) / 2 * C_ZR / 2) ** 2 / (A_DH ** 3 / 4) ** 2 * 100, color=HYDR, lw=1.0)
    for t, I, d in hy:
        hkl = None
        for h, k, l in itertools.product(range(0, 4), repeat=3):
            if (h, k, l) != (0, 0, 0) and h >= k >= l and abs(A_DH / np.sqrt(h * h + k * k + l * l) - d) < 1e-3:
                hkl = (h, k, l); break
        if hkl and t < 80:
            axs[2].text(t, 13, r"$\delta(%d%d%d)$" % hkl, color=HYDR, fontsize=6.8, ha="center", rotation=90, va="bottom")
    axs[2].set_ylim(0, 35); axs[2].set_ylabel("I, \\%")
    axs[2].text(78.5, 29, r"$\alpha$-Zr + 5\,\% $\delta$-гидрида (зелёный --- вклад гидрида)", ha="right", fontsize=8.5)
    axs[2].set_xlabel(r"$2\theta$, \textdegree{} (Cu $K\alpha$)")
    for ax, s in zip(axs, "абв"):
        label(ax, s, -0.04, 0.85)
    fig.tight_layout(h_pad=0.4)
    save(fig, "m3_xrd")
    return zr, hy


def f_peaks():
    """а) уширение пика (10-11) α-Zr по Шерреру для кристаллитов 1 мкм, 50 нм и 15 нм; б) метод sin²ψ:
    межплоскостное расстояние от наклона образца при одноосном напряжении 0 и ±200 МПа."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.4))
    t0 = 36.52; th = np.radians(t0 / 2)
    tt = np.linspace(34.5, 38.5, 1000)
    for L, c in ((1000, INK), (50, CIRC), (15, RAD)):
        beta = 0.9 * LAM / (L * 10 * np.cos(th))                  # рад (L в нм → Å)
        w = max(np.degrees(beta), 0.05)
        a.plot(tt, 1 / (1 + ((tt - t0) / (w / 2)) ** 2) * (0.05 / w) ** 0.0, color=c, label=f"{L:g} нм" if L < 1000 else "1 мкм")
    a.set_xlabel(r"$2\theta$, \textdegree"); a.set_ylabel("I (норм.)"); a.legend(fontsize=7.6, title="кристаллит", title_fontsize=7.6); label(a, "а")
    E, nu = 99e3, 0.34
    d0 = 2.458
    psi = np.linspace(0, 60, 50); s2 = np.sin(np.radians(psi)) ** 2
    for s, c in ((0, INK), (200, RAD), (-200, CIRC)):
        dd = d0 * (1 + (1 + nu) / E * s * s2 - nu / E * s)
        b.plot(s2, dd, color=c, label=fr"$\sigma = {s:+d}$ МПа" if s else r"$\sigma = 0$")
    b.set_xlabel(r"$\sin^2\psi$"); b.set_ylabel(r"$d_{10\bar11}$, \AA"); b.legend(fontsize=7.6); label(b, "б")
    b.text(0.4, 2.4567, r"наклон $\propto \sigma(1+\nu)/E$", color=SUB, fontsize=8)
    fig.tight_layout(w_pad=2.0)
    save(fig, "m4_peaks")


def f_dsc():
    """Расчётные кривые ДСК образца Э635 со 160 ppm (10 К/мин, инерция прибора 0.5 мин): при нагреве — эндотермический
    пик растворения, оканчивается на TSSD; при охлаждении — экзотермический пик выпадения, начинается на TSSP."""
    TSSD, TSSP = (5.56, 2.24), (5.08, 1.68)
    cl = lambda T, bk: 10 ** (bk[0] - bk[1] * 1000 / (T + 273.15))
    H = 160.0; rate = 10.0
    T = np.linspace(150, 480, 3000)
    Cd = np.minimum(H, cl(T, TSSD))
    q_h = -np.gradient(Cd, T) * rate
    Tc = T[::-1]
    Cc = np.minimum(H, cl(Tc, TSSP))
    q_c = np.gradient(Cc, Tc) * rate          # выделение тепла: вверх
    def lag(q, dt_K):
        k = np.exp(-np.arange(0, 400) * (T[1] - T[0]) / dt_K); k /= k.sum()
        return np.convolve(q, k)[:len(q)]
    fig, ax = plt.subplots(figsize=(W1 * 0.72, 2.5))
    qh = lag(q_h, 5.0); qc = lag(q_c, 5.0)
    ax.plot(T, qh, color=RAD)
    ax.plot(Tc, qc, color=CIRC)
    ax.text(160, -9.5, "нагрев: растворение\n(эндотермический)", color=RAD, fontsize=7.6, va="top")
    ax.text(152, 16.8, "охлаждение: выпадение\n(экзотермический)", color=CIRC, fontsize=7.6, va="top")
    Td = 2240 / (5.56 - np.log10(H)) - 273.15; Tp = 1680 / (5.08 - np.log10(H)) - 273.15
    ax.axvline(Td, color=RAD, lw=0.6, ls=":"); ax.axvline(Tp, color=CIRC, lw=0.6, ls=":")
    ax.text(Td + 3, qh.min() * 0.85, f"TSSD\n{Td:.0f}\\,\\textdegree C", color=RAD, fontsize=7.8)
    ax.text(Tp + 4, qc.max() * 0.85, f"TSSP\n{Tp:.0f}\\,\\textdegree C", color=CIRC, fontsize=7.8, ha="left")
    ax.axhline(0, color=DIM, lw=0.6)
    ax.set_xlabel(r"$T$, \textdegree C"); ax.set_ylabel(r"тепловой поток (усл.), экзо $\uparrow$")
    save(fig, "m5_dsc")


def f_optics():
    """а) как выглядит пластинка гидрида толщиной 0.3 мкм: оптика (λ = 0.55 мкм, NA = 0.5) размывает её до ~0.7 мкм,
    травление расширяет след на 1–2 мкм с каждой стороны; б) видимая доля площади от утолщения травлением при
    густоте линий 10 и 15 мм/мм² (истинная доля гидрида ~1 %)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.4))
    x = np.linspace(-6, 6, 3000)
    t = 0.3
    plate = (np.abs(x) < t / 2).astype(float)
    fw = 0.61 * 0.55 / 0.5
    psf = np.exp(-4 * np.log(2) * (x / fw) ** 2); psf /= psf.sum()
    img = np.convolve(plate, psf, mode="same")
    a.fill_between(x, 0, plate, color=HYD_F, alpha=0.8, label="пластинка 0.3 мкм")
    a.plot(x, img / img.max(), color=CIRC, label="оптика без травления")
    for e, c in ((1.0, PLAST), (2.0, RAD)):
        groove = (np.abs(x) < t / 2 + e).astype(float)
        a.plot(x, np.convolve(groove, psf, mode="same"), color=c, label=f"травление, +{e:g} мкм на сторону")
    a.set_xlabel("поперёк пластинки, мкм"); a.set_ylabel("затемнение (норм.)"); a.set_ylim(0, 1.95); a.legend(fontsize=6.9, loc="upper center")
    label(a, "а")
    e = np.linspace(0, 3, 100)
    for dens, c in ((10, CIRC), (15, RAD)):
        y = dens * 1e-3 * (t + 2 * e) * 100
        b.plot(e, y, color=c)
        b.text(3.05, y[-1], f"{dens} мм/мм$^2$", color=c, fontsize=7.6, va="center")
    b.set_xlim(0, 3.9); b.set_ylim(0, 11)
    b.set_xlabel("утолщение травлением с каждой стороны, мкм"); b.set_ylabel("видимая доля площади, \\%")
    label(b, "б")
    fig.tight_layout(w_pad=2.0)
    save(fig, "m6_optics")


if __name__ == "__main__":
    f_spectrum(); f_bragg()
    zr, hy = f_xrd()
    print("Zr:", [(round(t, 2), round(b, 1)) for t, I, b, d in zr if t < 80][:12])
    print("δ:", [round(t, 2) for t, I, d in hy if t < 80])
    f_peaks(); f_dsc(); f_optics()
