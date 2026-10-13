"""Рисунки главы 5: мысленный опыт Эшелби; упругая энергия включения от формы (расчёт методом Фурье, плоская
деформация) против точных пределов; поле гидростатического напряжения у пластинки; энергия тонкой пластинки δ-гидрида
от ориентации и работа приложенного напряжения."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from style import *

EA, EC = 0.0458, 0.0723          # несоответствие δ-гидрида: в базисной плоскости и вдоль c (Carpenter 1973)
NU = 0.34
MU = 1.0
LAM = 2 * MU * NU / (1 - 2 * NU)
E = 2 * MU * (1 + NU)
E_PHYS = 90e9                    # Па, порядок модуля Юнга Zr при 300–400 °C
V_H_HYD = 1.0e-5                 # объём гидрида на моль водорода, м³ (ZrH1.66: 16.4 см³/моль Zr)


def solve(chi, exx, eyy, ezz, n):
    """Плоская деформация, однородная изотропная упругость, периодическое поле n×n; среднее напряжение — ноль.
    chi — индикатор включения, ε* = diag(exx, eyy, ezz) внутри. → (энергия на площадь включения, поля e, σ)."""
    k = 2 * np.pi * np.fft.fftfreq(n)
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K2 = KX ** 2 + KY ** 2; K2[0, 0] = 1.0
    ex_, ey_, ez_ = exx * chi, eyy * chi, ezz * chi
    tr = ex_ + ey_ + ez_
    txx = LAM * tr + 2 * MU * ex_
    tyy = LAM * tr + 2 * MU * ey_
    Txx, Tyy = np.fft.fft2(txx), np.fft.fft2(tyy)
    t1, t2 = KX * Txx, KY * Tyy                          # t = k·τ̂ (τ_xy = 0)
    nx, ny = KX / np.sqrt(K2), KY / np.sqrt(K2)
    c = (LAM + MU) / (LAM + 2 * MU)
    nt = nx * t1 + ny * t2
    v1 = (t1 - c * nx * nt) / (MU * K2)
    v2 = (t2 - c * ny * nt) / (MU * K2)
    exx_t = np.real(np.fft.ifft2(KX * v1)); eyy_t = np.real(np.fft.ifft2(KY * v2))
    exy_t = np.real(np.fft.ifft2(0.5 * (KX * v2 + KY * v1)))
    # среднюю деформацию подбираем так, чтобы среднее напряжение было нулевым (тело свободно в плоскости в среднем)
    m_ex, m_ey, m_ez = ex_.mean(), ey_.mean(), ez_.mean()
    A = np.array([[LAM + 2 * MU, LAM], [LAM, LAM + 2 * MU]])
    rhs = np.array([LAM * (m_ex + m_ey + m_ez) + 2 * MU * m_ex, LAM * (m_ex + m_ey + m_ez) + 2 * MU * m_ey])
    mx, my = np.linalg.solve(A, rhs)
    exx_t += mx; eyy_t += my
    e1, e2, e3, e12 = exx_t - ex_, eyy_t - ey_, -ez_, exy_t
    trE = e1 + e2 + e3
    w = 0.5 * (LAM * trE ** 2 + 2 * MU * (e1 ** 2 + e2 ** 2 + e3 ** 2 + 2 * e12 ** 2))
    s1 = LAM * trE + 2 * MU * e1; s2 = LAM * trE + 2 * MU * e2; s3 = LAM * trE + 2 * MU * e3
    return w.sum() / chi.sum(), dict(sxx=s1, syy=s2, szz=s3, sxy=2 * MU * e12)


def ellipse_mask(n, a, b):
    y, x = np.mgrid[0:n, 0:n] - n / 2 + 0.5
    return ((x / a) ** 2 + (y / b) ** 2 <= 1.0).astype(float)


def thin_limit(e_in1, e_in2):
    """Энергия тонкой пластинки на объём: деформации в её плоскости стеснены, по нормали — свободны."""
    return 0.5 * E / (1 - NU ** 2) * (e_in1 ** 2 + e_in2 ** 2 + 2 * NU * e_in1 * e_in2)


def f_shape():
    """Энергия включения на единицу объёма от отношения осей эллипса (расчёт Фурье, 1024², длинная полуось 80 ячеек)."""
    n, a = 1024, 80.0
    asp = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.06])
    ref = thin_limit(EA, EA)
    out = {"dil": [], "basal": [], "prism": []}
    for r in asp:
        chi = ellipse_mask(n, a, max(a * r, 2.5))
        e0 = (2 * EA + EC) / 3
        out["dil"].append(solve(chi, e0, e0, e0, n)[0] / thin_limit(e0, e0))
        out["basal"].append(solve(chi, EA, EC, EA, n)[0] / ref)          # пластинка вдоль x = a, нормаль — c
        out["prism"].append(solve(chi, EC, EA, EA, n)[0] / ref)          # пластинка вдоль c, нормаль — a
    fig, ax = plt.subplots(figsize=(W2 + 0.9, 2.6))
    ax.plot(asp, out["dil"], "o-", color=SUB, ms=3.5, label=r"чистое расширение $\varepsilon^*\mathbf I$")
    ax.plot(asp, out["basal"], "o-", color=CIRC, ms=3.5, label=r"гидрид, пластинка $\parallel$ базису")
    ax.plot(asp, out["prism"], "o-", color=RAD, ms=3.5, label=r"гидрид, пластинка $\parallel c$")
    ax.axhline(1.0, color=CIRC, ls=":", lw=0.9)
    ax.axhline(thin_limit(EC, EA) / ref, color=RAD, ls=":", lw=0.9)
    ax.text(0.98, 1.0 + 0.04, "предел тонкой пластинки", color=CIRC, fontsize=7.6, ha="right")
    ax.text(0.98, thin_limit(EC, EA) / ref + 0.04, "предел тонкой пластинки", color=RAD, fontsize=7.6, ha="right")
    ax.set_xscale("log"); ax.set_xlim(0.05, 1.1)
    ax.set_xlabel(r"отношение осей $b/a$ (1 --- круг, $\to 0$ --- пластинка)")
    ax.set_ylabel(r"$w / w_{\text{баз}}$")
    ax.legend(loc="center left", fontsize=7.6, bbox_to_anchor=(0.0, 0.62))
    save(fig, "f5_shape")
    return asp, out


def f_field():
    """Гидростатическое напряжение σ_h вокруг тонкой пластинки δ-гидрида (b/a = 0.1, нормаль вдоль c)."""
    n, a = 512, 60.0
    chi = ellipse_mask(n, a, 6.0)
    _, F = solve(chi, EA, EC, EA, n)
    sh = (F["sxx"] + F["syy"] + F["szz"]) / 3 * (E_PHYS / E) / 1e6        # МПа: расчёт при μ = 1, пересчёт к E_PHYS
    c = n // 2; w = 130
    S = sh[c - w // 2 - 15:c + w // 2 + 15, c - w:c + w]
    S = np.where(chi[c - w // 2 - 15:c + w // 2 + 15, c - w:c + w] > 0, np.nan, S)
    fig, ax = plt.subplots(figsize=(W1 * 0.72, 2.3))
    cmap = LinearSegmentedColormap.from_list("d", [CIRC, "#ffffff", RAD])
    lim = 350
    im = ax.imshow(np.clip(S, -lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), extent=(-w, w, w // 2 + 15, -w // 2 - 15))
    ax.add_patch(Ellipse((0, 0), 2 * a, 12, facecolor=HYD_F, edgecolor=CIRC, lw=0.8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0, -22, "гидрид", color=CIRC, ha="center", fontsize=8)
    ax.text(a + 4, -14, "растяжение\nу торца", color=RAD, fontsize=7.6, ha="left")
    ax.text(0, 30, "сжатие у широких граней", color=CIRC, fontsize=7.6, ha="center")
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label(r"$\sigma_h$ в матрице, МПа", fontsize=8)
    save(fig, "f5_field")
    return float(np.nanmax(S)), float(np.nanmin(S))


def f_orient():
    """а) энергия тонкой пластинки δ-гидрида на объём от угла между нормалью пластинки и осью c;
    б) работа приложенного растяжения −σ ε*_xx от угла между осью c зерна и направлением растяжения."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.6))
    th = np.linspace(0, 90, 181)
    t = np.radians(th)
    ett = EA * np.cos(t) ** 2 + EC * np.sin(t) ** 2
    w = thin_limit(ett, EA) * E_PHYS / E / 1e6      # МДж/м³
    a.plot(th, w, color=INK)
    phi = np.degrees(np.arctan((2 / np.sqrt(3)) * 1.593 / 7))
    a.axvline(phi, color=HYDR, lw=0.8, ls="--")
    a.text(phi + 2, w.min() + 5, r"$\{10\bar 1 7\}$: $%.0f^\circ$" % phi, color=HYDR, fontsize=8)
    a.set_xlabel(r"угол нормали пластинки к оси $c$, \textdegree"); a.set_ylabel(r"$w$, МДж/м$^3$")
    ax2 = a.twinx(); ax2.set_ylim(np.array(a.get_ylim()) * 1e6 * V_H_HYD / 1000); ax2.set_ylabel(r"кДж на моль H", color=SUB)
    ax2.spines["right"].set_visible(True); ax2.tick_params(colors=SUB)
    label(a, "а")
    psi = np.linspace(0, 90, 181); p = np.radians(psi)
    exx = EA * np.sin(p) ** 2 + EC * np.cos(p) ** 2
    for s, c in ((50, HYDR), (100, LOAD), (150, RAD)):
        b.plot(psi, s * exx, color=c, label=fr"$\sigma = {s}$ МПа")
    b.set_xlabel(r"угол оси $c$ к направлению растяжения $\psi$, \textdegree")
    b.set_ylabel(r"$\sigma\,\varepsilon^*_{xx}$, МДж/м$^3$"); b.legend(loc="upper right"); label(b, "б")
    fig.tight_layout(w_pad=2.2)
    save(fig, "f5_orient")
    return w.min(), w.max()


def f_eshelby():
    """Мысленный опыт Эшелби: вырезать, превратить свободно, сжать обратно и вставить, отпустить."""
    fig, axs = plt.subplots(1, 4, figsize=(W1, 1.75))
    titles = ["1. вырезать", "2. превратить свободно", "3. сжать и вставить", "4. отпустить"]
    for k, ax in enumerate(axs):
        ax.add_patch(Rectangle((-2, -1.3), 4, 2.6, facecolor=METAL, edgecolor=METAL_E))
        if k == 0:
            ax.add_patch(Ellipse((0, 0), 1.6, 0.8, facecolor="white", edgecolor=SUB, ls="--"))
            ax.add_patch(Ellipse((0, -2.0), 1.6, 0.8, facecolor=HYD_F, edgecolor=CIRC))
        if k == 1:
            ax.add_patch(Ellipse((0, 0), 1.6, 0.8, facecolor="white", edgecolor=SUB, ls="--"))
            ax.add_patch(Ellipse((0, -2.05), 1.6 * 1.05, 0.8 * 1.25, facecolor=HYD_F, edgecolor=CIRC))
            ax.text(1.15, -2.05, r"$\varepsilon^*$", color=CIRC, fontsize=9)
        if k == 2:
            ax.add_patch(Ellipse((0, 0), 1.6, 0.8, facecolor=HYD_F, edgecolor=CIRC))
            for ang in np.radians(np.arange(0, 360, 45)):
                x0, y0 = 1.25 * np.cos(ang), 0.75 * np.sin(ang)
                ax.annotate("", xy=(0.82 * np.cos(ang), 0.42 * np.sin(ang)), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=LOAD, lw=0.9))
            ax.text(0, -1.0, r"$-\mathbf C:\varepsilon^*$", color=LOAD, fontsize=8.5, ha="center")
        if k == 3:
            for rr, al in ((1.9, 0.08), (1.45, 0.14), (1.15, 0.22)):
                ax.add_patch(Ellipse((0, 0), 1.6 * rr, 0.8 * rr * 1.25, facecolor=PLAST, alpha=al, edgecolor="none"))
            ax.add_patch(Ellipse((0, 0), 1.6 * 1.02, 0.8 * 1.12, facecolor=HYD_F, edgecolor=CIRC))
            ax.text(0, -1.0, "напряжения", color=PLAST, fontsize=7.5, ha="center")
        ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.75, 1.45); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(titles[k], fontsize=8.5, color=INK, loc="center")
    fig.tight_layout(w_pad=0.2)
    save(fig, "f5_eshelby")


if __name__ == "__main__":
    asp, out = f_shape()
    print("форма:", {k: [round(v, 3) for v in vv] for k, vv in out.items()})
    print("пределы: базис 1, призма", round(thin_limit(EC, EA) / thin_limit(EA, EA), 3))
    print("σh max/min, МПа:", f_field())
    print("w min/max, МДж/м³:", f_orient())
    f_eshelby()
