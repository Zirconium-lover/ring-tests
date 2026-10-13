"""Рисунки главы «Текстура»: одно зерно в системе трубы и его полюсы; полюсные фигуры схематичной текстуры трубы;
обратные полюсные фигуры; коэффициенты текстуры из рентгеновской съёмки и факторы Кернса по ним; схема влияния
коэффициента Q при прокатке труб."""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import gaussian_filter

from style import *
import texture_model as tm

LAM = 1.5406
A_ZR, C_ZR = 3.232, 5.147


def project(v):
    """Равноплощадная проекция полюсов v (R, T, A) с центром в R: радиус √2·sin(α/2) ≤ 1."""
    v = v / np.linalg.norm(v, axis=1)[:, None]
    al = np.arccos(np.clip(v[:, 0], -1, 1))
    rho = np.sqrt(2) * np.sin(al / 2)
    az = np.arctan2(v[:, 2], v[:, 1])
    return rho * np.cos(az), rho * np.sin(az)


def pf_frame(ax, lab=True):
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=SUB, lw=0.8))
    ax.plot([-1, 1], [0, 0], color=DIM, lw=0.4); ax.plot([0, 0], [-1, 1], color=DIM, lw=0.4)
    if lab:
        ax.text(1.06, 0, "T", color=INK, fontsize=8, va="center")
        ax.text(0, 1.06, "A", color=INK, fontsize=8, ha="center")
        ax.text(0.05, 0.05, "R", color=INK, fontsize=7.5)
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18); ax.set_aspect("equal"); ax.axis("off")


def pf_density(v, nal=36, naz=72, sig=1.0):
    """Плотность полюсов на полярной сетке (в долях случайной, m.r.d.)."""
    al = np.degrees(np.arccos(np.clip(v[:, 0], -1, 1)))
    az = np.degrees(np.arctan2(v[:, 2], v[:, 1]))
    ae = np.linspace(0, 90, nal + 1); ze = np.linspace(-180, 180, naz + 1)
    H, _, _ = np.histogram2d(al, az, bins=[ae, ze])
    om = (np.cos(np.radians(ae[:-1])) - np.cos(np.radians(ae[1:])))[:, None] * np.radians(ze[1] - ze[0])
    D = (H / len(v)) / (om / (2 * np.pi))
    D = gaussian_filter(D, sig, mode=("nearest", "wrap"))
    ac = 0.5 * (ae[1:] + ae[:-1]); zc = 0.5 * (ze[1:] + ze[:-1])
    ac = np.r_[0.0, ac, 90.0]
    D = np.vstack([np.full((1, naz), D[0].mean()), D, D[-1:]])
    zc = np.r_[zc, zc[0] + 360]; D = np.hstack([D, D[:, :1]])
    AL, AZ = np.meshgrid(np.radians(ac), np.radians(zc), indexing="ij")
    rho = np.sqrt(2) * np.sin(AL / 2)
    return rho * np.cos(AZ), rho * np.sin(AZ), D


LEV = [0.5, 1, 2, 3, 4, 5, 6, 8, 10]
BOUNDS = [0] + LEV + [50]
NORM = BoundaryNorm(BOUNDS, 256)


def pf_plot(ax, v, title):
    X, Y, D = pf_density(v)
    cs = ax.contourf(X, Y, D, levels=BOUNDS, cmap="Blues", norm=NORM)
    ax.contour(X, Y, D, levels=[1], colors=[SUB], linewidths=0.5)
    pf_frame(ax)
    ax.text(0, -1.32, title, ha="center", fontsize=8.5)
    return cs


def hex_prism(ax, g, scale=1.0, origin=(0, 0, 0)):
    """ГПУ-призма в осях образца; оси рисунка: x = T, y = A, z = R."""
    ca = tm.CA
    ang = np.radians(np.arange(0, 360, 60))
    hexv = np.stack([np.cos(ang), np.sin(ang), np.zeros(6)], 1)
    bot = hexv - [0, 0, ca / 2]; top = hexv + [0, 0, ca / 2]
    P = lambda p: (g @ np.asarray(p)) * scale + origin            # → (R, T, A)
    xyz = lambda p: (P(p)[1], P(p)[2], P(p)[0])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = [[xyz(p) for p in bot], [xyz(p) for p in top]]
    for i in range(6):
        j = (i + 1) % 6
        faces.append([xyz(bot[i]), xyz(bot[j]), xyz(top[j]), xyz(top[i])])
    cols = [HYD_F, HYD_F] + [METAL] * 6
    ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors=SUB, linewidths=0.6, alpha=0.55))
    c0, c1 = xyz([0, 0, 0]), xyz([0, 0, ca * 1.15])
    ax.quiver(*c0, *(np.subtract(c1, c0)), color=RAD, lw=1.6, arrow_length_ratio=0.15)
    ax.text(*c1, r"  $c$", color=RAD, fontsize=9)
    a0 = xyz([1.0, 0, 0]); a1 = xyz([1.6, 0, 0])
    ax.quiver(*a0, *(np.subtract(a1, a0)), color=CIRC, lw=1.0, arrow_length_ratio=0.25)
    ax.text(*a1, r"$a_1$", color=CIRC, fontsize=8)


def f_single():
    """а) одно зерно (ГПУ-призма) в осях трубы R, T, A; ось c наклонена от R к T на 35°;
    б) его полюсы (0002), {10-10}, {11-20} на полюсной фигуре; в) полюсы (0002) 400 зёрен трубы."""
    fig = plt.figure(figsize=(W1, 2.35))
    a = fig.add_axes([0.0, 0.0, 0.36, 1.0], projection="3d")
    g = (tm.rot(1, np.radians([-8.0])) @ tm.rot(2, np.radians([35.0])) @
         np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)[None])[0]
    hex_prism(a, g)
    L = 2.0
    for vec, name in (((0, 0, L), "R"), ((L, 0, 0), "T"), ((0, L, 0), "A")):
        a.quiver(0, 0, 0, *vec, color=INK, lw=0.9, arrow_length_ratio=0.08)
        a.text(*(np.array(vec) * 1.08), name, fontsize=9)
    a.set_xlim(-1.5, 1.8); a.set_ylim(-1.5, 1.8); a.set_zlim(-1.3, 2.0)
    a.set_box_aspect((1, 1, 1)); a.view_init(elev=16, azim=-62); a.set_axis_off()
    a.text2D(0.04, 0.92, "а", transform=a.transAxes, fontsize=10.5, fontweight="bold")
    b = fig.add_axes([0.37, 0.08, 0.30, 0.86])
    G1 = g[None]
    for (h, k, l), m, c, nm in (((0, 0, 1), "o", RAD, r"$(0002)$"), ((1, 0, 0), "s", CIRC, r"$\{10\bar10\}$"),
                               ((1, 1, 0), "^", HYDR, r"$\{11\bar20\}$")):
        X, Y = project(tm.poles(G1, h, k, l))
        b.plot(X, Y, m, color=c, ms=5.5, label=nm, mec="white", mew=0.4)
    pf_frame(b)
    b.legend(fontsize=7.0, loc="lower left", bbox_to_anchor=(-0.08, -0.12), handletextpad=0.2, borderaxespad=0)
    label(b, "б", 0.08, 0.97)
    c = fig.add_axes([0.69, 0.08, 0.30, 0.86])
    G = tm.sample(400, seed=11)
    X, Y = project(tm.poles(G, 0, 0, 1))
    c.plot(X, Y, ".", color=RAD, ms=2.0, alpha=0.7)
    pf_frame(c); label(c, "в", 0.08, 0.97)
    c.text(0, -1.32, r"$(0002)$, 400 зёрен", ha="center", fontsize=8)
    save(fig, "t1_single")


def f_polefigs(G=None, Gr=None):
    """Полюсные фигуры (равноплощадная проекция, центр — R): (0002), {10-10}, {11-20} холоднодеформированной трубы
    и {10-10} рекристаллизованной; уровни — в долях случайной текстуры."""
    G = tm.sample() if G is None else G
    Gr = tm.sample(kind="rxa") if Gr is None else Gr
    fig, axs = plt.subplots(1, 4, figsize=(W1, 1.95))
    cs = None
    for ax, (GG, hkl, t) in zip(axs, ((G, (0, 0, 1), r"$(0002)$"), (G, (1, 0, 0), r"$\{10\bar10\}$"),
                                     (G, (1, 1, 0), r"$\{11\bar20\}$"), (Gr, (1, 0, 0), r"$\{10\bar10\}$, рекрист."))):
        cs = pf_plot(ax, tm.poles(GG, *hkl), t)
    for ax, s in zip(axs, "абвг"):
        label(ax, s, 0.1, 0.95)
    fig.subplots_adjust(left=0.01, right=0.9, wspace=0.08, bottom=0.12, top=0.97)
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.66])
    cb = fig.colorbar(cs, cax=cax, ticks=LEV); cb.ax.tick_params(labelsize=6.5); cax.set_ylim(0, 10)
    cb.set_label("доли случайной", fontsize=7)
    save(fig, "t2_polefigs")


def ipf_density(G, d, nal=18, nph=6, sig=0.7):
    u = np.einsum("nji,j->ni", G, np.asarray(d, float))
    al, ph = tm.to_fundamental(u)
    ae = np.linspace(0, 90, nal + 1); pe = np.linspace(0, 30, nph + 1)
    H, _, _ = np.histogram2d(al, ph, bins=[ae, pe])
    om = (np.cos(np.radians(ae[:-1])) - np.cos(np.radians(ae[1:])))[:, None] * np.radians(pe[1] - pe[0])
    D = (H / len(u)) / (om / (np.pi / 6))
    D = gaussian_filter(D, sig, mode="reflect")
    ac = np.r_[0.0, 0.5 * (ae[1:] + ae[:-1]), 90.0]; pc = np.r_[0.0, 0.5 * (pe[1:] + pe[:-1]), 30.0]
    D = np.vstack([np.full((1, nph), D[0].mean()), D, D[-1:]]); D = np.hstack([D[:, :1], D, D[:, -1:]])
    AL, PH = np.meshgrid(np.radians(ac), np.radians(pc), indexing="ij")
    rho = np.sqrt(2) * np.sin(AL / 2)
    return rho * np.cos(PH), rho * np.sin(PH), D


def ipf_frame(ax):
    ax.add_patch(Wedge((0, 0), 1, 0, 30, fill=False, ec=SUB, lw=0.8))
    ax.text(-0.03, -0.02, r"$[0001]$", ha="right", va="top", fontsize=7)
    ax.text(1.0, -0.03, r"$\langle 2\bar1\bar10\rangle$", ha="center", va="top", fontsize=7)
    ax.text(np.cos(np.radians(30)), np.sin(np.radians(30)) + 0.03, r"$\langle 10\bar10\rangle$", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-0.25, 1.12); ax.set_ylim(-0.17, 0.66); ax.set_aspect("equal"); ax.axis("off")


def f_ipf(G=None):
    """Обратные полюсные фигуры трубы для направлений R, T, A: где в кристалле оказывается данное направление образца."""
    G = tm.sample() if G is None else G
    fig, axs = plt.subplots(1, 3, figsize=(W1, 1.55))
    cs = None
    for ax, (d, nm) in zip(axs, (((1, 0, 0), "радиус R"), ((0, 1, 0), "окружность T"), ((0, 0, 1), "ось трубы A"))):
        X, Y, D = ipf_density(G, d)
        cs = ax.contourf(X, Y, D, levels=BOUNDS, cmap="Blues", norm=NORM)
        ax.contour(X, Y, D, levels=[1], colors=[SUB], linewidths=0.5)
        ipf_frame(ax)
        ax.text(-0.2, 0.36, nm, fontsize=8.5, ha="left")
    for ax, s in zip(axs, "абв"):
        label(ax, s, 0.08, 0.86)
    fig.subplots_adjust(left=0.01, right=0.9, wspace=0.06, bottom=0.03, top=0.98)
    cax = fig.add_axes([0.915, 0.15, 0.012, 0.7])
    cb = fig.colorbar(cs, cax=cax, ticks=LEV); cb.ax.tick_params(labelsize=6.5); cax.set_ylim(0, 10)
    save(fig, "t3_ipf")


def reflections(tt_max):
    """Отражения α-Zr (Cu Kα) до 2θ = tt_max: уникальные направления нормалей (по одному от семейства)."""
    out = []
    for h, k, l in itertools.product(range(0, 6), range(0, 6), range(0, 10)):
        if (h, k, l) == (0, 0, 0) or k > h:
            continue
        F2 = abs(1 + np.exp(2j * np.pi * (h / 3 + 2 * k / 3 + l / 2))) ** 2
        if F2 < 1e-6:
            continue
        d = 1 / np.sqrt(4 / 3 * (h * h + h * k + k * k) / A_ZR ** 2 + l * l / C_ZR ** 2)
        if LAM / (2 * d) >= 1:
            continue
        tt = 2 * np.degrees(np.arcsin(LAM / (2 * d)))
        if tt > tt_max:
            continue
        n = tm.hkil_normal(h, k, l)
        if any(np.allclose(np.abs(n), np.abs(m[3]), atol=1e-6) for m in out):
            continue                                       # высший порядок того же направления
        out.append((tt, (h, k, l), d, n))
    return sorted(out, key=lambda r: r[0])


def voronoi_weights(normals, n=400000, seed=2):
    """Доли телесного угла фундаментального сектора, ближайшие к каждой нормали (ячейки Вороного)."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    al, ph = tm.to_fundamental(u)
    a, p = np.radians(al), np.radians(ph)
    U = np.stack([np.sin(a) * np.cos(p), np.sin(a) * np.sin(p), np.cos(a)], 1)
    Nr = []
    for v in normals:
        al1, ph1 = tm.to_fundamental(v[None])
        a1, p1 = np.radians(al1[0]), np.radians(ph1[0])
        Nr.append([np.sin(a1) * np.cos(p1), np.sin(a1) * np.sin(p1), np.cos(a1)])
    Nr = np.array(Nr)
    idx = np.argmax(U @ Nr.T, axis=1)
    return np.bincount(idx, minlength=len(normals)) / n, Nr, (U, idx)


def hkil(h, k, l):
    i = -(h + k)
    return "".join(str(v) if v >= 0 else r"\bar{%d}" % -v for v in (h, k, i, l))


def kearns_xrd(G, tt_max, d=(1, 0, 0)):
    refl = reflections(tt_max)
    P = np.array([tm.pole_density(G, tm.family(*r[1]), d) for r in refl])
    w, Nr, extra = voronoi_weights([r[3] for r in refl])
    cos2 = Nr[:, 2] ** 2
    f = np.sum(w * P * cos2) / np.sum(w * P)
    return refl, P, w, Nr, f, extra


def f_tc(G=None):
    """а) плотности полюсов (коэффициенты текстуры) отражений α-Zr при съёмке с поверхности трубы (нормаль R);
    б) ячейки Вороного в обратной полюсной фигуре, по которым суммируется фактор Кернса."""
    G = tm.sample() if G is None else G
    fR = tm.kearns(G)[0]
    refl, P, w, Nr, f140, (U, idx) = kearns_xrd(G, 140)
    _, _, _, _, f80, _ = kearns_xrd(G, 80)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 2.55), gridspec_kw=dict(width_ratios=[1.35, 1]))
    order = np.argsort(np.degrees(np.arccos(np.abs(Nr[:, 2]))))
    xs = np.arange(len(refl))
    beta = np.degrees(np.arccos(np.abs(Nr[:, 2])))
    a.bar(xs, P[order], color=[CIRC if refl[i][0] <= 80 else DIM for i in order], width=0.7)
    a.axhline(1, color=SUB, lw=0.6, ls=":")
    a.set_xticks(xs)
    a.set_xticklabels([r"$%s$\,\,(%.0f\textdegree)" % (hkil(*refl[i][1]), beta[i]) for i in order], fontsize=6.4, rotation=90)
    a.set_ylabel(r"$P_{hkl}$ (доли случайной)")
    a.text(0.98, 0.95, r"синие: $2\theta \le 80^\circ$; серые: до $140^\circ$", transform=a.transAxes, ha="right", va="top", fontsize=7.4)
    label(a, "а")
    # ячейки Вороного в секторе
    ii = np.argsort(-P)
    rk = np.empty(len(P), int); rk[ii] = np.arange(len(P))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(U), 30000, replace=False)
    al = np.arccos(U[sel, 2]); ph = np.arctan2(U[sel, 1], U[sel, 0])
    rho = np.sqrt(2) * np.sin(al / 2)
    vals = P[idx[sel]]
    sc = b.scatter(rho * np.cos(ph), rho * np.sin(ph), c=vals, s=1.2, cmap="Blues", vmin=0, vmax=P.max() * 1.05, rasterized=True)
    al1 = np.arccos(Nr[:, 2]); ph1 = np.arctan2(Nr[:, 1], Nr[:, 0]); r1 = np.sqrt(2) * np.sin(al1 / 2)
    b.plot(r1 * np.cos(ph1), r1 * np.sin(ph1), "o", color=RAD, ms=2.5)
    ipf_frame(b)
    b.text(0.5, -0.38, f"$f_R$ по рентгену: {f80:.2f} (до 80°), {f140:.2f} (до 140°);\nточное значение {fR:.2f}".replace("°", r"\textdegree"),
           ha="center", fontsize=7.6, transform=b.transData)
    b.set_ylim(-0.55, 0.66)
    label(b, "б", 0.05, 0.98)
    fig.tight_layout(w_pad=1.0)
    save(fig, "t4_tc")
    return f80, f140, fR, refl, P


def f_pilger():
    """Прокатка трубы (пильгер): отношение Q = ln(t0/t)/ln(D0/D). а) Q > 1 — в основном утоняется стенка,
    оси c уходят к радиусу; б) Q < 1 — в основном уменьшается диаметр, оси c уходят к окружности."""
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.3))
    for ax, (Q, D1, t1, chi, nm) in zip(axs, ((2.5, 0.86, 0.10, 20, r"$Q > 1$: утоняется стенка"),
                                            (0.5, 0.62, 0.17, 70, r"$Q < 1$: уменьшается диаметр"))):
        D0, t0 = 1.0, 0.24
        for x0, D, t, c, lab in ((-1.25, D0, t0, METAL_E, "до прохода"), (1.25, D1, t1, METAL_E, "после")):
            ax.add_patch(Wedge((x0, 0), D / 2 + t / 2, 0, 360, width=t, fc=METAL, ec=c))
            ax.text(x0, -0.86, lab, ha="center", fontsize=8)
        ax.annotate("", xy=(0.55, 0), xytext=(-0.55, 0), arrowprops=dict(arrowstyle="-|>", color=LOAD, lw=1.4))
        # оси c в стенке после прохода: по два отрезка (±χ от радиуса) в шести местах
        x0 = 1.25
        for th in np.radians(np.arange(0, 360, 60) + 30):
            r = D1 / 2
            cx, cy = x0 + r * np.cos(th), r * np.sin(th)
            for s in (-1, 1):
                ang = th + s * np.radians(chi)
                L = 0.11
                ax.plot([cx - L * np.cos(ang), cx + L * np.cos(ang)], [cy - L * np.sin(ang), cy + L * np.sin(ang)],
                        color=RAD, lw=1.3, solid_capstyle="round")
        ax.text(0, 0.72, nm, ha="center", fontsize=8.5)
        ax.set_xlim(-1.9, 1.9); ax.set_ylim(-0.98, 0.85); ax.set_aspect("equal"); ax.axis("off")
    label(axs[0], "а", 0.04, 0.95); label(axs[1], "б", 0.04, 0.95)
    fig.tight_layout(w_pad=0.5)
    save(fig, "t5_pilger")


if __name__ == "__main__":
    G = tm.sample(); Gr = tm.sample(kind="rxa")
    print("Кернс CWSR (R, T, A):", np.round(tm.kearns(G), 3), " RXA:", np.round(tm.kearns(Gr), 3))
    f_single(); f_polefigs(G, Gr); f_ipf(G)
    f80, f140, fR, refl, P = f_tc(G)
    print("f_R по рентгену до 80°:", round(f80, 3), "до 140°:", round(f140, 3), "точно:", round(fR, 3))
    for r, p in zip(refl, P):
        print(f"  {r[0]:6.2f}  {r[1]}  P = {p:.2f}")
    f_pilger()
