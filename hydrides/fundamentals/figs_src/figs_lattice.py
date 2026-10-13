"""Рисунки главы «Кристаллическая решётка»: ячейки ПК, ОЦК, ГЦК, ГПУ; плотные упаковки ABAB и ABCABC;
индексы Миллера в кубической ячейке; междоузлия ГЦК и решётка δ-гидрида (флюорит); стереографическая проекция."""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from style import *


def cube_edges(ax, s=1.0, color=SUB, lw=0.8):
    for a in itertools.product([0, s], repeat=3):
        for b in itertools.product([0, s], repeat=3):
            if sum(x != y for x, y in zip(a, b)) == 1 and a < b:
                ax.plot(*zip(a, b), color=color, lw=lw)


def atoms(ax, P, color=METAL_E, s=60, ec=INK):
    P = np.array(P, float)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, color=color, edgecolor=ec, linewidth=0.5, depthshade=False)


def setup3d(ax, title, azim=-58, below=False):
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=azim); ax.set_axis_off()
    if below:
        ax.text2D(0.5, -0.02, title, transform=ax.transAxes, fontsize=8.3, color=INK, ha="center", va="top")
    else:
        ax.set_title(title, fontsize=9, color=INK, loc="center", pad=-2)


def f_cells():
    """Элементарные ячейки: ПК, ОЦК, ГЦК, ГПУ — число атомов, координационное число, плотность упаковки."""
    fig = plt.figure(figsize=(W1, 2.15))
    corners = list(itertools.product([0, 1], repeat=3))
    ax = fig.add_subplot(1, 4, 1, projection="3d"); cube_edges(ax); atoms(ax, corners)
    setup3d(ax, "ПК\n1 атом, $z=6$, $f=0.52$", below=True)
    ax = fig.add_subplot(1, 4, 2, projection="3d"); cube_edges(ax); atoms(ax, corners); atoms(ax, [(0.5, 0.5, 0.5)], color=CIRC)
    setup3d(ax, "ОЦК\n2 атома, $z=8$, $f=0.68$", below=True)
    ax = fig.add_subplot(1, 4, 3, projection="3d"); cube_edges(ax); atoms(ax, corners)
    faces = [(0.5, 0.5, 0), (0.5, 0.5, 1), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0.5, 0.5), (1, 0.5, 0.5)]
    atoms(ax, faces, color=CIRC)
    setup3d(ax, "ГЦК\n4 атома, $z=12$, $f=0.74$", below=True)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    A, C = 0.62, 0.62 * 1.593
    ang = np.radians(np.arange(0, 360, 60) + 30)
    hx, hy = A * np.cos(ang) + 0.5, A * np.sin(ang) + 0.5
    for z in (0, C):
        ax.plot(np.r_[hx, hx[0]], np.r_[hy, hy[0]], z, color=SUB, lw=0.8)
    for x, y in zip(hx, hy):
        ax.plot([x, x], [y, y], [0, C], color=SUB, lw=0.8)
    P = [(x, y, z) for z in (0, C) for x, y in zip(np.r_[hx, 0.5], np.r_[hy, 0.5])]
    atoms(ax, P)
    r = A / np.sqrt(3)
    atoms(ax, [(0.5 + r * np.cos(np.radians(t)), 0.5 + r * np.sin(np.radians(t)), C / 2) for t in (90, 210, 330)], color=CIRC)
    ax.set_zlim(0, 1)
    setup3d(ax, "ГПУ\n6 атомов, $z=12$, $f=0.74$", below=True)
    fig.subplots_adjust(left=0, right=1, wspace=0.0)
    save(fig, "l1_cells")


def f_stacking():
    """Плотноупакованные слои: позиции A, B, C. ГПУ — ABAB, ГЦК — ABCABC (вид сверху и сбоку)."""
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.35), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axs[0]
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3) / 2])
    shifts = {"A": np.zeros(2), "B": (a1 + a2) / 3, "C": 2 * (a1 + a2) / 3}
    cols = {"A": METAL_E, "B": HYD_F, "C": "#f5c27a"}
    for key in ("A", "B", "C"):
        for i in range(-1, 5):
            for j in range(-1, 5):
                p = i * a1 + j * a2 + shifts[key]
                if 0 <= p[0] <= 4.2 and 0 <= p[1] <= 3.4:
                    if key == "A":
                        ax.add_patch(plt.Circle(p, 0.5, fc=cols[key], ec=INK, lw=0.5))
                    else:
                        ax.plot(*p, "o", ms=4.5, color=CIRC if key == "B" else LOAD)
    ax.text(1.0, 3.55, "слой A; ямки B (синие) и C (жёлтые)", fontsize=8, ha="left")
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.6, 3.9); ax.set_aspect("equal"); ax.axis("off"); label(ax, "а", 0.05, 1.0)
    for ax, seq, name, lab_ in ((axs[1], "ABABAB", "ГПУ (Zr): ABAB", "б"), (axs[2], "ABCABC", "ГЦК (гидрид): ABCABC", "в")):
        for k, s_ in enumerate(seq):
            x0 = {"A": 0.0, "B": 1 / 3, "C": 2 / 3}[s_]
            for i in range(4):
                ax.add_patch(plt.Circle((x0 + i, k * 0.82), 0.33, fc=cols[s_], ec=INK, lw=0.5))
            ax.text(-0.75, k * 0.82, s_, va="center", fontsize=8.5, color=SUB)
        ax.set_xlim(-0.9, 3.7); ax.set_ylim(-0.6, 5.0); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(name, fontsize=8.8, loc="center"); label(ax, lab_, 0.05, 1.0)
    fig.tight_layout(w_pad=0.3)
    save(fig, "l2_stacking")


def f_miller():
    """Плоскости (100), (110), (111) и направления [100], [110], [111] в кубической ячейке."""
    fig = plt.figure(figsize=(W1, 2.15))
    planes = [("(100)", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], [1, 0, 0]),
              ("(110)", [(1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1)], [1, 1, 0]),
              ("(111)", [(1, 0, 0), (0, 1, 0), (0, 0, 1)], [1, 1, 1])]
    for k, (name, poly, d) in enumerate(planes):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        cube_edges(ax)
        ax.add_collection3d(Poly3DCollection([poly], facecolor=CIRC, alpha=0.3, edgecolor=CIRC))
        ax.quiver(0, 0, 0, *(np.array(d) / np.linalg.norm(d) * 1.0), color=RAD, arrow_length_ratio=0.15, lw=1.4)
        ax.text(*(np.array(d) / np.linalg.norm(d) * 1.08 + np.array([0.02, 0.02, 0.05])), "[" + name[1:-1] + "]", color=RAD, fontsize=8.5)
        for v, t in (((1.18, 0, 0), "$x$"), ((0, 1.18, 0), "$y$"), ((0, 0, 1.15), "$z$")):
            ax.text(*v, t, color=SUB, fontsize=8)
        setup3d(ax, "плоскость " + name, azim=-58 if k < 2 else 25)
    fig.subplots_adjust(left=0, right=1, wspace=0.0)
    save(fig, "l3_miller")


def f_sites():
    """а) октаэдрические (жёлтые) и тетраэдрические (зелёные) междоузлия ГЦК; б) δ-гидрид: ГЦК подрешётка Zr,
    водород в тетраэдрических позициях (структура флюорита, ZrH2 при полном заполнении)."""
    fig = plt.figure(figsize=(W1, 2.6))
    corners = list(itertools.product([0, 1], repeat=3))
    faces = [(0.5, 0.5, 0), (0.5, 0.5, 1), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0.5, 0.5), (1, 0.5, 0.5)]
    ax = fig.add_subplot(1, 2, 1, projection="3d"); cube_edges(ax)
    atoms(ax, corners + faces, s=55)
    octa = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 1, 1), (1, 0.5, 1), (1, 1, 0.5),
            (0.5, 1, 0), (1, 0.5, 0), (0.5, 0, 1), (0, 0.5, 1), (1, 0, 0.5), (0, 1, 0.5)]
    atoms(ax, octa, color=LOAD, s=22, ec=LOAD)
    tetra = list(itertools.product([0.25, 0.75], repeat=3))
    atoms(ax, tetra, color=HYDR, s=22, ec=HYDR)
    setup3d(ax, "ГЦК: октаэдрические и тетраэдрические пустоты")
    ax.text2D(0.02, 0.95, "а", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    ax = fig.add_subplot(1, 2, 2, projection="3d"); cube_edges(ax)
    atoms(ax, corners + faces, s=80, color=METAL_E)
    atoms(ax, tetra, color=HYDR, s=26, ec=INK)
    for p in tetra:                                   # связи H — ближайшие Zr (тетраэдр)
        for q in corners + faces:
            if abs(np.linalg.norm(np.subtract(p, q)) - np.sqrt(3) / 4) < 1e-6:
                ax.plot(*zip(p, q), color=HYDR, lw=0.4, alpha=0.6)
    setup3d(ax, r"$\delta$-ZrH$_x$: Zr (ГЦК) + H в тетраэдрах")
    ax.text2D(0.02, 0.95, "б", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    fig.subplots_adjust(left=0, right=1, wspace=0.0)
    save(fig, "l4_sites")


def stereo(v):
    v = np.asarray(v, float); v = v / np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    return v[0] / (1 + v[2]), v[1] / (1 + v[2])


def hkil_normal(h, k, l, ca=1.593):
    """Нормаль к плоскости (hkil) ГПУ в декартовых осях (x ∥ a1, z ∥ c)."""
    a1 = np.array([1.0, 0, 0]); a2 = np.array([-0.5, np.sqrt(3) / 2, 0]); c = np.array([0, 0, ca])
    V = np.dot(a1, np.cross(a2, c))
    b1 = np.cross(a2, c) / V; b2 = np.cross(c, a1) / V; b3 = np.cross(a1, a2) / V
    return h * b1 + k * b2 + l * b3


def f_stereo():
    """а) построение стереографической проекции; б) стандартная проекция (001) кубического кристалла;
    в) проекция (0001) ГПУ циркония: базис, призмы, пирамиды {10-11} и {10-17}."""
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.35))
    ax = axs[0]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=SUB, lw=0.8)
    ax.axhline(0, color=DIM, lw=0.6)
    ax.plot([0], [-1], "o", color=INK, ms=4); ax.text(0.06, -1.12, "S (южный полюс)", fontsize=7, color=SUB)
    P = np.array([np.sin(np.radians(50)), np.cos(np.radians(50))])
    ax.annotate("", xy=P, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=RAD, lw=1.2))
    ax.text(P[0] + 0.04, P[1] + 0.04, "нормаль\nк плоскости", fontsize=7, color=RAD)
    ax.plot([0, P[0]], [-1, P[1]], color=CIRC, lw=0.9, ls="--")
    x_proj = P[0] / (1 + P[1])
    ax.plot([x_proj], [0], "o", color=CIRC, ms=4.5)
    ax.text(x_proj, -0.17, "проекция", fontsize=7, color=CIRC, ha="center")
    ax.text(-1.0, 0.06, "плоскость проекции", fontsize=7, color=SUB)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.15, 1.25); ax.set_ylim(-1.25, 1.25); label(ax, "а", 0.08, 1.0)
    ax = axs[1]
    ax.plot(np.cos(t), np.sin(t), color=SUB, lw=0.8)
    fams = {(1, 0, 0): (INK, 5), (1, 1, 0): (CIRC, 4), (1, 1, 1): (RAD, 4)}
    for base, (c, ms) in fams.items():
        seen = set()
        for p in set(itertools.permutations(base)):
            for sgn in itertools.product([1, -1], repeat=3):
                v = tuple(np.multiply(p, sgn))
                if v[2] < 0 or v in seen:
                    continue
                seen.add(v)
                x, y = stereo(v)
                ax.plot(x, y, "o" if base != (1, 0, 0) else "s", color=c, ms=ms)
                if v[2] >= 0 and abs(x) < 1.01 and abs(y) < 1.01 and (v[0] >= 0 and v[1] >= 0):
                    ax.text(x + 0.05, y + 0.04, "".join(str(i) if i >= 0 else r"$\bar{%d}$" % -i for i in v), fontsize=6.5, color=c)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.2, 1.3); ax.set_ylim(-1.2, 1.2); label(ax, "б", 0.08, 1.0)
    ax = axs[2]
    ax.plot(np.cos(t), np.sin(t), color=SUB, lw=0.8)
    groups = [((0, 0, 1), "(0001)", INK), ((1, 0, 0), r"$\{10\bar10\}$", RAD), ((1, 0, 1), r"$\{10\bar11\}$", PLAST), ((1, 0, 7), r"$\{10\bar17\}$", HYDR)]
    for (h, k, l), name, c in groups:
        pts = []
        for rot in range(6):
            hh, kk = [(h, k), (k, -h - k), (-h - k, h)][rot % 3]
            if rot >= 3:
                hh, kk = -hh, -kk
            pts.append(stereo(hkil_normal(hh, kk, l)))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "o", color=c, ms=4.5 if l != 7 else 3.5)
        x, y = pts[0]
        off = {INK: (-0.15, -0.2), HYDR: (0.12, -0.32)}.get(c, (0.06, 0.06))
        ax.text(x + off[0], y + off[1], name, fontsize=7, color=c)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.2, 1.4); ax.set_ylim(-1.2, 1.2); label(ax, "в", 0.08, 1.0)
    fig.tight_layout(w_pad=0.3)
    save(fig, "l5_stereo")


if __name__ == "__main__":
    f_cells(); f_stacking(); f_miller(); f_sites(); f_stereo()
