"""Общий стиль ролика — как в hydrides_film_v7: чёрный фон, Computer Modern (LaTeX, T2A),
заголовок слева сверху, серый подзаголовок, тёмные панели с рамкой; радиальные гидриды
красные, окружные (тангенциальные) — голубые."""
import os
import numpy as np
from manim import *

DATA = os.environ.get("FILM_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

RED_H = ManimColor("#ff462d")       # радиальные
CYAN_H = ManimColor("#00afd6")      # окружные
MID_H = ManimColor("#9c8f9a")       # наклонные 40–65°
ORANGE_H = ManimColor("#e87800")
YELLOW_H = ManimColor("#f5c518")
TEAL_H = ManimColor("#1fbf8f")
PANEL = ManimColor("#141613")
BORDER = ManimColor("#3e403d")
SUB = ManimColor("#a8a8a8")
DIM = ManimColor("#6a6a6a")

RU = TexTemplate(preamble=r"""
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english,russian]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
""")


def T(s, size=32, color=WHITE, left=False, **kw):
    if left:
        kw["tex_environment"] = "flushleft"
    return Tex(s, tex_template=RU, font_size=size, color=color, **kw)


def TL(s, size=28, color=WHITE, width=None, **kw):
    """Абзац, выровненный влево; при необходимости ужимается до ширины width."""
    t = T(s, size, color, left=True, **kw)
    if width and t.width > width:
        t.scale_to_fit_width(width)
    return t


def T1(s, size=28, color=WHITE, width=13.0, **kw):
    """Одна строка без переносов LaTeX; при необходимости ужимается до ширины width."""
    t = T(r"\mbox{" + s + "}", size, color, **kw)
    if t.width > width:
        t.scale_to_fit_width(width)
    return t


def M(s, size=40, color=WHITE, **kw):
    return MathTex(s, tex_template=RU, font_size=size, color=color, **kw)


def title_block(title, subtitle=None):
    t = T(title, size=50).to_corner(UL, buff=0.5).shift(0.08 * DOWN)
    g = VGroup(t)
    if subtitle:
        s = T(subtitle, size=31, color=SUB).next_to(t, DOWN, buff=0.22, aligned_edge=LEFT)
        g.add(s)
    return g


def panel(w, h, center=ORIGIN):
    return Rectangle(width=w, height=h, fill_color=PANEL, fill_opacity=1, stroke_color=BORDER,
                     stroke_width=2).move_to(center)


def plate_color(psi):
    dev = np.degrees(np.abs(np.arctan(np.tan(psi))))
    return CYAN_H if dev <= 40 else (RED_H if dev >= 65 else MID_H)


class Mapper:
    """Перевод координат автомата (строка ND вниз, столбец TD вправо, мкм) в координаты кадра."""

    def __init__(self, size_um, center, width):
        self.H, self.W = size_um
        self.c = np.array(center, float)
        self.s = width / self.W                  # единиц кадра на мкм

    def pt(self, row, col):
        return self.c + np.array([(col - self.W / 2) * self.s, (self.H / 2 - row) * self.s, 0.0])

    @property
    def height(self):
        return self.H * self.s

    @property
    def width(self):
        return self.W * self.s


def plates_group(P, mp, width=3.0, scale_len=1.0):
    g = VGroup()
    for cy, cx, psi, half in P:
        dy, dx = -np.sin(psi) * half * scale_len, np.cos(psi) * half * scale_len
        ln = Line(mp.pt(cy - dy, cx - dx), mp.pt(cy + dy, cx + dx), stroke_width=width, color=plate_color(psi))
        ln.set_cap_style(CapStyleType.ROUND)
        g.add(ln)
    return g


def _cmap(stops):
    pos = np.array([s[0] for s in stops]); cols = np.array([s[1] for s in stops], float)

    def f(x):
        x = np.clip(x, 0, 1)
        return np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], -1)
    return f


def hex2rgb(h):
    h = h.lstrip("#"); return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


FAV_CMAP = _cmap([(0.0, hex2rgb("#000000")), (0.45, hex2rgb("#2a2000")), (0.75, hex2rgb("#8a6400")), (1.0, hex2rgb("#ffd24d"))])
DIV_CMAP = _cmap([(0.0, hex2rgb("#00afd6")), (0.3, hex2rgb("#06404f")), (0.5, hex2rgb("#141613")),
                  (0.7, hex2rgb("#6b3a00")), (1.0, hex2rgb("#ffb000"))])


def rgba_image(rgb, alpha=None):
    a = np.full(rgb.shape[:2], 255, np.uint8) if alpha is None else (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return np.dstack([rgb.astype(np.uint8), a])


def fav_image(fav, mp, vmin=-5.0):
    """Карта выгоды зарождения (log10 веса, 0 — максимум) → ImageMobject на месте панели."""
    x = (fav - vmin) / (-vmin)
    if np.ptp(fav) < 1e-3:                 # до первой пластинки выгода везде одинаковая
        x = np.full_like(fav, 0.42)
    img = ImageMobject(rgba_image(FAV_CMAP(x ** 1.4)))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    img.stretch_to_fit_width(mp.width); img.stretch_to_fit_height(mp.height); img.move_to(mp.c)
    return img


def div_image(arr, mp_center, size, lim, gamma=1.0):
    v = np.clip(np.nan_to_num(arr, nan=0.0) / lim, -1, 1)
    x = 0.5 + 0.5 * np.sign(v) * np.abs(v) ** gamma
    rgb = DIV_CMAP(x)
    alpha = np.where(np.isnan(arr), 0.0, 1.0)
    img = ImageMobject(rgba_image(rgb, alpha))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    img.stretch_to_fit_width(size); img.stretch_to_fit_height(size); img.move_to(mp_center)
    return img


def grain_overlay(grains, mp, color="#3e403d", alpha=0.9):
    b = np.zeros(grains.shape, bool)
    b[:-1] |= grains[:-1] != grains[1:]; b[:, :-1] |= grains[:, :-1] != grains[:, 1:]
    rgb = np.zeros(grains.shape + (3,)); rgb[:] = hex2rgb(color)
    img = ImageMobject(rgba_image(rgb, b * alpha))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.stretch_to_fit_width(mp.width); img.stretch_to_fit_height(mp.height); img.move_to(mp.c)
    return img


def sigma_arrows(rect, n=3, length=0.75, gap=0.15, color=WHITE, label=True):
    g = VGroup()
    ys = np.linspace(rect.get_top()[1], rect.get_bottom()[1], n + 2)[1:-1]
    for y in ys:
        g.add(Arrow(start=[rect.get_right()[0] + gap, y, 0], end=[rect.get_right()[0] + gap + length, y, 0], buff=0,
                    color=color, stroke_width=6, max_tip_length_to_length_ratio=0.35))
        g.add(Arrow(start=[rect.get_left()[0] - gap, y, 0], end=[rect.get_left()[0] - gap - length, y, 0], buff=0,
                    color=color, stroke_width=6, max_tip_length_to_length_ratio=0.35))
    if label:
        g.add(M(r"\sigma", 40).next_to(g[0], UP, buff=0.15))
    return g


def r_axis(rect):
    a = Arrow(rect.get_corner(UL) + np.array([0.3, -0.85, 0]), rect.get_corner(UL) + np.array([0.3, -0.2, 0]),
              buff=0, color=GREY_B, stroke_width=5, max_tip_length_to_length_ratio=0.3)
    return VGroup(a, M("r", 30, color=GREY_B).next_to(a, RIGHT, buff=0.1).shift(0.15 * UP))


def legend_row():
    items = VGroup()
    for col, txt in ((CYAN_H, r"окружные ($\le 40^\circ$ к дуге)"), (MID_H, r"наклонные"), (RED_H, r"радиальные ($\ge 65^\circ$)")):
        ln = Line(ORIGIN, 0.45 * RIGHT, color=col, stroke_width=7); ln.set_cap_style(CapStyleType.ROUND)
        items.add(VGroup(ln, T(txt, 24, color=SUB).next_to(ln, RIGHT, buff=0.15)))
    return items.arrange(RIGHT, buff=0.6)


def rhf_of(P):
    if len(P) == 0:
        return 0.0
    dev = np.degrees(np.abs(np.arctan(np.tan(P[:, 2]))))
    w = np.where(dev <= 40, 0, np.where(dev < 65, 0.5, 1.0))
    L = 2 * P[:, 3]
    return float((L * w).sum() / L.sum())


def numbered(i, text, width=10.5, color=WHITE, size=28):
    c = Circle(radius=0.22, color=YELLOW_H, stroke_width=3)
    n = T(str(i), 26, color=YELLOW_H).move_to(c)
    body = T(text, size, color=color, left=True)
    if body.width > width:
        body.scale_to_fit_width(width)
    body.next_to(c, RIGHT, buff=0.35)
    c.align_to(body, UP).shift(0.02 * DOWN); n.move_to(c)
    return VGroup(VGroup(c, n), body)
