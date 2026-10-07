"""Оформление рисунков учебника: Manim на белом фоне в духе 3Blue1Brown — Computer Modern (LaTeX, T2A),
крупные чистые формы, цвет несёт смысл: окружные гидриды синие, радиальные красные, наклонные серо-лиловые,
нагрузка — золотая, пластичность — фиолетовая, водород — зелёный."""
import os
import numpy as np
from manim import *

DATA = os.environ.get("TB_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
config.background_color = WHITE

INK = ManimColor("#1d1d1f")
SUB = ManimColor("#5f6368")
DIM = ManimColor("#9aa0a6")
FAINT = ManimColor("#e8eaed")
CIRC = ManimColor("#2a6fdb")        # окружные гидриды
RAD = ManimColor("#d93a3a")         # радиальные
MIDC = ManimColor("#9b8aa6")        # наклонные
LOAD = ManimColor("#d18b00")        # нагрузка
PLAST = ManimColor("#7b4ea3")       # пластичность
HYDR = ManimColor("#1e9e6a")        # водород
METAL = ManimColor("#f3f1ec")       # металл (заливка)
METAL_E = ManimColor("#c9c4b8")
HYD_F = ManimColor("#9ec3f5")       # заливка гидрида

for cls in (Tex, MathTex, Text):
    cls.set_default(color=INK)
for cls in (Line, Arrow, DashedLine, Circle, Rectangle, Square, Polygon, Dot, Arc, Brace, NumberLine, Axes, VMobject):
    try:
        cls.set_default(color=INK)
    except Exception:
        pass

RU = TexTemplate(preamble=r"""
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english,russian]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
""")


def T(s, size=30, color=INK, **kw):
    return Tex(s, tex_template=RU, font_size=size, color=color, **kw)


def T1(s, size=28, color=INK, width=None):
    t = T(r"\mbox{" + s + "}", size, color)
    if width and t.width > width:
        t.scale_to_fit_width(width)
    return t


def M(s, size=40, color=INK, **kw):
    return MathTex(s, tex_template=RU, font_size=size, color=color, **kw)


def caption(s, size=26, color=SUB):
    return T(s, size, color)


def hexrgb(h):
    h = str(h).lstrip("#"); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def cmap(stops):
    pos = np.array([s[0] for s in stops]); cols = np.array([hexrgb(s[1]) for s in stops])

    def f(x):
        x = np.clip(x, 0, 1)
        return np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], -1)
    return f


# расходящаяся: синий (невыгодно, сжатие) — белый — красно-золотой (выгодно)
DIV = cmap([(0.0, "#1f4fa8"), (0.25, "#6f9de0"), (0.5, "#ffffff"), (0.75, "#f2a65a"), (1.0, "#b8401a")])
SEQ = cmap([(0.0, "#ffffff"), (0.35, "#d9c8ea"), (0.7, "#9466c4"), (1.0, "#4b2179")])


def field_image(arr, width, height=None, lim=None, cm=DIV, gamma=0.6, nan_rgb="#ffffff", vmin=None, vmax=None):
    """Поле → ImageMobject (строки сверху вниз)."""
    a = np.asarray(arr, float)
    if cm is DIV:
        lim = lim or np.nanpercentile(np.abs(a), 98)
        v = np.clip(a / lim, -1, 1)
        x = 0.5 + 0.5 * np.sign(v) * np.abs(v) ** gamma
    else:
        vmin = np.nanmin(a) if vmin is None else vmin; vmax = np.nanmax(a) if vmax is None else vmax
        x = np.clip((a - vmin) / (vmax - vmin), 0, 1) ** gamma
    rgb = cm(np.nan_to_num(x, nan=0.5))
    rgb[np.isnan(a)] = hexrgb(nan_rgb)
    img = ImageMobject(np.dstack([rgb.astype(np.uint8), np.full(a.shape, 255, np.uint8)]))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    img.stretch_to_fit_width(width); img.stretch_to_fit_height(height or width * a.shape[0] / a.shape[1])
    return img


def colorbar(cm, lo_txt, hi_txt, mid_txt=None, length=3.0, thick=0.18, label=None, vertical=True):
    n = 64
    x = np.linspace(1, 0, n)[:, None] if vertical else np.linspace(0, 1, n)[None, :]
    rgb = cm(x)
    img = ImageMobject(np.dstack([rgb.astype(np.uint8), np.full(x.shape, 255, np.uint8)]))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    if vertical:
        img.stretch_to_fit_height(length); img.stretch_to_fit_width(thick)
        frame = Rectangle(width=thick, height=length, stroke_width=1.2, stroke_color=SUB).move_to(img)
        hi = T(hi_txt, 22, SUB).next_to(img, RIGHT, buff=0.12).align_to(img, UP)
        lo = T(lo_txt, 22, SUB).next_to(img, RIGHT, buff=0.12).align_to(img, DOWN)
        g = Group(img, frame, hi, lo)
        if mid_txt:
            g.add(T(mid_txt, 22, SUB).next_to(img, RIGHT, buff=0.12))
        if label:
            g.add(T(label, 22, SUB).next_to(img, UP, buff=0.15))
    else:
        img.stretch_to_fit_width(length); img.stretch_to_fit_height(thick)
        frame = Rectangle(width=length, height=thick, stroke_width=1.2, stroke_color=SUB).move_to(img)
        lo = T(lo_txt, 22, SUB).next_to(img, DOWN, buff=0.1).align_to(img, LEFT)
        hi = T(hi_txt, 22, SUB).next_to(img, DOWN, buff=0.1).align_to(img, RIGHT)
        g = Group(img, frame, lo, hi)
        if mid_txt:
            g.add(T(mid_txt, 22, SUB).next_to(img, DOWN, buff=0.1))
        if label:
            g.add(T(label, 22, SUB).next_to(img, UP, buff=0.1))
    return g


def frame_box(mob, buff=0.0, color=METAL_E, width=1.5):
    return SurroundingRectangle(mob, buff=buff, color=color, stroke_width=width, corner_radius=0.0)


def plate(width, height, color=CIRC, fill=HYD_F, angle=0.0, center=ORIGIN, sw=2.5):
    r = Rectangle(width=width, height=height, fill_color=fill, fill_opacity=1, stroke_color=color, stroke_width=sw)
    return r.rotate(angle).move_to(center)


def plate_color(psi):
    dev = np.degrees(np.abs(np.arctan(np.tan(psi))))
    return CIRC if dev <= 40 else (RAD if dev >= 65 else MIDC)


class Mapper:
    """Координаты автомата (строка ND вниз, столбец TD вправо, мкм) → кадр."""

    def __init__(self, size_um, center, width):
        self.H, self.W = size_um
        self.c = np.array(center, float)
        self.s = width / self.W

    def pt(self, row, col):
        return self.c + np.array([(col - self.W / 2) * self.s, (self.H / 2 - row) * self.s, 0.0])

    @property
    def width(self):
        return self.W * self.s

    @property
    def height(self):
        return self.H * self.s


def plates_group(P, mp, width=2.4, kinds=None):
    g = VGroup()
    for i, (cy, cx, psi, half) in enumerate(P):
        if half <= 0:
            continue
        dy, dx = -np.sin(psi) * half, np.cos(psi) * half
        ln = Line(mp.pt(cy - dy, cx - dx), mp.pt(cy + dy, cx + dx), stroke_width=width, color=plate_color(psi))
        ln.set_cap_style(CapStyleType.ROUND)
        g.add(ln)
    return g


def arrows_pair(center, direction, length=0.7, gap=0.2, color=LOAD, sw=6):
    d = np.array(direction, float); d /= np.linalg.norm(d)
    a1 = Arrow(center + d * gap, center + d * (gap + length), buff=0, color=color, stroke_width=sw,
               max_tip_length_to_length_ratio=0.3)
    a2 = Arrow(center - d * gap, center - d * (gap + length), buff=0, color=color, stroke_width=sw,
               max_tip_length_to_length_ratio=0.3)
    return VGroup(a1, a2)


def hoop_arrows(box, n=3, length=0.55, gap=0.12, color=LOAD, label=r"\sigma_\theta"):
    g = VGroup()
    ys = np.linspace(box.get_top()[1], box.get_bottom()[1], n + 2)[1:-1]
    for y in ys:
        g.add(Arrow([box.get_right()[0] + gap, y, 0], [box.get_right()[0] + gap + length, y, 0], buff=0, color=color,
                    stroke_width=5, max_tip_length_to_length_ratio=0.32))
        g.add(Arrow([box.get_left()[0] - gap, y, 0], [box.get_left()[0] - gap - length, y, 0], buff=0, color=color,
                    stroke_width=5, max_tip_length_to_length_ratio=0.32))
    if label:
        g.add(M(label, 34, LOAD).next_to(g[0], UP, buff=0.12))
    return g


def axes_rt(origin, length=0.7, color=SUB):
    a1 = Arrow(origin, origin + np.array([length, 0, 0]), buff=0, color=color, stroke_width=3,
               max_tip_length_to_length_ratio=0.25)
    a2 = Arrow(origin, origin + np.array([0, length, 0]), buff=0, color=color, stroke_width=3,
               max_tip_length_to_length_ratio=0.25)
    return VGroup(a1, a2, M(r"\theta", 26, color).next_to(a1, RIGHT, buff=0.06), M("r", 26, color).next_to(a2, UP, buff=0.06))


def numbered(i, text, size=26, width=5.5, color=INK, accent=CIRC):
    c = Circle(radius=0.2, color=accent, stroke_width=2.5)
    n = T(str(i), 24, accent).move_to(c)
    b = T(text, size, color)
    if b.width > width:
        b.scale_to_fit_width(width)
    b.next_to(c, RIGHT, buff=0.25)
    return VGroup(VGroup(c, n), b)


def load(name):
    return np.load(os.path.join(DATA, name), allow_pickle=True)
