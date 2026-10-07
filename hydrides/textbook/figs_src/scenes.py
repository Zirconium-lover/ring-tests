"""Рисунки учебника (Manim CE, белый фон). Каждая сцена — один неподвижный кадр:
manim -s -qh scenes.py <Сцена>   (render.sh собирает все в ../figs)."""
import json
import numpy as np
from manim import *
from style import *

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "automaton"))
import thermo  # noqa: E402


# ============================================================ глава 1. задача
class F01_Tube(Scene):
    """Труба, сечение r–θ, окружные и радиальные гидриды."""
    def construct(self):
        c = np.array([-4.2, -0.2, 0])
        outer, inner = Circle(radius=2.6, color=INK, stroke_width=3), Circle(radius=2.05, color=INK, stroke_width=3)
        ring = Difference(Circle(radius=2.6), Circle(radius=2.05), fill_color=METAL, fill_opacity=1, stroke_width=0)
        tube = VGroup(ring, outer, inner).move_to(c)
        rng = np.random.default_rng(3)
        hyd = VGroup()
        for _ in range(70):
            a = rng.uniform(0, 2 * np.pi); r = rng.uniform(2.12, 2.53); L = rng.uniform(0.12, 0.3)
            tdir = np.array([-np.sin(a), np.cos(a), 0])
            p = c + r * np.array([np.cos(a), np.sin(a), 0])
            hyd.add(Line(p - tdir * L / 2, p + tdir * L / 2, color=CIRC, stroke_width=2.2))
        lab = T(r"сечение стенки $r$--$\theta$", 26, SUB).next_to(tube, DOWN, buff=0.25)
        # окно увеличения
        a0 = np.radians(20); pw = c + 2.33 * np.array([np.cos(a0), np.sin(a0), 0])
        win = Square(0.42, color=LOAD, stroke_width=3).move_to(pw)
        # две увеличенные стенки
        boxes = []
        for k, (cx, txt, radial) in enumerate(((1.3, r"без нагрузки: \textcolor[HTML]{2A6FDB}{окружные}", False),
                                               (4.9, r"под $\sigma_\theta$: \textcolor[HTML]{D93A3A}{радиальные}", True))):
            bx = Rectangle(width=3.2, height=3.2, fill_color=METAL, fill_opacity=1, stroke_color=METAL_E,
                           stroke_width=2).move_to([cx, 0.3, 0])
            g = VGroup()
            rr = np.random.default_rng(10 + k)
            for _ in range(26):
                p = np.array([cx + rr.uniform(-1.4, 1.4), 0.3 + rr.uniform(-1.4, 1.4), 0])
                L = rr.uniform(0.25, 0.6)
                ang = rr.normal(0, 0.25) + (np.pi / 2 if radial and rr.random() < 0.8 else 0)
                d = np.array([np.cos(ang), np.sin(ang), 0])
                g.add(Line(p - d * L / 2, p + d * L / 2, color=RAD if abs(np.sin(ang)) > 0.7 else CIRC, stroke_width=3))
            boxes.append(VGroup(bx, g, T(txt, 25).next_to(bx, UP, buff=0.18)))
        b0, b1 = boxes
        ar = hoop_arrows(b1[0], n=3, length=0.45)
        ax = axes_rt(b0[0].get_corner(DL) + np.array([-0.85, 0.05, 0]), 0.55)
        link = DashedLine(win.get_right(), b0[0].get_left(), color=LOAD, stroke_width=2, dash_length=0.08)
        foot = T(r"трещина идёт вдоль гидридов: радиальные пересекают стенку --- опаснее", 26, SUB).to_edge(DOWN, buff=0.35)
        self.add(tube, hyd, lab, win, link, *boxes, ar, ax, foot)


class F02_Map(Scene):
    """Карта модели: что на входе, какие блоки, что на выходе."""
    def construct(self):
        def box(txt, sub, col, w=3.1, h=1.35, ch=None):
            r = RoundedRectangle(width=w, height=h, corner_radius=0.15, stroke_color=col, stroke_width=3,
                                 fill_color=WHITE, fill_opacity=1)
            t = T(txt, 27, col)
            s_ = T(sub, 20, SUB)
            if s_.width > w - 0.25:
                s_.scale_to_fit_width(w - 0.25)
            grp = VGroup(t, s_).arrange(DOWN, buff=0.1)
            if ch:
                grp.add(T(ch, 18, DIM)); grp.arrange(DOWN, buff=0.08)
            grp.move_to(r)
            return VGroup(r, grp)
        inp = box(r"Условия опыта", r"сплав, текстура, зерно\\водород, $T_{\max}$, $\dot T$, $\sigma$", INK, 2.7, 2.0).move_to([-5.75, 0.3, 0])
        th = box(r"Растворимость", r"TSSD, TSSP, пересыщение", HYDR, 2.9, ch=r"гл. 2").move_to([-2.35, 1.45, 0])
        me = box(r"Механика", r"несоответствие, поле, пластичность", PLAST, 2.9, ch=r"гл. 3--4").move_to([1.35, 1.45, 0])
        nu = box(r"Зарождение", r"теория зарождения, границы", RAD, 2.9, ch=r"гл. 5").move_to([-2.35, -0.95, 0])
        gr = box(r"Рост и диффузия", r"водород, кончики пластинок", HYDR, 2.9, ch=r"гл. 6").move_to([1.35, -0.95, 0])
        frame = RoundedRectangle(width=7.5, height=5.0, corner_radius=0.3, stroke_color=CIRC, stroke_width=2.5,
                                 fill_color=ManimColor("#f4f8fe"), fill_opacity=1).move_to([-0.5, 0.2, 0])
        ft = T(r"клеточный автомат: зёрна, шаг по времени (гл. 7)", 22, CIRC).next_to(frame, UP, buff=0.1)
        st = box(r"Структура", r"пластинки, пакеты, стопки", CIRC, 2.8).move_to([5.7, 1.45, 0])
        out = box(r"Меры", r"RHF, межзёренные, связность", INK, 2.8, ch=r"гл. 8").move_to([5.7, -0.85, 0])
        exp = box(r"Опыт", r"шлифы, EBSD, пороги", SUB, 2.8, ch=r"гл. 9--10").move_to([5.7, -3.0, 0])
        A = lambda a, b, col=SUB: Arrow(a, b, buff=0.1, color=col, stroke_width=4, max_tip_length_to_length_ratio=0.25,
                                        max_stroke_width_to_length_ratio=8)
        arrows = VGroup(
            A(inp[0].get_right() + 0.5 * UP, th[0].get_left()), A(inp[0].get_right() + 0.5 * DOWN, nu[0].get_left()),
            A(th[0].get_bottom(), nu[0].get_top(), HYDR), A(me[0].get_corner(DL) + [0.4, 0, 0], nu[0].get_corner(UR) + [-0.4, 0, 0], PLAST),
            A(nu[0].get_right(), gr[0].get_left(), RAD),
            A(gr[0].get_top() + 0.5 * RIGHT, me[0].get_bottom() + 0.5 * RIGHT, PLAST),
            A([frame.get_right()[0], st.get_center()[1], 0], st[0].get_left(), CIRC), A(st[0].get_bottom(), out[0].get_top()),
            DoubleArrow(out[0].get_bottom(), exp[0].get_top(), buff=0.1, color=LOAD, stroke_width=4,
                        max_tip_length_to_length_ratio=0.25))
        n1 = T(r"новые пластинки\\меняют поле", 19, PLAST).next_to(arrows[5], LEFT, buff=0.1)
        n2 = T(r"сравнение", 20, LOAD).next_to(arrows[8], RIGHT, buff=0.1)
        tt = T(r"Модель целиком", 32).to_edge(UP, buff=0.25)
        self.add(tt, frame, ft, inp, th, me, nu, gr, st, out, exp, arrows, n1, n2)


# ============================================================ глава 2. растворимость
class F03_Solubility(Scene):
    def construct(self):
        L = thermo.TSS["E635"]
        ax = Axes(x_range=[150, 450, 50], y_range=[0, 300, 50], x_length=8.0, y_length=5.6, tips=False,
                  axis_config=dict(color=INK, stroke_width=2, include_numbers=True, font_size=24,
                                   decimal_number_config=dict(color=INK, num_decimal_places=0))).shift(2.2 * LEFT + 0.1 * DOWN)
        xl = T(r"температура, $^\circ$C", 24, SUB).next_to(ax.x_axis, DOWN, buff=0.45)
        yl = T(r"водород в растворе, ppm", 24, SUB).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.55)
        Tp300 = float(thermo.T_line(300, L["TSSP"]))
        fd = lambda t: float(thermo.c_line(t, L["TSSD"])); fp = lambda t: float(min(300, thermo.c_line(t, L["TSSP"])))
        d = ax.plot(fd, x_range=[150, 450], color=HYDR, stroke_width=5)
        p = ax.plot(fp, x_range=[150, Tp300], color=RAD, stroke_width=5)
        band = ax.get_area(ax.plot(fp, x_range=[150, 450], stroke_width=0), bounded_graph=ax.plot(fd, x_range=[150, 450], stroke_width=0),
                           color=FAINT, opacity=1)
        H, Tmax = 178.0, 415.0
        Td = float(thermo.T_line(H, L["TSSD"])); Tp = float(thermo.T_line(H, L["TSSP"]))
        heat = VGroup(ax.plot(fd, x_range=[200, Td], color=INK, stroke_width=3), Line(ax.c2p(Td, H), ax.c2p(Tmax, H), color=INK, stroke_width=3))
        cool = VGroup(DashedLine(ax.c2p(Tmax, H + 5), ax.c2p(Tp, H + 5), color=CIRC, stroke_width=4, dash_length=0.1),
                      ax.plot(fp, x_range=[160, Tp], color=CIRC, stroke_width=4))
        dots = VGroup(Dot(ax.c2p(Td, H), color=HYDR, radius=0.07), Dot(ax.c2p(Tp, H), color=RAD, radius=0.07))
        t1 = T(r"TSSD:\\растворение", 26, HYDR).move_to(ax.c2p(432, 135))
        t2 = T(r"TSSP:\\выпадение", 26, RAD).move_to(ax.c2p(290, 230))
        Tx = 300.0
        cd, cp = fd(Tx), fp(Tx)
        br = DoubleArrow(ax.c2p(Tx, cd), ax.c2p(Tx, cp), buff=0, color=LOAD, stroke_width=3, max_tip_length_to_length_ratio=0.15)
        bt = T(r"гистерезис", 22, LOAD).next_to(br, RIGHT, buff=0.1)
        hl = T(r"$H = 178$ ppm", 22, SUB).move_to(ax.c2p(366, 150))
        tm = T(r"$T_{\max}$", 22, SUB).next_to(ax.c2p(Tmax, H), RIGHT, buff=0.08)
        legend = VGroup(
            VGroup(Line(ORIGIN, 0.5 * RIGHT, color=INK, stroke_width=3), T(r"нагрев: гидриды растворяются по TSSD", 22, SUB)).arrange(RIGHT, buff=0.15),
            VGroup(DashedLine(ORIGIN, 0.5 * RIGHT, color=CIRC, stroke_width=4), T(r"охлаждение: раствор пересыщается", 22, SUB)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, 0.5 * RIGHT, color=CIRC, stroke_width=4), T(r"выпадение идёт вдоль TSSP", 22, SUB)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.6, 2.5, 0])
        f = M(r"\lg C = b - k\,\frac{1000}{T}", 40).next_to(legend, DOWN, buff=0.5)
        tab = VGroup(T(r"Э635 (Плясов и др. 2023):", 22, SUB), T(r"TSSD: $b=5.56,\ k=2.24$", 22, HYDR),
                     T(r"TSSP: $b=5.08,\ k=1.68$", 22, RAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(f, DOWN, buff=0.35)
        q = VGroup(T(r"наклон $\Rightarrow$ теплота:", 22, SUB), M(r"Q = \ln 10\cdot 1000\,k\,R", 30),
                   T(r"TSSP Э635: $Q\approx 32$ кДж/моль", 22, SUB)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(tab, DOWN, buff=0.4).align_to(tab, LEFT)
        self.add(band, ax, xl, yl, d, p, heat, cool, dots, t1, t2, br, bt, hl, tm, legend, f, tab, q)


class F04_Scales(Scene):
    """Два языка одной величины: градусы и мегапаскали."""
    def construct(self):
        top = NumberLine(x_range=[0, 14, 2], length=11, color=INK, include_numbers=True, font_size=26,
                         decimal_number_config=dict(color=INK, num_decimal_places=0)).shift(1.3 * UP)
        bot = NumberLine(x_range=[0, 14 * 5.4, 10.8], length=11, color=INK, include_numbers=True, font_size=26,
                         decimal_number_config=dict(color=INK, num_decimal_places=1)).shift(1.0 * DOWN)
        tl = T(r"сдвиг температуры выпадения, $^\circ$C", 26, SUB).next_to(top, UP, buff=0.35)
        bl = T(r"добавка к движущей силе $\Delta$, МПа", 26, SUB).next_to(bot, DOWN, buff=0.55)
        links = VGroup(*[DashedLine(top.n2p(v), bot.n2p(v * 5.4), color=DIM, stroke_width=1.5, dash_length=0.07)
                         for v in range(0, 15, 2)])
        conv = M(r"\frac{\partial \Delta}{\partial T}=n_H\,\frac{Q}{T}\approx 1.01\cdot10^5\cdot\frac{32.2\cdot10^3}{603}\ \text{Па/К}"
                 r"\approx 5.4\ \text{МПа}/^\circ\text{C}", 34).move_to(0.15 * DOWN)
        bg = BackgroundRectangle(conv, color=WHITE, fill_opacity=1, buff=0.12)
        marks = VGroup()
        for v, txt, col in ((1.5, r"граница", RAD), (5.0, r"фора зёрен Zr-Nb", CIRC), (12.4, r"фора Zry-4", CIRC)):
            d = Dot(top.n2p(v), color=col, radius=0.08)
            marks.add(d, T(txt, 20, col).next_to(d, UP, buff=0.12))
        self.add(links, top, bot, tl, bl, bg, conv, marks)


# ============================================================ глава 3. несоответствие
class F05_Eshelby(Scene):
    """Мысленный опыт Эшелби в четыре шага."""
    def construct(self):
        xs = [-5.2, -1.75, 1.75, 5.2]
        titles = [r"1. вырезать", r"2. превратить\\(свободно)", r"3. сжать\\до старой формы", r"4. вставить\\и отпустить"]
        g = VGroup()
        for k, x in enumerate(xs):
            c = np.array([x, -0.2, 0])
            blk = Square(2.6, fill_color=METAL, fill_opacity=1, stroke_color=METAL_E, stroke_width=2).move_to(c)
            hole = Rectangle(width=1.4, height=0.36, stroke_color=INK, stroke_width=2, fill_color=WHITE, fill_opacity=1).move_to(c)
            t = T(titles[k], 26).next_to(blk, UP, buff=0.25)
            if k == 0:
                piece = Rectangle(width=1.4, height=0.36, fill_color=METAL, fill_opacity=1, stroke_color=INK,
                                  stroke_width=2).move_to(c + 1.75 * DOWN)
                arr = Arrow(c, c + 1.45 * DOWN, buff=0.25, color=SUB, stroke_width=3)
                g.add(blk, hole, piece, arr, t)
            elif k == 1:
                old = DashedVMobject(Rectangle(width=1.4, height=0.36, stroke_color=SUB), num_dashes=24).move_to(c + 1.75 * DOWN)
                new = Rectangle(width=1.4 * 1.15, height=0.36 * 1.9, fill_color=HYD_F, fill_opacity=1, stroke_color=CIRC,
                                stroke_width=2.5).move_to(c + 1.75 * DOWN)
                e = M(r"\varepsilon^*", 32, CIRC).next_to(new, RIGHT, buff=0.15)
                g.add(blk, hole, new, old, e, t)
            elif k == 2:
                new = Rectangle(width=1.4, height=0.36, fill_color=HYD_F, fill_opacity=1, stroke_color=CIRC, stroke_width=2.5).move_to(c + 1.75 * DOWN)
                ar = VGroup(Arrow(new.get_top() + 0.55 * UP, new.get_top(), buff=0.03, color=LOAD, stroke_width=4),
                            Arrow(new.get_bottom() + 0.55 * DOWN, new.get_bottom(), buff=0.03, color=LOAD, stroke_width=4),
                            Arrow(new.get_left() + 0.5 * LEFT, new.get_left(), buff=0.03, color=LOAD, stroke_width=4),
                            Arrow(new.get_right() + 0.5 * RIGHT, new.get_right(), buff=0.03, color=LOAD, stroke_width=4))
                lab = M(r"-C:\varepsilon^*", 30, LOAD).next_to(new, DOWN, buff=0.6)
                g.add(blk, hole, new, ar, lab, t)
            else:
                inc = Rectangle(width=1.4 * 1.06, height=0.36 * 1.35, fill_color=HYD_F, fill_opacity=1, stroke_color=CIRC,
                                stroke_width=2.5).move_to(c)
                halo = VGroup(*[Ellipse(width=1.4 * (1.06 + 0.45 * i), height=0.36 * (1.35 + 1.4 * i), stroke_color=PLAST,
                                        stroke_width=2.2 - 0.4 * i, stroke_opacity=0.9 - 0.2 * i).move_to(c) for i in range(1, 4)])
                lab = T(r"матрица напряжена,\\включение сжато", 22, SUB).next_to(blk, DOWN, buff=0.3)
                g.add(blk, halo, inc, lab, t)
        arrs = VGroup(*[Arrow([xs[i] + 1.35, 1.6, 0], [xs[i + 1] - 1.35, 1.6, 0], buff=0, color=DIM, stroke_width=2.5,
                              max_tip_length_to_length_ratio=0.25) for i in range(3)])
        foot = M(r"\sigma=C:\left(\varepsilon-\varepsilon^*\right),\qquad \nabla\cdot\sigma=0", 36).to_edge(DOWN, buff=0.3)
        self.add(g, foot)


class F06_Work(Scene):
    """Работа нагрузки над несоответствием: радиальная и окружная пластинка."""
    def construct(self):
        res = Group()
        for k, (x, radial) in enumerate(((-3.6, False), (3.0, True))):
            c = np.array([x, 0.35, 0])
            box = Square(3.6, fill_color=METAL, fill_opacity=1, stroke_color=METAL_E, stroke_width=2).move_to(c)
            pl = plate(0.32, 2.0, RAD, ManimColor("#f6b0aa"), 0, c) if radial else plate(2.0, 0.32, CIRC, HYD_F, 0, c)
            # несоответствие: по нормали 7.2 %, вдоль 4.6 %
            if radial:
                en = VGroup(Arrow(c + 0.18 * RIGHT, c + 0.95 * RIGHT, buff=0, color=RAD, stroke_width=6),
                            Arrow(c + 0.18 * LEFT, c + 0.95 * LEFT, buff=0, color=RAD, stroke_width=6))
                et = VGroup(Arrow(c + 1.02 * UP, c + 1.45 * UP, buff=0, color=RAD, stroke_width=3.5),
                            Arrow(c + 1.02 * DOWN, c + 1.45 * DOWN, buff=0, color=RAD, stroke_width=3.5))
            else:
                en = VGroup(Arrow(c + 0.18 * UP, c + 0.95 * UP, buff=0, color=CIRC, stroke_width=6),
                            Arrow(c + 0.18 * DOWN, c + 0.95 * DOWN, buff=0, color=CIRC, stroke_width=6))
                et = VGroup(Arrow(c + 1.02 * RIGHT, c + 1.45 * RIGHT, buff=0, color=CIRC, stroke_width=3.5),
                            Arrow(c + 1.02 * LEFT, c + 1.45 * LEFT, buff=0, color=CIRC, stroke_width=3.5))
            ha = hoop_arrows(box, n=3, length=0.45, label=None)
            if radial:
                f = M(r"g=\frac{\sigma_\theta\,\varepsilon_n}{\varepsilon_n}=\sigma_\theta", 34)
            else:
                f = M(r"g=\frac{\sigma_\theta\,\varepsilon_t}{\varepsilon_n}=0.64\,\sigma_\theta", 34)
            f.next_to(box, DOWN, buff=0.35)
            ttl = T(r"радиальная: нагрузка тянет\\вдоль большого $\varepsilon_n=7.2\%$" if radial else
                    r"окружная: нагрузка тянет\\вдоль малого $\varepsilon_t=4.6\%$", 26).next_to(box, UP, buff=0.3)
            res.add(box, pl, en, et, ha, f, ttl)
        diff = M(r"g_{\text{рад}}-g_{\text{окр}}=\frac{\varepsilon_n-\varepsilon_t}{\varepsilon_n}\,\sigma_\theta=0.36\,\sigma_\theta"
                 r"\quad\text{(упругость)}", 34, LOAD).to_edge(DOWN, buff=0.25)
        self.add(res, diff, M(r"\sigma_\theta", 32, LOAD).move_to([-0.3, 2.6, 0]))


class F07_Field(Scene):
    """Упругое поле одной пластинки: где выгодно зарождаться следующей."""
    def construct(self):
        z = load("plate_elastic.npz")
        g = z["g_par"][::-1]          # строки сверху вниз = n по убыванию
        fr = z["fr"][::-1]
        n = g.shape[0]; dx = float(z["dx"]); Lw = float(z["L"])
        crop = slice(int(n * 0.2), int(n * 0.8))
        gc = np.where(fr[crop, crop] > 0.5, np.nan, g[crop, crop])
        W = 5.6
        img = field_image(gc, W, lim=600, gamma=0.55, nan_rgb="#9ec3f5").move_to([-2.6, -0.1, 0])
        fb = frame_box(img)
        scale = W / (Lw * 0.6)
        bar = Line(img.get_corner(DL) + np.array([0.25, 0.3, 0]), img.get_corner(DL) + np.array([0.25 + 2 * scale, 0.3, 0]),
                   color=INK, stroke_width=4)
        bt = T(r"2 мкм", 20).next_to(bar, UP, buff=0.06)
        cb = colorbar(DIV, r"$-600$", r"$+600$", r"0", length=3.6, label=r"$g$, МПа").next_to(img, RIGHT, buff=0.4)
        tip = img.get_center() + np.array([2.5 * scale, 0, 0])
        a1 = Arrow(tip + np.array([0.6, -1.5, 0]), tip + np.array([0.25, -0.12, 0]), buff=0, color=INK, stroke_width=3)
        t1 = T(r"перед кончиком:\\выгодно продолжить", 22).next_to(a1.get_start(), DOWN, buff=0.08)
        side = img.get_center() + np.array([0, 0.45, 0])
        a2 = Arrow(side + np.array([-1.2, 1.4, 0]), side, buff=0, color=INK, stroke_width=3)
        t2 = T(r"над гранью: невыгодно", 22).next_to(a2.get_start(), UP, buff=0.05)
        right = VGroup(
            M(r"g(\mathbf x)=\frac{\sigma(\mathbf x):\varepsilon^*_{\text{нов}}}{\varepsilon_n}", 38),
            T(r"--- выгода зародыша той же ориентации\\в точке $\mathbf x$ от поля пластинки", 24, SUB),
            T(r"красное --- поле помогает,\\синее --- мешает", 24, SUB),
            T(r"упругость, плоская деформация\\со свободным удлинением по оси", 22, DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([4.6, 0.2, 0])
        self.add(img, fb, bar, bt, cb, a1, t1, a2, t2, right)


class F08_Fourier(Scene):
    """Как поле считается через Фурье."""
    def construct(self):
        z = load("plate_elastic.npz")
        fr = z["fr"][::-1]; g = z["g_par"][::-1]
        n = fr.shape[0]; c = slice(int(n * 0.25), int(n * 0.75))
        W = 2.4
        i1 = field_image(fr[c, c], W, cm=SEQ, gamma=1.0, vmin=0, vmax=1).move_to([-5.2, 0.6, 0])
        # «спектр»: модуль Фурье-образа доли гидрида
        F = np.abs(np.fft.fftshift(np.fft.fft2(fr)))
        cc = slice(int(n * 0.4), int(n * 0.6))
        F = np.log10(F + 1e-3)[cc, cc]
        i2 = field_image(F, W, cm=SEQ, gamma=1.2, vmin=F.min() + 2, vmax=F.max()).move_to([-1.75, 0.6, 0])
        i4 = field_image(np.where(fr[c, c] > 0.5, np.nan, g[c, c]), W, lim=600, gamma=0.55, nan_rgb="#9ec3f5").move_to([5.2, 0.6, 0])
        mid = VGroup(M(r"\hat u=-i\,K^{-1}(C:\hat\varepsilon^*)\,\mathbf k", 28),
                     M(r"K_{ij}=C_{ikjl}\,k_k k_l", 28)).arrange(DOWN, buff=0.25).move_to([1.75, 0.6, 0])
        heads = VGroup(T(r"$\varepsilon^*(\mathbf x)$: где гидрид", 24).next_to(i1, UP, buff=0.2),
                       T(r"$\hat\varepsilon^*(\mathbf k)$: БПФ", 24).next_to(i2, UP, buff=0.2),
                       T(r"равновесие --- алгебра\\для каждой $\mathbf k$", 24).next_to(mid, UP, buff=0.35),
                       T(r"$\sigma(\mathbf x)$: обратное БПФ", 24).next_to(i4, UP, buff=0.2))
        A = lambda a, b: Arrow(a.get_right(), b.get_left(), buff=0.15, color=SUB, stroke_width=3, max_tip_length_to_length_ratio=0.2)
        arr = VGroup(A(i1, i2), A(i2, mid), A(mid, i4))
        foot = VGroup(T(r"Почему это удобно: производная $\partial/\partial x \;\to\; i k_x$, а $\nabla\cdot\sigma=0$ из"
                        r" дифференциального уравнения превращается в систему $2\times2$ на каждой частоте.", 24, SUB),
                      T(r"Плата --- периодичность: ячейка повторяется во все стороны, как обои.", 24, SUB)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(Group(i1, i4), DOWN, buff=0.6)
        foot.scale_to_fit_width(min(foot.width, 12.5)).set_x(0)
        self.add(i1, i2, i4, frame_box(i1), frame_box(i2), frame_box(i4), mid, heads, arr, foot)


# ============================================================ глава 4. пластичность
class F09_Yield(Scene):
    """Несоответствие против деформации текучести."""
    def construct(self):
        ax = Axes(x_range=[0, 8, 1], y_range=[0, 4, 1], x_length=9.5, y_length=4.2, tips=False,
                  axis_config=dict(color=INK, stroke_width=2),
                  x_axis_config=dict(include_numbers=True, font_size=24, decimal_number_config=dict(color=INK, num_decimal_places=0)),
                  y_axis_config=dict(include_ticks=False)).shift(0.5 * UP)
        xl = T(r"деформация, \%", 24, SUB).next_to(ax.x_axis, DOWN, buff=0.4)
        rows = [(3, 7.2, RAD, r"$\varepsilon_n$ --- несоответствие по нормали"),
                (2, 4.58, CIRC, r"$\varepsilon_t$ --- несоответствие вдоль пластинки"),
                (1, 0.39, PLAST, r"$\sigma_y/E = 350/90\,000$ --- предел упругости")]
        g = VGroup()
        for y, v, col, txt in rows:
            bar = Rectangle(width=ax.c2p(v, 0)[0] - ax.c2p(0, 0)[0], height=0.55, fill_color=col, fill_opacity=0.85,
                            stroke_width=0).move_to(ax.c2p(v / 2, y))
            lab = T(txt, 24, col).next_to(ax.c2p(0, y + 0.3), RIGHT, buff=0.1).shift(0.18 * UP)
            val = T(f"{v:g}\\,\\%", 24, col).next_to(bar, RIGHT, buff=0.12)
            g.add(bar, lab, val)
        bt = T(r"несоответствие в 12--18 раз больше предела упругости:\\металлу вокруг пластинки приходится течь", 24, SUB).next_to(xl, DOWN, buff=0.3)
        self.add(ax, xl, g, bt)


class F10_Halo(Scene):
    """Пластический ореол: зона течения и как он меняет поле."""
    def construct(self):
        h = load("halo_a2.5_al0_0.npz"); e = load("plate_elastic.npz")
        # пластика: массивы [t, n] шаг 0.1 в ячейке 30 мкм; окно t ±7, n ±4
        n = h["p"].shape[0]; x = (np.arange(n) + 0.5) * 0.1 - 15
        it = (np.abs(x) < 7); jn = (np.abs(x) < 4)
        p = h["p"][np.ix_(it, jn)].T[::-1]; gp = h["g"][np.ix_(it, jn)].T[::-1]; mk = h["mask"][np.ix_(it, jn)].T[::-1]
        ne = e["fr"].shape[0]; xe = (np.arange(ne) + 0.5) * float(e["dx"]) - float(e["L"]) / 2
        ite = np.abs(xe) < 7; jne = np.abs(xe) < 4
        ge = e["g_par"][np.ix_(jne, ite)][::-1]; fe_ = e["fr"][np.ix_(jne, ite)][::-1]
        W = 4.1
        i0 = field_image(np.where(mk, np.nan, p), W, cm=SEQ, gamma=0.45, vmin=0, vmax=0.05, nan_rgb="#9ec3f5")
        i1 = field_image(np.where(fe_ > 0.5, np.nan, ge), W, lim=600, gamma=0.55, nan_rgb="#9ec3f5")
        i2 = field_image(np.where(mk, np.nan, gp), W, lim=600, gamma=0.55, nan_rgb="#9ec3f5")
        Group(i0, i1, i2).arrange(RIGHT, buff=0.45).shift(0.65 * UP)
        heads = VGroup(T(r"зона течения $p$ (Мизес, $\sigma_y{=}350$ МПа)", 22, PLAST).next_to(i0, UP, buff=0.15),
                       T(r"$g$ соседа: упругость", 22).next_to(i1, UP, buff=0.15),
                       T(r"$g$ соседа: с пластичностью", 22).next_to(i2, UP, buff=0.15))
        cb = colorbar(DIV, r"$-600$", r"$+600$", r"0", length=3.6, thick=0.16, vertical=False, label=r"$g$, МПа").next_to(Group(i1, i2), DOWN, buff=0.35)
        cb0 = colorbar(SEQ, r"0", r"0.05", None, length=2.6, thick=0.16, vertical=False, label=r"$p$").next_to(i0, DOWN, buff=0.35)
        eq = VGroup(M(r"\sigma=C:\big(\varepsilon-\varepsilon^*-\textcolor[HTML]{7B4EA3}{\varepsilon_p}\big)", 34),
                    T(r"$\Rightarrow$ то же поле даёт упругий расчёт с собственной деформацией $\varepsilon^*+\varepsilon_p$", 24, SUB)
                    ).arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.25)
        self.add(i0, i1, i2, frame_box(i0), frame_box(i1), frame_box(i2), heads, cb, cb0, eq)


class F11_Superpose(Scene):
    """Сложение ореолов против честного расчёта трёх пластинок и против потолка."""
    def construct(self):
        z = load("superpose.npz")
        n = z["deck_ref"].shape[0]; x = (np.arange(n) + 0.5) * 30.0 / n - 15
        it = (x > -12) & (x < 8.5); jn = (x > -3.2) & (x < 2.6)
        W = 3.85
        rows = Group()
        for conf, lab in (("deck", r"колода"), ("chain", r"цепочка")):
            hyd = z[f"{conf}_hyd"][np.ix_(it, jn)].T[::-1]
            ims = Group()
            for key in ("ref", "halo", "cap"):
                a = z[f"{conf}_{key}"][np.ix_(it, jn)].T[::-1]
                ims.add(field_image(np.where(hyd, np.nan, a), W, lim=700, gamma=0.55, nan_rgb="#cfd8e6"))
            ims.arrange(RIGHT, buff=0.3)
            # места следующей пластинки
            for im in ims:
                sx = W / (x[it][-1] - x[it][0]); sy = sx
                for (st, sn), col in (((2.5, 1.2), LOAD), ((5.8, 0.6), HYDR)):
                    cx = im.get_left()[0] + (st - x[it][0]) * sx; cy = im.get_bottom()[1] + (sn - x[jn][0]) * sy
                    ims.add(DashedVMobject(Rectangle(width=5 * sx, height=0.6 * sy, stroke_color=col, stroke_width=2.5).move_to([cx, cy, 0]), num_dashes=22))
            ims.add(T(lab, 26).next_to(ims[0], LEFT, buff=0.25))
            rows.add(ims)
        rows.arrange(DOWN, buff=0.45).move_to([0.3, 0.6, 0])
        heads = VGroup(*[T(t, 23).next_to(rows[0][i], UP, buff=0.15) for i, t in
                         enumerate((r"честный расчёт трёх пластинок", r"сумма ореолов одиночной", r"автомат раньше: потолок $\pm180$"))])
        leg = VGroup(VGroup(DashedLine(ORIGIN, 0.5 * RIGHT, color=LOAD, stroke_width=3), T(r"место следующей в колоде", 21, SUB)).arrange(RIGHT, buff=0.12),
                     VGroup(DashedLine(ORIGIN, 0.5 * RIGHT, color=HYDR, stroke_width=3), T(r"место следующей в цепочке", 21, SUB)).arrange(RIGHT, buff=0.12),
                     T(r"радиальные пластинки, $\sigma_\theta=110$ МПа; $g$ соседей без вклада нагрузки", 21, SUB)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(rows, DOWN, buff=0.35)
        self.add(rows, heads, leg)


class F12_FEM(Scene):
    """Сверка Фурье с МКЭ: производные величины."""
    def construct(self):
        d = json.load(open(os.path.join(DATA, "fft_fem.json")))
        items = [(r"одиночная, рад.\,$-$\,окр.", d["derived"]["рад − окр single U110"]),
                 (r"колода, рад.\,$-$\,окр.", d["derived"]["рад − окр deck U110"]),
                 (r"колода окр.: прибавка", d["derived"]["Δg колл. deck circ U110"]),
                 (r"колода рад.: прибавка", d["derived"]["Δg колл. deck rad U110"]),
                 (r"цепочка рад.: прибавка", d["derived"]["Δg колл. chain rad U110"])]
        ax = Axes(x_range=[-20, 180, 20], y_range=[0, len(items) + 1, 1], x_length=8.5, y_length=4.6, tips=False,
                  axis_config=dict(color=INK, stroke_width=2),
                  x_axis_config=dict(include_numbers=True, font_size=22, decimal_number_config=dict(color=INK, num_decimal_places=0)),
                  y_axis_config=dict(include_ticks=False, stroke_opacity=0)).shift(1.6 * RIGHT + 0.4 * UP)
        g = VGroup()
        for i, (lab, (f, c)) in enumerate(items):
            y = len(items) - i
            g.add(DashedLine(ax.c2p(-20, y), ax.c2p(180, y), color=FAINT, stroke_width=1.5))
            g.add(Dot(ax.c2p(c, y), radius=0.11, color=INK), Dot(ax.c2p(f, y), radius=0.08, color=LOAD))
            g.add(T(lab, 24).next_to(ax.c2p(-20, y), LEFT, buff=0.3))
        xl = T(r"МПа (110 МПа окружного напряжения)", 22, SUB).next_to(ax.x_axis, DOWN, buff=0.4)
        leg = VGroup(VGroup(Dot(radius=0.11, color=INK), T(r"МКЭ (CalculiX, сетка 0.04 мкм)", 22)).arrange(RIGHT, buff=0.15),
                     VGroup(Dot(radius=0.08, color=LOAD), T(r"Фурье (клетка 0.1 мкм)", 22, LOAD)).arrange(RIGHT, buff=0.15)
                     ).arrange(RIGHT, buff=0.7).next_to(ax, UP, buff=0.35)
        note = T(r"сама $g$ новой пластинки: расхождение 1--21 МПа из $\approx 2700$ (0.04--0.8\,\%)", 22, SUB).next_to(xl, DOWN, buff=0.3)
        self.add(ax, g, xl, leg, note)


# ============================================================ глава 5. зарождение
class F13_CNT(Scene):
    """Энергия зародыша от размера: барьер."""
    def construct(self):
        ax = Axes(x_range=[0, 3, 0.5], y_range=[-1.2, 1.6, 0.5], x_length=7.5, y_length=5.0, tips=True,
                  axis_config=dict(color=INK, stroke_width=2, include_ticks=False, tip_width=0.18, tip_height=0.18)).shift(2.0 * LEFT)
        xl = T(r"размер зародыша $r$", 24, SUB).next_to(ax.x_axis, DOWN, buff=0.2).align_to(ax.x_axis, RIGHT)
        yl = T(r"$\Delta G(r)$", 26, SUB).next_to(ax.y_axis, UP, buff=0.1)
        cols = (RAD, LOAD, HYDR)
        g = VGroup()
        for D, col in zip((1.0, 1.35, 1.8), cols):
            f = lambda r, D=D: 3 * r ** 2 - 2 * D * r ** 3
            rs = 1.0 / D
            r_end = next(r for r in np.linspace(0.5, 3, 500) if f(r) < -1.1)
            c = ax.plot(f, x_range=[0, r_end], color=col, stroke_width=4)
            g.add(c, Dot(ax.c2p(rs, f(rs)), color=col, radius=0.07))
            g.add(DashedLine(ax.c2p(rs, 0), ax.c2p(rs, f(rs)), color=col, stroke_width=1.5, dash_length=0.06))
        tb = VGroup(T(r"малое $\Delta$", 22, RAD), T(r"среднее", 22, LOAD), T(r"большое $\Delta$", 22, HYDR)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(ax.c2p(2.4, 1.2))
        br = T(r"барьер $\Delta G^*$", 24, RAD).next_to(ax.c2p(1.0, 1.0), UP, buff=0.12)
        right = VGroup(M(r"\Delta G(r) = \underbrace{\gamma\,S(r)}_{\text{граница}} - \underbrace{\Delta\,V(r)}_{\text{выигрыш}}", 32),
                       M(r"\Delta G^* \propto \frac{\gamma^3}{\Delta^2}", 38),
                       M(r"J = \nu\, e^{-\Delta G^*/kT}", 38),
                       T(r"$\Delta$ --- движущая сила на единицу\\объёма гидрида, МПа", 22, SUB)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([4.3, 0.3, 0])
        self.add(ax, xl, yl, g, tb, br, right)


class F14_Rate(Scene):
    """Скорость зарождения в модели: крутизна."""
    def construct(self):
        B, D0 = 80.0, 15.0
        ax = Axes(x_range=[8, 34, 2], y_range=[-80, 80, 20], x_length=8.6, y_length=5.0, tips=False,
                  axis_config=dict(color=INK, stroke_width=2, include_numbers=True, font_size=22,
                                   decimal_number_config=dict(color=INK, num_decimal_places=0))).shift(1.6 * LEFT + 0.1 * DOWN)
        xl = T(r"движущая сила $\Delta$, МПа", 24, SUB).next_to(ax.x_axis, DOWN, buff=0.45)
        yl = T(r"$\ln(r/\nu_c)$", 26, SUB).next_to(ax.y_axis, UP, buff=0.15)
        f = lambda D: -B * ((D0 / D) ** 2 - 1)
        c = ax.plot(f, x_range=[D0 / np.sqrt(2) + 0.02, 34], color=RAD, stroke_width=5)
        asym = DashedLine(ax.c2p(8, B), ax.c2p(34, B), color=DIM, stroke_width=1.5)
        at = T(r"$+B$", 22, DIM).next_to(ax.c2p(34, B), RIGHT, buff=0.1)
        d0 = DashedLine(ax.c2p(D0, -80), ax.c2p(D0, 80), color=SUB, stroke_width=1.5)
        d0t = T(r"$\Delta_0$: линия TSSP", 22, SUB).next_to(ax.c2p(D0, -80), UP, buff=0.1).shift(0.95 * RIGHT)
        tang = Line(ax.c2p(D0 - 3.5, -2 * B / D0 * 3.5), ax.c2p(D0 + 3.5, 2 * B / D0 * 3.5), color=LOAD, stroke_width=3)
        tt = T(r"наклон $2B/\Delta_0\approx 11$ на МПа\\$\approx 58$ на $1\,^\circ$C!", 22, LOAD).next_to(ax.c2p(D0 + 3.5, 2 * B / D0 * 3.5), RIGHT, buff=0.1)
        right = VGroup(M(r"r=\nu_c\exp\!\Big[-B\Big(\frac{\Delta_0^2}{\Delta^2}-1\Big)\Big]", 32),
                       M(r"\Delta=\Delta_0+n_HRT\ln\frac{c}{c_{\text{TSSP}}}+\varepsilon_n g+n_H\frac{Q}{T}\,\delta T", 26),
                       T(r"$B=80$, $\Delta_0=15$ МПа", 22, SUB)).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_corner(DR, buff=0.5)
        self.add(ax, xl, yl, asym, at, d0, d0t, c, tang, tt, right)


class F15_GB(Scene):
    """Зародыш на границе зёрен: линза и фактор формы."""
    def construct(self):
        c = np.array([-3.8, 0.0, 0])
        gb = Line(c + 3.0 * LEFT, c + 3.0 * RIGHT, color=INK, stroke_width=3)
        th = np.radians(67)
        R = 1.6
        h = R * (1 - np.cos(th))
        cap_up = Arc(radius=R, start_angle=PI / 2 - th, angle=2 * th, color=CIRC, stroke_width=4).shift(c + (-(R - h)) * UP)
        cap_dn = Arc(radius=R, start_angle=-PI / 2 - th, angle=2 * th, color=CIRC, stroke_width=4).shift(c + (R - h) * UP)
        fill = VGroup(cap_up.copy(), cap_dn.copy())
        lens = VGroup(cap_up, cap_dn)
        ga = T(r"зерно 1", 24, SUB).move_to(c + [-2.0, 1.2, 0]); gb2 = T(r"зерно 2", 24, SUB).move_to(c + [-2.0, -1.2, 0])
        x0 = R * np.sin(th)
        ang = Arc(radius=0.55, start_angle=PI, angle=-th, arc_center=c + x0 * RIGHT, color=LOAD, stroke_width=3)
        at = M(r"\theta", 32, LOAD).move_to(c + x0 * RIGHT + 0.85 * np.array([np.cos(PI - th / 2), np.sin(PI - th / 2), 0]))
        lab = VGroup(T(r"$\gamma_{\alpha\alpha}=0.14$ Дж/м$^2$ --- граница", 22, SUB), T(r"$\gamma_{\alpha\delta}=0.18$ Дж/м$^2$ --- гидрид/металл", 22, SUB),
                     M(r"\cos\theta=\frac{\gamma_{\alpha\alpha}}{2\gamma_{\alpha\delta}}=0.39", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(gb, DOWN, buff=1.4)
        ax = Axes(x_range=[0, 180, 30], y_range=[0, 1, 0.25], x_length=5.2, y_length=3.4, tips=False,
                  axis_config=dict(color=INK, stroke_width=2, include_numbers=True, font_size=20,
                                   decimal_number_config=dict(color=INK, num_decimal_places=2))).move_to([3.6, 0.6, 0])
        ax.x_axis.numbers.set_opacity(0)
        for v in (0, 60, 120, 180):
            ax.add(T(f"{v}$^\\circ$", 20).next_to(ax.c2p(v, 0), DOWN, buff=0.15))
        f = lambda t: (2 + np.cos(np.radians(t))) * (1 - np.cos(np.radians(t))) ** 2 / 4
        cur = ax.plot(f, x_range=[0, 180], color=CIRC, stroke_width=4)
        dot = Dot(ax.c2p(67, f(67)), color=LOAD, radius=0.09)
        dl = T(r"$f=0.22$", 24, LOAD).next_to(dot, UL, buff=0.08)
        yl = M(r"f(\theta)=\frac{(2+\cos\theta)(1-\cos\theta)^2}{4}", 28).next_to(ax, UP, buff=0.25)
        res = VGroup(T(r"барьер на границе $=f\cdot B$", 24),
                     M(r"\Delta_{\text{нач}}^{\text{гр}}=\Delta_0\sqrt f\ \Rightarrow\ \frac{\Delta_0(1-\sqrt f)}{5.4}\approx 1.5\,^\circ\text{C}", 28)
                     ).arrange(DOWN, buff=0.2).next_to(ax, DOWN, buff=0.7)
        self.add(gb, lens, ga, gb2, ang, at, lab, ax, cur, dot, dl, yl, res)


class F16_Race(Scene):
    """Гонка семейств зёрен: откуда порог."""
    def construct(self):
        ax = Axes(x_range=[0, 250, 50], y_range=[0, 20, 4], x_length=8.6, y_length=5.0, tips=False,
                  axis_config=dict(color=INK, stroke_width=2, include_numbers=True, font_size=22,
                                   decimal_number_config=dict(color=INK, num_decimal_places=0))).shift(1.4 * LEFT + 0.2 * DOWN)
        xl = T(r"окружное напряжение $\sigma_\theta$, МПа", 24, SUB).next_to(ax.x_axis, DOWN, buff=0.45)
        yl = T(r"фора по температуре выпадения, $^\circ$C", 24, SUB).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.55)
        bias, gbd, k = 10.9, 1.5, 0.08
        rad = ax.plot(lambda s: k * s, x_range=[0, 250], color=RAD, stroke_width=5)
        c1 = DashedLine(ax.c2p(0, bias), ax.c2p(250, bias), color=CIRC, stroke_width=4, dash_length=0.12)
        c2 = Line(ax.c2p(0, bias + gbd), ax.c2p(250, bias + gbd), color=CIRC, stroke_width=4)
        s1, s2 = bias / k, (bias + gbd) / k
        p1, p2 = Dot(ax.c2p(s1, bias), color=INK, radius=0.08), Dot(ax.c2p(s2, bias + gbd), color=INK, radius=0.08)
        v1 = DashedLine(ax.c2p(s1, 0), ax.c2p(s1, bias), color=DIM, stroke_width=1.5)
        v2 = DashedLine(ax.c2p(s2, 0), ax.c2p(s2, bias + gbd), color=DIM, stroke_width=1.5)
        l1 = T(r"радиальные: $0.08\,^\circ$C/МПа $\cdot\,\sigma_\theta$ (Vizcaíno 2014)", 22, RAD).next_to(ax.c2p(200, 16), UP, buff=0.05).shift(1.0 * LEFT)
        l2 = T(r"окружные на границах: фора зёрен + 1.5$^\circ$C", 22, CIRC).next_to(ax.c2p(5, bias + gbd), UP, buff=0.08, aligned_edge=LEFT)
        l3 = T(r"окружные в теле зерна: фора зёрен 10.9$^\circ$C", 22, CIRC).next_to(ax.c2p(5, bias), DOWN, buff=0.12, aligned_edge=LEFT)
        t1 = T(f"{s1:.0f}", 22, INK).next_to(ax.c2p(s1, 0), UP, buff=0.12).shift(0.3 * LEFT)
        t2 = T(f"{s2:.0f}", 22, INK).next_to(ax.c2p(s2, 0), UP, buff=0.12).shift(0.3 * RIGHT)
        right = VGroup(T(r"Кто раньше достиг своей\\температуры выпадения, тот\\и забирает водород", 24),
                       M(r"\sigma_{\text{порог}}=\frac{\text{фора}}{0.08\ ^\circ\text{C/МПа}}", 32),
                       T(r"два порога: 136 и 155 МПа\\(граница / тело зерна)", 22, SUB)).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_corner(UR, buff=0.4)
        self.add(ax, xl, yl, rad, c1, c2, v1, v2, p1, p2, l1, l2, l3, t1, t2, right)


# ============================================================ глава 6. водород
class F17_Diffusion(Scene):
    def construct(self):
        D0, Q = 7.9e-7, 44.4e3
        ax = Axes(x_range=[150, 450, 50], y_range=[0, 400, 100], x_length=8.0, y_length=5.0, tips=False,
                  axis_config=dict(color=INK, stroke_width=2, include_numbers=True, font_size=22,
                                   decimal_number_config=dict(color=INK, num_decimal_places=0))).shift(1.8 * LEFT + 0.1 * DOWN)
        xl = T(r"температура, $^\circ$C", 24, SUB).next_to(ax.x_axis, DOWN, buff=0.45)
        yl = T(r"$\sqrt{2Dt}$, мкм", 24, SUB).next_to(ax.y_axis, UP, buff=0.15)
        Dm = lambda T_: D0 * np.exp(-Q / (8.314 * (T_ + 273.15)))
        g = VGroup()
        for t, col, lab in ((10, HYDR, r"10 с"), (60, CIRC, r"1 мин"), (600, PLAST, r"10 мин")):
            Tmax_ = min(450, next((T_ for T_ in np.linspace(150, 450, 600) if np.sqrt(2 * Dm(T_) * t) * 1e6 > 398), 450))
            c = ax.plot(lambda T_, t=t: np.sqrt(2 * Dm(T_) * t) * 1e6, x_range=[150, Tmax_], color=col, stroke_width=4)
            g.add(c)
            Tl = 440 if t == 10 else (300 if t == 60 else 205)
            g.add(T(lab, 22, col).next_to(ax.c2p(Tl, min(400, np.sqrt(2 * Dm(Tl) * t) * 1e6)), UL, buff=0.05))
        wall = DashedLine(ax.c2p(150, 240), ax.c2p(450, 240), color=DIM, stroke_width=1.5)
        wt = T(r"поле автомата 240 мкм", 20, DIM).next_to(ax.c2p(150, 240), UR, buff=0.05)
        right = VGroup(M(r"\frac{\partial c}{\partial t}=D\,\nabla^2 c", 36),
                       M(r"D=D_0\,e^{-Q_D/RT}", 34),
                       T(r"$D_0=7.9\cdot10^{-7}$ м$^2$/с, $Q_D=44.4$ кДж/моль", 22, SUB),
                       T(r"при 300$^\circ$C: $D\approx7\cdot10^{-11}$ м$^2$/с,\\за минуту водород уходит на $\sim$90 мкм", 22, SUB)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_corner(UR, buff=0.45)
        self.add(ax, xl, yl, wall, wt, g, right)


class F18_Growth(Scene):
    """Рост кончика и обеднение вокруг пластинки."""
    def construct(self):
        n = 220; x = np.linspace(-8, 8, n); X, Y = np.meshgrid(x, x * 0.55)
        a = 2.2
        d = np.sqrt(np.maximum(np.abs(X) - a, 0) ** 2 + np.maximum(np.abs(Y) - 0.15, 0) ** 2)
        c = 1 - 0.85 * np.exp(-d / 2.2)
        cm = cmap([(0.0, "#ffffff"), (0.5, "#bfe6d4"), (1.0, "#1e9e6a")])
        img = field_image(c[::-1], 9.0, height=9.0 * 0.55, cm=cm, gamma=1.0, vmin=0.1, vmax=1.0).shift(0.4 * UP)
        s = 9.0 / 16
        pl = Rectangle(width=2 * a * s, height=0.3 * s * 1.6, fill_color=HYD_F, fill_opacity=1, stroke_color=CIRC, stroke_width=3).move_to(img)
        tipR, tipL = pl.get_right(), pl.get_left()
        ar = VGroup(Arrow(tipR, tipR + 1.1 * RIGHT, buff=0.05, color=RAD, stroke_width=6), Arrow(tipL, tipL + 1.1 * LEFT, buff=0.05, color=RAD, stroke_width=6))
        flux = VGroup(*[Arrow(img.get_center() + np.array([np.cos(t) * 3.4, np.sin(t) * 1.9, 0]), img.get_center() + np.array([np.cos(t) * 1.9, np.sin(t) * 0.9, 0]),
                              buff=0, color=HYDR, stroke_width=3, max_tip_length_to_length_ratio=0.2) for t in np.linspace(0, 2 * PI, 10, endpoint=False)])
        f = VGroup(M(r"v=k_{\text{tip}}\,\frac{D}{h}\,\frac{c-c_{\text{TSSD}}}{C_{\text{гидр}}}", 34),
                   T(r"$k_{\text{tip}}=0.05$ (допущение), $h=0.6$ мкм, $C_{\text{гидр}}\approx 15\,700$ ppm", 22, SUB)
                   ).arrange(DOWN, buff=0.18).next_to(img, DOWN, buff=0.35)
        lab = T(r"водород в растворе: светлее --- обеднено", 22, HYDR).next_to(img, UP, buff=0.12)
        self.add(img, frame_box(img), flux, pl, ar, f, lab)


# ============================================================ глава 7. автомат
class F19_Grains(Scene):
    def construct(self):
        z = load("grains.npz")
        gr = z["grains"]; psi = z["gpsi"]; ny, nx = gr.shape
        W = 6.2
        # цвет зерна по крутизне базисного следа
        dev = np.abs(np.degrees(np.arctan(np.tan(psi))))
        cm = cmap([(0.0, "#e6effc"), (0.6, "#f3eef6"), (1.0, "#fbe3e3")])
        img = field_image((dev / 90.0)[gr][::-1] * 0 + (dev / 90.0)[gr], W, cm=cm, gamma=1.0, vmin=0, vmax=1).shift(2.9 * LEFT)
        b = np.zeros(gr.shape, bool); b[:-1] |= gr[:-1] != gr[1:]; b[:, :-1] |= gr[:, :-1] != gr[:, 1:]
        rgb = np.zeros(gr.shape + (3,)); rgb[:] = hexrgb("#8a8a8a")
        bim = ImageMobject(np.dstack([rgb.astype(np.uint8), (b * 255).astype(np.uint8)]))
        bim.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]); bim.stretch_to_fit_width(W); bim.stretch_to_fit_height(W); bim.move_to(img)
        mp = Mapper((30.0, 30.0), img.get_center(), W)
        seg = VGroup()
        for k in range(len(psi)):
            ys, xs = np.nonzero(gr == k)
            if len(ys) < 40:
                continue
            cy, cx = ys.mean() * 0.1, xs.mean() * 0.1
            seg.add(*plates_group(np.array([[cy, cx, psi[k], 0.9]]), mp, 3.0))
        # грани, подходящие под правило габитуса
        cells = z["gcells"]; pf = z["gpsi_f"]; pair = z["gpair"]
        ang = lambda a, b_: np.abs((a - b_ + np.pi / 2) % np.pi - np.pi / 2)
        ok = np.minimum(ang(pf, psi[pair[:, 0]]), ang(pf, psi[pair[:, 1]])) <= np.radians(15)
        m = np.zeros(gr.size, bool); m[cells[ok]] = True; m = m.reshape(gr.shape)
        m = m | np.roll(m, 1, 0) | np.roll(m, 1, 1)
        rgb2 = np.zeros(gr.shape + (3,)); rgb2[:] = hexrgb("#d18b00")
        gim = ImageMobject(np.dstack([rgb2.astype(np.uint8), (m * 255).astype(np.uint8)]))
        gim.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]); gim.stretch_to_fit_width(W); gim.stretch_to_fit_height(W); gim.move_to(img)
        ax = axes_rt(img.get_corner(DL) + np.array([-0.9, 0.05, 0]), 0.55)
        right = VGroup(T(r"Зёрна --- ячейки Вороного,\\вытянутые по окружности ($2.5\times4.5$ мкм)", 24),
                       T(r"В каждом зерне след базисной плоскости:\\угол $\psi$ к окружности, $\pm\chi_0$ с разбросом", 24),
                       VGroup(Line(ORIGIN, 0.5 * RIGHT, color=CIRC, stroke_width=4), T(r"/", 1).set_opacity(0),
                              Line(ORIGIN, 0.5 * RIGHT, color=RAD, stroke_width=4), T(r"--- куда ляжет пластинка", 22, SUB)).arrange(RIGHT, buff=0.1),
                       VGroup(Line(ORIGIN, 0.5 * RIGHT, color=LOAD, stroke_width=5), T(r"грани, где возможен межзёренный\\гидрид: след грани в $15^\circ$ от базиса соседа", 22, SUB)).arrange(RIGHT, buff=0.15),
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([3.8, 0, 0])
        self.add(img, bim, gim, seg, frame_box(img), ax, right)


class F20_Algorithm(Scene):
    """Шаг автомата по времени."""
    def construct(self):
        steps = [(r"поле $\sigma(\mathbf x)$ от всех пластинок (Фурье, ореолы)", PLAST),
                 (r"выгода $g(\mathbf x)$ и движущая сила $\Delta(\mathbf x)$", RAD),
                 (r"скорость $r(\mathbf x)$; число зародышей --- случайное (Пуассон)", RAD),
                 (r"зародыши длиной $L_{\min}$, если вокруг хватает водорода", CIRC),
                 (r"рост кончиков, переход через границу зерна", HYDR),
                 (r"диффузия водорода за $\delta t$", HYDR),
                 (r"остывание: $T \leftarrow T-\dot T\,\delta t$", INK)]
        nodes = VGroup()
        for i, (txt, col) in enumerate(steps):
            box = RoundedRectangle(width=6.4, height=0.72, corner_radius=0.12, stroke_color=col, stroke_width=2.5, fill_color=WHITE, fill_opacity=1)
            t = T(txt, 22, col)
            if t.width > 6.1:
                t.scale_to_fit_width(6.1)
            num = T(f"{i + 1}", 22, col).next_to(box, LEFT, buff=0.15)
            nodes.add(VGroup(box, t.move_to(box), num))
        nodes.arrange(DOWN, buff=0.22).move_to([-2.4, 0, 0])
        arr = VGroup(*[Arrow(nodes[i][0].get_bottom(), nodes[i + 1][0].get_top(), buff=0.02, color=DIM, stroke_width=2.5,
                             max_tip_length_to_length_ratio=0.5) for i in range(len(steps) - 1)])
        back = CurvedArrow(nodes[-1][0].get_right() + 0.05 * RIGHT, nodes[0][0].get_right() + 0.05 * RIGHT, angle=TAU / 3.2,
                           color=DIM, stroke_width=3, tip_length=0.2)
        right = VGroup(T(r"Что хранит автомат", 26),
                       T(r"$\bullet$ карту зёрен и их ориентаций", 22, SUB), T(r"$\bullet$ список пластинок: центр, угол, длина", 22, SUB),
                       T(r"$\bullet$ поле водорода $c(\mathbf x)$", 22, SUB), T(r"$\bullet$ собственную деформацию $\varepsilon^*+\varepsilon_p$", 22, SUB),
                       T(r"$\bullet$ напряжения $\sigma(\mathbf x)$", 22, SUB),
                       T(r"шаг $\delta T\le 0.5\,^\circ$C; сетка 0.4 мкм, поле $240\times240$ мкм,\\$\sim$600 шагов, 4--7 мин на ядро", 21, DIM)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.9, 0, 0])
        self.add(nodes, arr, back, right)


# ============================================================ глава 8. меры
class F21_RHF(Scene):
    def construct(self):
        ax = Axes(x_range=[0, 90, 15], y_range=[0, 1.2, 0.5], x_length=6.2, y_length=3.6, tips=False,
                  axis_config=dict(color=INK, stroke_width=2, include_numbers=True, font_size=22),
                  x_axis_config=dict(decimal_number_config=dict(color=INK, num_decimal_places=0)),
                  y_axis_config=dict(decimal_number_config=dict(color=INK, num_decimal_places=1))).shift(3.0 * LEFT + 0.3 * DOWN)
        xl = T(r"угол следа к окружности $\varphi$, $^\circ$", 22, SUB).next_to(ax.x_axis, DOWN, buff=0.4)
        yl = T(r"вес $w(\varphi)$", 22, SUB).next_to(ax.y_axis, UP, buff=0.15)
        steps = VGroup(Line(ax.c2p(0, 0), ax.c2p(40, 0), color=CIRC, stroke_width=6), Line(ax.c2p(40, 0.5), ax.c2p(65, 0.5), color=MIDC, stroke_width=6),
                       Line(ax.c2p(65, 1), ax.c2p(90, 1), color=RAD, stroke_width=6))
        f = M(r"\mathrm{RHF}=\frac{\sum_i L_i\,w(\varphi_i)}{\sum_i L_i}", 36).move_to([3.4, 1.8, 0])
        ex = VGroup()
        rng = np.random.default_rng(2)
        box = Rectangle(width=4.2, height=2.6, stroke_color=METAL_E, fill_color=METAL, fill_opacity=1).move_to([3.4, -0.9, 0])
        for _ in range(18):
            ph = rng.choice([rng.uniform(0, 30), rng.uniform(45, 60), rng.uniform(70, 90)], p=None)
            L = rng.uniform(0.3, 0.9); pc = box.get_center() + np.array([rng.uniform(-1.7, 1.7), rng.uniform(-1.0, 1.0), 0])
            d = np.array([np.cos(np.radians(ph)), np.sin(np.radians(ph)), 0])
            col = CIRC if ph <= 40 else (MIDC if ph < 65 else RAD)
            ex.add(Line(pc - d * L / 2, pc + d * L / 2, color=col, stroke_width=4))
        note = T(r"веса Simon / PROPHET; RHF $<0.05$ --- шум", 22, SUB).next_to(box, DOWN, buff=0.2)
        self.add(ax, xl, yl, steps, f, box, ex, note)


class F22_RHCP(Scene):
    """Путь трещины наименьшей цены (RHCP)."""
    def construct(self):
        import heapq
        nx_, ny_ = 46, 30
        rng = np.random.default_rng(7)
        hyd = np.zeros((ny_, nx_), bool)
        for _ in range(30):
            y0, x0 = rng.integers(0, ny_), rng.integers(0, nx_)
            vert = rng.random() < 0.45
            L = rng.integers(3, 8)
            for k in range(L):
                yy, xx = (y0 + k, x0) if vert else (y0, x0 + k)
                if 0 <= yy < ny_ and 0 <= xx < nx_:
                    hyd[yy, xx] = True
        cost = np.where(hyd, 1 / 50, 1.0)
        dist = np.full((ny_, nx_), np.inf); prev = {}
        pq = [(cost[ny_ - 1, x], (ny_ - 1, x)) for x in range(nx_)]
        for d0, (y, x) in pq:
            dist[y, x] = d0
        heapq.heapify(pq)
        while pq:
            d0, (y, x) = heapq.heappop(pq)
            if d0 > dist[y, x]:
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, (x + dx) % nx_
                if 0 <= yy < ny_ and d0 + cost[yy, xx] < dist[yy, xx]:
                    dist[yy, xx] = d0 + cost[yy, xx]; prev[(yy, xx)] = (y, x); heapq.heappush(pq, (dist[yy, xx], (yy, xx)))
        xe = int(np.argmin(dist[0])); node = (0, xe); path = [node]
        while node in prev:
            node = prev[node]; path.append(node)
        cs = 0.135
        o = np.array([-6.6, -2.0, 0])
        sq = VGroup()
        for y in range(ny_):
            for x in range(nx_):
                if hyd[y, x]:
                    sq.add(Square(cs, fill_color=RAD if True else CIRC, fill_opacity=0.85, stroke_width=0).move_to(o + [x * cs, (ny_ - 1 - y) * cs, 0]))
        frame = Rectangle(width=nx_ * cs, height=ny_ * cs, stroke_color=METAL_E, stroke_width=2, fill_color=METAL, fill_opacity=1).move_to(o + [(nx_ - 1) * cs / 2, (ny_ - 1) * cs / 2, 0])
        pts = [o + [x * cs, (ny_ - 1 - y) * cs, 0] for y, x in path]
        line = VMobject(color=LOAD, stroke_width=5).set_points_as_corners(pts)
        top = T(r"наружная поверхность", 20, SUB).next_to(frame, UP, buff=0.08); bot = T(r"внутренняя", 20, SUB).next_to(frame, DOWN, buff=0.08)
        H = ny_; c_min = float(dist[0].min())
        rh = (1 - c_min / H) / (1 - 1 / 50)
        right = VGroup(T(r"Цена шага: металл --- 1, гидрид --- $1/50$", 24),
                       T(r"Путь наименьшей цены сквозь стенку\\(алгоритм Дейкстры)", 24),
                       M(r"\mathrm{RHCP}=\frac{1-\text{цена}/H}{1-1/50}", 34),
                       T(f"здесь: RHCP $= {rh:.2f}$", 24, LOAD),
                       T(r"0 --- гидридов на пути нет, 1 --- путь целиком по гидриду\\(PROPHET, Kim и др. 2022)", 21, SUB)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.7, 0, 0])
        self.add(frame, sq, line, top, bot, right)


# ============================================================ глава 9. результаты
class F23_Structures(Scene):
    """Структуры автомата: потолок против ореолов."""
    def construct(self):
        z = load("structures.npz")
        W = 2.95
        grid = Group()
        for row, (tag, lab) in enumerate((("cap", r"потолок $\pm180$ МПа"), ("halo", r"пластические ореолы"))):
            r = Group()
            for s in (0, 125, 175, 250):
                key = f"{tag}_{s}"
                c = np.array([0, 0, 0])
                box = Square(W, fill_color=WHITE, fill_opacity=1, stroke_color=METAL_E, stroke_width=1.5)
                mp = Mapper((240.0, 240.0), box.get_center(), W)
                pg = plates_group(z[key], mp, 1.6) if key in z.files else VGroup()
                r.add(Group(box, pg))
            r.arrange(RIGHT, buff=0.25)
            r.add(T(lab, 24, PLAST if tag == "halo" else SUB).rotate(PI / 2).next_to(r, LEFT, buff=0.15))
            grid.add(r)
        grid.arrange(DOWN, buff=0.3).move_to([0, 0.3, 0])
        heads = VGroup(*[T(f"{s} МПа", 24, LOAD if s else SUB).next_to(grid[0][i], UP, buff=0.12) for i, s in enumerate((0, 125, 175, 250))])
        leg = VGroup(VGroup(Line(ORIGIN, 0.45 * RIGHT, color=CIRC, stroke_width=4), T(r"положе $40^\circ$", 21, SUB)).arrange(RIGHT, buff=0.1),
                     VGroup(Line(ORIGIN, 0.45 * RIGHT, color=MIDC, stroke_width=4), T(r"$40$--$65^\circ$", 21, SUB)).arrange(RIGHT, buff=0.1),
                     VGroup(Line(ORIGIN, 0.45 * RIGHT, color=RAD, stroke_width=4), T(r"круче $65^\circ$", 21, SUB)).arrange(RIGHT, buff=0.1),
                     T(r"поле $240\times240$ мкм, 178 ppm, 3$^\circ$C/мин, старые параметры", 21, DIM)).arrange(RIGHT, buff=0.45).next_to(grid, DOWN, buff=0.25)
        self.add(grid, heads, leg)


class F24_Params(Scene):
    """Откуда берутся параметры."""
    def construct(self):
        cols = [(r"измерено", HYDR, [r"TSSD, TSSP (ДСК)", r"$\varepsilon_n,\ \varepsilon_t$ (решётки)", r"$D_0,\ Q_D$", r"$0.08\,^\circ$C/МПа (синхротрон)",
                                     r"$E,\ \nu,\ \sigma_y$", r"текстура, зерно (EBSD)"]),
                (r"рассчитано", PLAST, [r"поле пластинок (Фурье)", r"ореолы (Фурье + Мизес)", r"$\gamma$ $\to$ фора границы 1.5$^\circ$C",
                                         r"перевод $^\circ$C$\leftrightarrow$МПа"]),
                (r"допущено", LOAD, [r"$k_{\text{tip}}=0.05$", r"допуск габитуса $15^\circ$", r"$L_{\min},\ L_{\max}$", r"плоское сечение r--$\theta$"]),
                (r"подогнано", RAD, [r"$B=80$, $\Delta_0=15$ МПа", r"фора зёрен 10.9--12.4$^\circ$C", r"(потолок $\sigma_{\text{cap}}$ ---\\заменён ореолами)"])]
        g = VGroup()
        for i, (h, col, items) in enumerate(cols):
            head = T(h, 30, col)
            chips = VGroup()
            for it in items:
                t = T(it, 21)
                if t.width > 2.9:
                    t.scale_to_fit_width(2.9)
                r = RoundedRectangle(width=3.1, height=max(0.62, t.height + 0.25), corner_radius=0.12, stroke_color=col, stroke_width=2,
                                     fill_color=WHITE, fill_opacity=1)
                chips.add(VGroup(r, t.move_to(r)))
            chips.arrange(DOWN, buff=0.18)
            g.add(VGroup(head, chips).arrange(DOWN, buff=0.3))
        g.arrange(RIGHT, buff=0.35, aligned_edge=UP).shift(0.2 * DOWN)
        arrow = Arrow(g[0].get_top() + 0.6 * UP + 0.5 * LEFT, g[3].get_top() + 0.6 * UP + 0.5 * RIGHT, buff=0, color=DIM, stroke_width=3)
        at = T(r"чем правее, тем слабее обоснование и тем важнее независимая проверка", 22, SUB).next_to(arrow, UP, buff=0.1)
        self.add(g, arrow, at)
