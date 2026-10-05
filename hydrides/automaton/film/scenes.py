"""Ролик «Клеточный автомат зарождения гидридов» (Manim CE), в стилистике hydrides_film_v7.
Данные — film_data.py; рендер — render.sh."""
import os
import numpy as np
from manim import *
from style import *

config.background_color = BLACK


def load(name):
    return np.load(os.path.join(DATA, name), allow_pickle=True)


def fade_all(scene, rt=0.8):
    scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=rt)


# --------------------------------------------------------------------------- 1. вступление
class S01_Intro(Scene):
    def construct(self):
        head = title_block("Как гидриды выстраиваются в пакеты", "Клеточный автомат зарождения гидридов")
        self.play(Write(head[0]), run_time=1.6)
        self.play(FadeIn(head[1], shift=0.1 * UP), run_time=0.8)
        g0 = load("growth_s0.npz"); g2 = load("growth_s250.npz")
        W = 4.4
        cL, cR = np.array([-3.4, -0.75, 0]), np.array([3.4, -0.75, 0])
        pL, pR = panel(W, W, cL), panel(W, W, cR)
        mL, mR = Mapper((200, 200), cL, W), Mapper((200, 200), cR, W)
        capL = T("без напряжения", 28, color=SUB).next_to(pL, UP, buff=0.15)
        capR = T(r"под окружным напряжением 250 МПа", 28, color=SUB).next_to(pR, UP, buff=0.15)
        self.play(FadeIn(pL), FadeIn(pR), FadeIn(capL), FadeIn(capR), FadeIn(r_axis(pL)), run_time=0.8)
        arr = sigma_arrows(pR, n=3, length=0.55, gap=0.12)
        self.play(GrowFromEdge(arr, LEFT), run_time=0.7)
        PL, PR = plates_group(g0["plates"], mL, 2.6), plates_group(g2["plates"], mR, 2.6)
        self.play(LaggedStart(*[FadeIn(x, scale=1.6) for x in PL], lag_ratio=0.02),
                  LaggedStart(*[FadeIn(x, scale=1.6) for x in PR], lag_ratio=0.02), run_time=5)
        q = T(r"Пакеты сложены из мелких пластинок. Почему они выстраиваются \\ "
              r"\textcolor[HTML]{00AFD6}{вдоль дуги} или \textcolor[HTML]{FF462D}{по радиусу}?", 30).to_edge(DOWN, buff=0.12)
        self.play(Write(q), run_time=2.0)
        self.wait(2.2)
        fade_all(self)


# --------------------------------------------------------------------------- 2. несоответствие и поле
class S02_Misfit(Scene):
    def construct(self):
        head = title_block("Гидрид больше металла", "Каждая пластинка давит на окружающий металл")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        W = 5.0; c = np.array([-3.5, -0.75, 0])
        p = panel(W, W, c)
        self.play(FadeIn(p), FadeIn(r_axis(p)), run_time=0.6)
        L0, h0 = 2.4, 0.2
        ghost = DashedVMobject(Rectangle(width=L0, height=h0, stroke_color=GREY_B, stroke_width=2).move_to(c), num_dashes=30)
        plate = RoundedRectangle(width=L0, height=h0, corner_radius=0.09, fill_color=CYAN_H, fill_opacity=1,
                                 stroke_width=0).move_to(c)
        self.play(FadeIn(plate), run_time=0.6)
        self.add(ghost)
        k = 6.0                                       # увеличение деформации для наглядности
        big = RoundedRectangle(width=L0 * (1 + 0.046 * k), height=h0 * (1 + 0.072 * k * 3), corner_radius=0.12,
                               fill_color=CYAN_H, fill_opacity=1, stroke_width=0).move_to(c)
        arrows = VGroup(
            *[Arrow(c + np.array([x, 0.12, 0]), c + np.array([x, 0.62, 0]), buff=0, color=WHITE, stroke_width=4,
                    max_tip_length_to_length_ratio=0.3) for x in (-0.7, 0, 0.7)],
            *[Arrow(c + np.array([x, -0.12, 0]), c + np.array([x, -0.62, 0]), buff=0, color=WHITE, stroke_width=4,
                    max_tip_length_to_length_ratio=0.3) for x in (-0.7, 0, 0.7)],
            Arrow(c + np.array([1.25, 0, 0]), c + np.array([1.65, 0, 0]), buff=0, color=WHITE, stroke_width=4,
                  max_tip_length_to_length_ratio=0.35),
            Arrow(c + np.array([-1.25, 0, 0]), c + np.array([-1.65, 0, 0]), buff=0, color=WHITE, stroke_width=4,
                  max_tip_length_to_length_ratio=0.35))
        self.play(Transform(plate, big), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08), run_time=1.6)
        note = T(r"(деформация увеличена для наглядности)", 22, color=DIM).next_to(p, DOWN, buff=0.15)
        self.play(FadeIn(note), run_time=0.4)
        x0 = 0.55
        l1 = TL("Объём гидрида на 12--17\\,\\% больше, чем у циркония", 28, width=6.3).move_to([x0, 1.75, 0], aligned_edge=LEFT)
        f1 = M(r"\varepsilon^*_{n} = 7{,}2\,\%", 44).move_to([x0, 0.95, 0], aligned_edge=LEFT)
        d1 = T("--- по нормали к пластинке", 26, color=SUB).next_to(f1, RIGHT, buff=0.3)
        f2 = M(r"\varepsilon^*_{t} = 4{,}6\,\%", 44).move_to([x0, 0.3, 0], aligned_edge=LEFT)
        d2 = T("--- в её плоскости", 26, color=SUB).next_to(f2, RIGHT, buff=0.3)
        self.play(FadeIn(l1), run_time=0.6)
        self.play(Write(f1), FadeIn(d1), run_time=1.0)
        self.play(Write(f2), FadeIn(d2), run_time=1.0)
        self.wait(0.8)
        # поле σ_yy вокруг пластинки (в масштабе)
        fld = load("field_one_plate.npz")
        img = div_image(fld["syy"], c, W - 0.04, 300.0)
        small = RoundedRectangle(width=W * 5 / 24, height=0.06, corner_radius=0.03, fill_color=WHITE, fill_opacity=1,
                                 stroke_width=0).move_to(c)
        self.play(FadeOut(arrows), FadeOut(ghost), FadeOut(note), Transform(plate, small), run_time=1.0)
        self.play(FadeIn(img), run_time=1.2)
        self.bring_to_front(plate)
        t1 = T("растяжение", 26, color=ManimColor("#ffb000")).move_to(c + np.array([1.55, 0.55, 0]))
        t2 = T("сжатие", 26, color=CYAN_H).move_to(c + np.array([0, 1.05, 0]))
        self.play(FadeIn(t1), FadeIn(t2), run_time=0.6)
        cap = T(r"Напряжение по нормали к пластинке, $\sigma_{rr}$", 24, color=SUB).next_to(p, DOWN, buff=0.15)
        self.play(FadeIn(cap), run_time=0.4)
        l3 = TL(r"Поле от всех пластинок считается \\ через преобразование Фурье --- одним шагом", 28, width=6.3).move_to([x0, -0.75, 0], aligned_edge=LEFT)
        f3 = M(r"\boldsymbol\varepsilon^*(\mathbf r)\xrightarrow{\ \mathcal F\ }\hat{\boldsymbol\varepsilon}^*(\mathbf k)"
               r"\longrightarrow\hat{\boldsymbol\sigma}(\mathbf k)\xrightarrow{\ \mathcal F^{-1}\ }\boldsymbol\sigma(\mathbf r)", 38)
        f3.scale_to_fit_width(6.2).move_to([x0, -1.85, 0], aligned_edge=LEFT)
        d3 = TL("микроупругость Хачатуряна, как в моделях фазового поля", 23, color=SUB, width=6.3).next_to(f3, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(FadeIn(l3), run_time=0.8)
        self.play(Write(f3), run_time=1.6)
        self.play(FadeIn(d3), run_time=0.6)
        self.wait(2.0)
        fade_all(self)


# --------------------------------------------------------------------------- 3. выгода зарождения
class S03_Favour(Scene):
    def construct(self):
        head = title_block("Где выгодно зародиться следующей", "Выгода --- работа поля над расширением новой пластинки")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        W = 5.0; c = np.array([-3.5, -0.75, 0])
        p = panel(W, W, c)
        fld = load("field_one_plate.npz")
        imgs = {k: div_image(fld[f"g{k}"], c, W - 0.04, 200.0, gamma=0.45) for k in (0, 45, 90)}
        old = RoundedRectangle(width=W * 5 / 24, height=0.06, corner_radius=0.03, fill_color=WHITE, fill_opacity=1,
                               stroke_width=0).move_to(c)
        self.play(FadeIn(p), run_time=0.4)
        self.play(FadeIn(imgs[0]), FadeIn(old), run_time=1.0)

        def icon(psi):
            a = np.radians(psi)
            ln = Line(ORIGIN, 0.55 * np.array([np.cos(a), np.sin(a), 0]), color=YELLOW_H, stroke_width=7)
            ln.set_cap_style(CapStyleType.ROUND)
            return ln.move_to(p.get_corner(UR) + np.array([-0.55, -0.55, 0]))

        ic = icon(0)
        lab = T(r"новая под $0^\circ$", 24, color=YELLOW_H).next_to(ic, DOWN, buff=0.15)
        self.play(FadeIn(ic), FadeIn(lab), run_time=0.5)
        x0 = 0.55
        f1 = M(r"g=\boldsymbol\sigma:\boldsymbol\varepsilon^*_{\text{новой}}", 46).move_to([x0, 1.6, 0], aligned_edge=LEFT)
        f2 = M(r"P\propto e^{\beta g}", 46).move_to([x0, 0.55, 0], aligned_edge=LEFT)
        d2 = T(r"$\beta = 0{,}12\ \text{МПа}^{-1}$ --- по опыту", 24, color=SUB).next_to(f2, RIGHT, buff=0.4)
        self.play(Write(f1), run_time=1.2)
        self.play(Write(f2), FadeIn(d2), run_time=1.2)
        tip1 = Circle(radius=0.22, color=YELLOW_H, stroke_width=4).move_to(c + np.array([W * 3.2 / 24, 0, 0]))
        tip2 = tip1.copy().move_to(c + np.array([-W * 3.2 / 24, 0, 0]))
        self.play(Create(tip1), Create(tip2), run_time=0.8)
        b1 = TL(r"Самое выгодное место --- \textcolor[HTML]{F5C518}{у концов} старой пластинки:\\ пластинки тянутся друг за другом цепочкой",
                28, width=6.3).move_to([x0, -0.65, 0], aligned_edge=LEFT)
        self.play(FadeIn(b1), run_time=0.8)
        self.wait(1.2)
        self.play(FadeOut(tip1), FadeOut(tip2), run_time=0.4)
        for psi in (45, 90):
            ic2 = icon(psi)
            lab2 = T(rf"новая под ${psi}^\circ$", 24, color=YELLOW_H).next_to(ic2, DOWN, buff=0.15)
            self.play(FadeOut(imgs[0] if psi == 45 else imgs[45]), FadeIn(imgs[psi]), Transform(ic, ic2),
                      Transform(lab, lab2), run_time=1.2)
            self.bring_to_front(old, ic, lab)
            self.wait(1.0)
        b2 = TL(r"Выгода зависит и от ориентации новой пластинки:\\ параллельной соседке --- больше всего", 28, color=SUB, width=6.3)
        b2.move_to([x0, -2.0, 0], aligned_edge=LEFT)
        self.play(FadeIn(b2), run_time=0.8)
        sc = VGroup(Rectangle(width=0.3, height=0.18, fill_color=ManimColor("#ffb000"), fill_opacity=1, stroke_width=0),
                    T("выгодно", 22, color=SUB), Rectangle(width=0.3, height=0.18, fill_color=CYAN_H, fill_opacity=1, stroke_width=0),
                    T("невыгодно", 22, color=SUB)).arrange(RIGHT, buff=0.15).next_to(p, DOWN, buff=0.15)
        self.play(FadeIn(sc), run_time=0.4)
        self.wait(2.0)
        fade_all(self)


# --------------------------------------------------------------------------- 4. цикл алгоритма
class S04_Algorithm(Scene):
    def construct(self):
        head = title_block("Как работает автомат", "Один шаг --- одна новая пластинка")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        z = load("steps_s0.npz")
        favs, Plist, grains = z["fav"], list(z["plates"]), z["grains"]
        # шаги, после которых пластинка действительно выросла
        seq = []
        for i in range(len(Plist) - 1):
            if len(Plist[i + 1]) > len(Plist[i]):
                seq.append((favs[i], Plist[i + 1][-1]))
        W = 5.2; c = np.array([-3.45, -0.8, 0])
        mp = Mapper((60.0, 60.0), c, W)
        p = panel(W, W, c)
        gr = grain_overlay(grains, mp, alpha=0.55).set_z_index(1)
        img = fav_image(favs[0], mp)
        self.play(FadeIn(p), FadeIn(img), FadeIn(gr), FadeIn(r_axis(p)), run_time=0.8)
        cap = T(r"поле 60\,$\times$\,60 мкм, тонкие линии --- границы зёрен", 22, color=SUB).next_to(p, DOWN, buff=0.15)
        self.play(FadeIn(cap), run_time=0.4)
        steps = VGroup(
            numbered(1, r"Поле напряжений от всех пластинок (Фурье)", 5.6),
            numbered(2, r"Выгода в каждой клетке: поле соседей, \\ текстура зерна, напряжение, запас водорода", 5.6),
            numbered(3, r"Случайный выбор клетки с вероятностью $\propto e^{\beta g}$", 5.6),
            numbered(4, r"Пластинка растёт по своей плоскости \\ до границы зерна; водород вокруг убывает", 5.6),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to([3.4, -0.35, 0])
        loop = T(r"\textit{повторять, пока не выпадет весь водород}", 24, color=SUB).next_to(steps, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(LaggedStart(*[FadeIn(s, shift=0.1 * RIGHT) for s in steps], lag_ratio=0.25), run_time=1.6)
        self.play(FadeIn(loop), run_time=0.4)
        hl = SurroundingRectangle(steps[0], color=YELLOW_H, buff=0.12, corner_radius=0.08, stroke_width=2.5)
        plates = VGroup()
        counter_lab = T("пластинок:", 26, color=SUB).next_to(p, UP, buff=0.12).align_to(p, LEFT)
        counter = Integer(0, font_size=34).next_to(counter_lab, RIGHT, buff=0.15)
        self.play(FadeIn(counter_lab), FadeIn(counter), run_time=0.3)

        def do_step(k, slow):
            fav, pl = seq[k]
            new_img = fav_image(fav, mp)
            ln = plates_group(np.array([pl]), mp, width=5.5)[0].set_z_index(2)
            ring = Circle(radius=0.2, color=YELLOW_H, stroke_width=4).move_to(mp.pt(pl[0], pl[1])).set_z_index(3)
            if slow:
                box = lambda i: SurroundingRectangle(steps[i], color=YELLOW_H, buff=0.12, corner_radius=0.08, stroke_width=2.5)
                self.play(Create(hl) if k == 0 else hl.animate.become(box(0)), run_time=0.5)
                self.play(FadeOut(img), FadeIn(new_img), run_time=0.9)
                self.play(hl.animate.become(box(1)), run_time=0.5)
                self.play(Indicate(new_img, scale_factor=1.0, color=None), run_time=0.6)
                self.play(hl.animate.become(box(2)), run_time=0.5)
                self.play(Create(ring), Flash(ring.get_center(), color=YELLOW_H, line_length=0.18, flash_radius=0.3), run_time=0.8)
                self.play(hl.animate.become(box(3)), run_time=0.5)
                self.play(GrowFromCenter(ln), FadeOut(ring), counter.animate.set_value(k + 1), run_time=0.9)
            else:
                self.play(FadeOut(img), FadeIn(new_img), FadeIn(ring), run_time=0.22)
                self.play(GrowFromCenter(ln), FadeOut(ring), counter.animate.set_value(k + 1), run_time=0.22)
            return new_img, ln

        for k in range(len(seq)):
            slow = k < 2
            if k == 2:
                self.play(hl.animate.become(SurroundingRectangle(steps, color=YELLOW_H, buff=0.15, corner_radius=0.08, stroke_width=2.5)), run_time=0.6)
                fast = T(r"дальше быстрее\dots", 24, color=YELLOW_H).next_to(loop, DOWN, buff=0.25, aligned_edge=LEFT)
                self.play(FadeIn(fast), run_time=0.3)
            new_img, ln = do_step(k, slow)
            self.remove(img); img = new_img
            plates.add(ln)
            self.bring_to_front(gr, plates)
        self.wait(1.5)
        fade_all(self)


# --------------------------------------------------------------------------- 5. зёрна и текстура
class S05_Texture(Scene):
    def construct(self):
        head = title_block("Зёрна и текстура", "В каждом зерне пластинка ложится под своим углом")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        z = load("steps_s250.npz")
        grains, gpsi = z["grains"], z["gpsi"]
        W = 5.2; c = np.array([-3.45, -0.8, 0])
        mp = Mapper((60.0, 60.0), c, W)
        p = panel(W, W, c)
        gr = grain_overlay(grains, mp, alpha=0.8)
        self.play(FadeIn(p), FadeIn(gr), FadeIn(r_axis(p)), run_time=0.8)
        ny, nx = grains.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        ticks = VGroup(); steep = []
        for g in range(len(gpsi)):
            m = grains == g
            if m.sum() < 25:
                continue
            cy, cx = yy[m].mean() * 0.4, xx[m].mean() * 0.4
            ln = plates_group(np.array([[cy, cx, gpsi[g], 1.1]]), mp, width=4.0)[0]
            ticks.add(ln)
            steep.append(abs(np.degrees(np.arctan(np.tan(gpsi[g])))) >= 55)
        self.play(LaggedStart(*[GrowFromCenter(t) for t in ticks], lag_ratio=0.004), run_time=2.2)
        x0 = 0.55
        l1 = TL(r"Пластинка гидрида ложится близко \\ к базисной плоскости кристалла", 28, width=6.3).move_to([x0, 1.65, 0], aligned_edge=LEFT)
        l2 = TL(r"Базисные полюсы наклонены от радиуса \\ на $\chi_0\approx 30^\circ$ --- по параметрам Кернса", 28, width=6.3).move_to([x0, 0.5, 0], aligned_edge=LEFT)
        f = M(r"\psi=\pm\chi_0+\text{разброс}", 42).move_to([x0, -0.45, 0], aligned_edge=LEFT)
        self.play(FadeIn(l1), run_time=0.8)
        self.play(FadeIn(l2), run_time=0.8)
        self.play(Write(f), run_time=1.0)
        self.wait(1.0)
        arr = sigma_arrows(p, n=3, length=0.5, gap=0.1)
        l3 = TL(r"Окружное напряжение делает выгодными \\ зёрна с \textcolor[HTML]{FF462D}{крутыми} пластинками", 28, width=6.3).move_to([x0, -1.55, 0], aligned_edge=LEFT)
        f2 = M(r"g_\sigma=\sigma\,\big(\varepsilon^*_{\theta\theta}-\bar\varepsilon^*\big)", 40).move_to([x0, -2.6, 0], aligned_edge=LEFT)
        self.play(GrowFromEdge(arr, LEFT), FadeIn(l3), run_time=0.9)
        anims = []
        for t, s in zip(ticks, steep):
            anims.append(t.animate.set_stroke(width=6, opacity=1.0) if s else t.animate.set_stroke(opacity=0.25))
        self.play(*anims, Write(f2), run_time=1.4)
        self.wait(2.2)
        fade_all(self)


# --------------------------------------------------------------------------- 6. правило 45°
def zigzag(psi, n, seg, start, color, vertical):
    t = np.radians(psi)
    pts = [np.array(start, float)]
    for i in range(n):
        s = 1 if i % 2 == 0 else -1
        d = np.array([s * np.cos(t), np.sin(t), 0]) if vertical else np.array([np.cos(t), s * np.sin(t), 0])
        pts.append(pts[-1] + seg * d)
    g = VGroup()
    for i in range(n):
        a, b = pts[i], pts[i + 1]
        mid, v = 0.5 * (a + b), (b - a)
        ln = Line(mid - 0.42 * v, mid + 0.42 * v, color=color, stroke_width=8)
        ln.set_cap_style(CapStyleType.ROUND)
        g.add(ln)
    return g


def axes(xr, yr, xl, yl, center, xn=None, yn=None, ydec=None, tips=True):
    yc = dict(numbers_to_include=yn) if yn is not None else {}
    if ydec is not None:
        yc["decimal_number_config"] = dict(num_decimal_places=ydec)
    ax = Axes(x_range=xr, y_range=yr, x_length=xl, y_length=yl,
              axis_config=dict(color=GREY_B, stroke_width=2, include_tip=tips, tip_length=0.18, font_size=22),
              x_axis_config=dict(numbers_to_include=xn) if xn is not None else {}, y_axis_config=yc).move_to(center)
    return ax


class S06_Rule45(Scene):
    def construct(self):
        head = title_block(r"Правило $45^\circ$", r"Куда растёт цепочка из пластинок с наклоном $\pm\psi$")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        pA = panel(6.0, 1.75, np.array([-3.55, 1.0, 0])); pB = panel(6.0, 2.75, np.array([-3.55, -1.5, 0]))
        self.play(FadeIn(pA), FadeIn(pB), run_time=0.5)
        zA = zigzag(25, 8, 0.6, [-6.15, 0.85, 0], CYAN_H, vertical=False)
        zB = zigzag(65, 7, 0.36, [-5.9, -2.65, 0], RED_H, vertical=True)
        lA = TL(r"$\pm 25^\circ$: цепочка \textcolor[HTML]{00AFD6}{вдоль дуги}", 24).next_to(pA.get_corner(UL), DR, buff=0.12)
        lB = TL(r"$\pm 65^\circ$: цепочка \textcolor[HTML]{FF462D}{по радиусу} \\ --- «колода карт»", 24).move_to([-4.6, -1.5, 0], aligned_edge=LEFT)
        self.play(FadeIn(lA), run_time=0.4)
        self.play(LaggedStart(*[GrowFromCenter(x) for x in zA], lag_ratio=0.35), run_time=2.2)
        self.play(FadeIn(lB), run_time=0.4)
        self.play(LaggedStart(*[GrowFromCenter(x) for x in zB], lag_ratio=0.35), run_time=2.2)
        data = np.loadtxt(os.path.join(DATA, "switch_curve.csv"), delimiter=",", skiprows=1)
        ax = axes([0, 90, 15], [0, 90, 15], 5.2, 3.7, [3.75, -0.35, 0], xn=[0, 15, 30, 45, 60, 75, 90], yn=[0, 45, 90])
        xl = T(r"наклон пластинок $\psi$, $^\circ$", 22, color=SUB).next_to(ax.x_axis, DOWN, buff=0.42)
        yl = T(r"направление цепочки, $^\circ$ от дуги", 22, color=SUB).next_to(ax.y_axis, UP, buff=0.1).align_to(ax, LEFT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.2)
        shL = Polygon(ax.c2p(0, 0), ax.c2p(45, 0), ax.c2p(45, 90), ax.c2p(0, 90), fill_color=CYAN_H, fill_opacity=0.1, stroke_width=0)
        shR = Polygon(ax.c2p(45, 0), ax.c2p(90, 0), ax.c2p(90, 90), ax.c2p(45, 90), fill_color=RED_H, fill_opacity=0.1, stroke_width=0)
        v45 = DashedLine(ax.c2p(45, 0), ax.c2p(45, 90), color=GREY_B, stroke_width=2)
        self.play(FadeIn(shL), FadeIn(shR), Create(v45), run_time=0.8)
        curve = VMobject(color=YELLOW_H, stroke_width=5).set_points_smoothly([ax.c2p(x, y) for x, y, _ in data])
        self.play(Create(curve), run_time=2.0)
        tr = ValueTracker(5)
        dot = always_redraw(lambda: Dot(ax.c2p(tr.get_value(), np.interp(tr.get_value(), data[:, 0], data[:, 1])), color=YELLOW_H, radius=0.09))
        self.add(dot)
        self.play(tr.animate.set_value(85), run_time=3.0, rate_func=there_and_back_with_pause)
        txt = T(r"При $45^\circ$ соседи противоположного наклона почти не взаимодействуют, и направление цепочки "
                r"\textcolor[HTML]{F5C518}{переключается}. \\ Один механизм --- и окружные пакеты, и «колода карт»", 25).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(txt), run_time=1.0)
        self.wait(2.5)
        fade_all(self)


# --------------------------------------------------------------------------- 7. проверка на снимках
class S07_Check(Scene):
    def construct(self):
        head = title_block("Проверка на снимках", "Наклон пластинок внутри пакетов: снимки Cinbiz из диссертации Lepine")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        data = np.load(os.path.join(DATA, "cinbiz_conditional.npy"), allow_pickle=True)
        name, cir, rad = data[1]
        bins = np.arange(0, 91, 5)
        ax = axes([0, 90, 15], [0, 22, 5], 6.6, 3.5, [-3.0, -0.15, 0], xn=[0, 15, 30, 45, 60, 75, 90], yn=[0, 5, 10, 15, 20])
        xl = T(r"наклон пластинки к дуге, $^\circ$", 22, color=SUB).next_to(ax.x_axis, DOWN, buff=0.42)
        yl = T(r"доля длины, \%", 22, color=SUB).next_to(ax.y_axis, UP, buff=0.1).align_to(ax, LEFT)
        cap = T(r"1 цикл охлаждения под 225 МПа", 22, color=DIM).next_to(ax, UP, buff=0.15).align_to(ax, RIGHT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(cap), run_time=1.0)
        v45 = DashedLine(ax.c2p(45, 0), ax.c2p(45, 20), color=GREY_B, stroke_width=2)
        self.play(Create(v45), run_time=0.4)

        def step(v, col):
            h, _ = np.histogram(v, bins=bins); h = h / h.sum() * 100
            pts = []
            for i in range(len(h)):
                pts += [ax.c2p(bins[i], h[i]), ax.c2p(bins[i + 1], h[i])]
            return VMobject(color=col, stroke_width=5).set_points_as_corners(pts)

        sc, sr = step(cir, CYAN_H), step(rad, RED_H)
        lc = T(r"в окружных пакетах", 22, color=CYAN_H).move_to(ax.c2p(13, 19.6), aligned_edge=LEFT)
        lr = T(r"в радиальных пакетах", 24, color=RED_H).move_to(ax.c2p(70, 12.3))
        self.play(Create(sc), FadeIn(lc), run_time=1.6)
        self.play(Create(sr), FadeIn(lr), run_time=1.6)
        rows = VGroup(
            TL(r"Медиана наклона \\ (с поправкой на метод):", 26),
            TL(r"\textcolor[HTML]{00AFD6}{окружные пакеты}: $\approx 19^\circ$ \\ \textcolor[HTML]{A8A8A8}{вручную у Cinbiz $19{,}4^\circ$}", 25),
            TL(r"\textcolor[HTML]{FF462D}{радиальные пакеты}: $\approx 47$--$52^\circ$", 25),
            TL(r"\textcolor[HTML]{A8A8A8}{автомат: $24^\circ$ и $58$--$67^\circ$}", 25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to([1.35, -0.25, 0], aligned_edge=LEFT)
        for r in rows:
            self.play(FadeIn(r, shift=0.05 * RIGHT), run_time=0.6)
        concl = VGroup(T1(r"Пластинки окружных пакетов --- \textcolor[HTML]{00AFD6}{положе $45^\circ$}, радиальных --- \textcolor[HTML]{FF462D}{круче}.", 24),
                       T1(r"Предсказание подтверждается, хотя распределение в радиальных пакетах широкое", 24, color=SUB)
                       ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(concl), run_time=1.0)
        self.wait(3.0)
        fade_all(self)


# --------------------------------------------------------------------------- 8. рост
class S08_Growth(Scene):
    def construct(self):
        head = title_block("Рост", "Каждая пластинка --- там, где выгоднее; фон --- карта выгоды зарождения")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        names = [("growth_s0.npz", "0 МПа"), ("growth_s100.npz", "100 МПа"), ("growth_s250.npz", "250 МПа")]
        W = 4.05
        xs = [-4.55, 0.0, 4.55]
        trk = ValueTracker(0.0)
        mobs = []
        for (fn, lab), x in zip(names, xs):
            z = load(fn)
            c = np.array([x, -0.65, 0])
            mp = Mapper((200.0, 200.0), c, W)
            p = panel(W, W, c)
            frac = z["frac"]; n = z["n"]; favs = z["fav"]; P = z["plates"]
            img = fav_image(favs[0], mp)
            pl = plates_group(P, mp, width=2.3)
            for ln in pl:
                ln.set_stroke(opacity=0.0)
            cap = T(r"$\sigma=$ " + lab, 28, color=SUB).next_to(p, UP, buff=0.12)
            rh = T("RHF", 26, color=SUB)
            num = DecimalNumber(0.0, num_decimal_places=2, font_size=32)
            row = VGroup(rh, num).arrange(RIGHT, buff=0.2).next_to(p, DOWN, buff=0.18)
            self.play(FadeIn(p), FadeIn(img), FadeIn(cap), FadeIn(row), run_time=0.5)
            self.add(pl)
            mobs.append(dict(img=img, pl=pl, num=num, frac=frac, n=n, favs=favs, P=P, mp=mp, last=[-1]))

        def upd_factory(d):
            def upd(_):
                t = trk.get_value()
                k = int(np.argmin(np.abs(d["frac"] - t * d["frac"][-1])))
                if k != d["last"][0]:
                    from style import FAV_CMAP, rgba_image
                    x = (d["favs"][k] + 5.0) / 5.0
                    d["img"].pixel_array = rgba_image(FAV_CMAP(np.clip(x, 0, 1) ** 1.4))
                    nn = d["n"][k] if t < 0.999 else len(d["P"])
                    for i, ln in enumerate(d["pl"]):
                        ln.set_stroke(opacity=1.0 if i < nn else 0.0)
                    d["num"].set_value(rhf_of(d["P"][:max(nn, 1)]))
                    d["last"][0] = k
            return upd

        for d in mobs:
            d["img"].add_updater(upd_factory(d))
        self.play(trk.animate.set_value(1.0), run_time=22, rate_func=linear)
        for d in mobs:
            d["img"].clear_updaters()
        self.wait(0.6)
        self.play(*[FadeOut(d["img"]) for d in mobs], run_time=1.2)
        leg = legend_row().scale(0.9).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(leg), run_time=0.5)
        self.wait(2.5)
        fade_all(self)



# --------------------------------------------------------------------------- 9. стенка при изгибе
class S09_Wall(Scene):
    def construct(self):
        head = title_block("Стенка канала Э635 при изгибе", "Растяжение снаружи, сжатие внутри --- как у кольца при сжатии плитами")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        w = load("wall.npz")
        H = 4.75; Wd = H * 240 / 800; cy = -0.75
        cases = [("wall_free_seed1", "без нагрузки"), ("wall_unif200_seed1", "$+200$ МПа"), ("wall_bend200_seed1", r"изгиб $\pm200$ МПа")]
        xs = [-5.75, -3.6, -1.45]
        trk = ValueTracker(0.0)
        groups = []
        for (k, lab), x in zip(cases, xs):
            c = np.array([x, cy, 0])
            mp = Mapper((800.0, 240.0), c, Wd)
            p = panel(Wd, H, c)
            cap = T(lab, 22, color=SUB).next_to(p, UP, buff=0.1)
            P = w[k]
            pl = plates_group(P, mp, width=2.0)
            for ln in pl:
                ln.set_stroke(opacity=0.0)
            self.play(FadeIn(p), FadeIn(cap), run_time=0.35)
            self.add(pl)
            groups.append((pl, P))
        xr = xs[2] + Wd / 2 + 0.12
        arrs = VGroup()
        for f in np.linspace(0.06, 0.94, 9):
            y = cy + H / 2 - f * H
            sgn = 1 - 2 * f
            if abs(sgn) < 0.08:
                continue
            L = 0.55 * abs(sgn)
            if sgn > 0:
                arrs.add(Arrow([xr, y, 0], [xr + L, y, 0], buff=0, color=ORANGE_H, stroke_width=4, max_tip_length_to_length_ratio=0.35))
            else:
                arrs.add(Arrow([xr + L, y, 0], [xr, y, 0], buff=0, color=CYAN_H, stroke_width=4, max_tip_length_to_length_ratio=0.35))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.08), run_time=1.0)

        def upd(_):
            t = trk.get_value()
            for pl, P in groups:
                nn = int(t * len(P))
                for i, ln in enumerate(pl):
                    ln.set_stroke(opacity=1.0 if i < nn else 0.0)
        holder = Dot(radius=0.001, fill_opacity=0).add_updater(upd)
        self.add(holder)
        self.play(trk.animate.set_value(1.0), run_time=9, rate_func=linear)
        holder.clear_updaters()
        # профиль RHF по толщине: глубина откладывается вниз
        x_left, x_right = 1.45, 4.75
        ytop, ybot = cy + H / 2, cy - H / 2

        def P2(v, d):
            return np.array([x_left + v * (x_right - x_left), ytop - d / 800 * (ytop - ybot), 0])

        frame_ax = VGroup(Line(P2(0, 0), P2(0, 800), color=GREY_B, stroke_width=2), Line(P2(0, 0), P2(1.0, 0), color=GREY_B, stroke_width=2))
        ticks = VGroup()
        for v in (0, 0.5, 1.0):
            ticks.add(Line(P2(v, 0), P2(v, 0) + 0.08 * UP, color=GREY_B, stroke_width=2),
                      T(f"{v:.1f}".replace(".", ","), 20, color=SUB).next_to(P2(v, 0), UP, buff=0.12))
        for d in (0, 400, 800):
            ticks.add(Line(P2(0, d), P2(0, d) + 0.08 * LEFT, color=GREY_B, stroke_width=2),
                      T(str(d), 20, color=SUB).next_to(P2(0, d), LEFT, buff=0.12))
        xl = T("RHF в слое", 22, color=SUB).next_to(P2(0.5, 0), UP, buff=0.45)
        yl = T("глубина от наружной поверхности, мкм", 20, color=SUB).rotate(PI / 2).next_to(P2(0, 400), LEFT, buff=0.55)
        self.play(Create(frame_ax), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=1.0)

        def prof(P, nb=8, Hum=800.0):
            dev = np.degrees(np.abs(np.arctan(np.tan(P[:, 2])))); L = 2 * P[:, 3]
            wgt = np.where(dev <= 40, 0, np.where(dev < 65, 0.5, 1.0))
            e = np.linspace(0, Hum, nb + 1); out = []
            for a, b in zip(e[:-1], e[1:]):
                m = (P[:, 0] >= a) & (P[:, 0] < b)
                out.append((L[m] * wgt[m]).sum() / max(L[m].sum(), 1e-9))
            return 0.5 * (e[1:] + e[:-1]), np.array(out)

        cols = [GREY_B, YELLOW_H, RED_H]
        labs = ["без нагрузки", "$+200$ МПа", r"изгиб $\pm200$"]
        for (pl, P), col in zip(groups, cols):
            yb, v = prof(P)
            pts = [P2(vv, d) for d, vv in zip(yb, v)]
            ln = VMobject(color=col, stroke_width=4).set_points_as_corners(pts)
            dots = VGroup(*[Dot(pt, radius=0.05, color=col) for pt in pts])
            self.play(Create(ln), FadeIn(dots), run_time=0.9)
        leg = VGroup(*[VGroup(Line(ORIGIN, 0.4 * RIGHT, color=c, stroke_width=4), T(t, 20, color=SUB)).arrange(RIGHT, buff=0.12)
                       for c, t in zip(cols, labs)]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        leg.next_to(P2(1.0, 400), RIGHT, buff=0.25)
        self.play(FadeIn(leg), run_time=0.5)
        txt = T(r"При изгибе радиальные гидриды --- \textcolor[HTML]{FF462D}{только в растянутой части стенки}", 26).to_edge(DOWN, buff=0.1)
        self.play(FadeIn(txt), run_time=0.8)
        self.wait(3.0)
        fade_all(self)


# --------------------------------------------------------------------------- 10. порог и текстура
class S10_Threshold(Scene):
    def construct(self):
        head = title_block("Порог переориентации и текстура", "Калибровка по Lepine и прогноз для Э635")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        s = load("series.npz")
        ax = axes([0, 300, 50], [0, 1, 0.2], 5.3, 3.3, [-3.55, -0.2, 0], xn=[0, 100, 200, 300], yn=[0, 0.5, 1.0], ydec=1)
        xl = T(r"окружное напряжение при охлаждении, МПа", 21, color=SUB).next_to(ax.x_axis, DOWN, buff=0.4)
        yl = T("RHF", 22, color=SUB).next_to(ax.y_axis, UP, buff=0.1)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)
        S = [0, 100, 150, 200, 250]
        m = [float(np.mean(s[f"rhf{v}"])) for v in S]
        line = VMobject(color=YELLOW_H, stroke_width=4).set_points_as_corners([ax.c2p(a, b) for a, b in zip(S, m)])
        dots = VGroup(*[Dot(ax.c2p(a, b), color=YELLOW_H, radius=0.07) for a, b in zip(S, m)])
        lep = VGroup(*[Square(0.16, fill_color=WHITE, fill_opacity=1, stroke_width=0).move_to(ax.c2p(a, b))
                       for a, b in ((0, 0.073), (200, 0.595), (250, 0.819))])
        lepH = VGroup(*[Square(0.15, stroke_color=WHITE, stroke_width=2.5).rotate(PI / 4).move_to(ax.c2p(a, b))
                        for a, b in ((0, 0.074), (200, 0.552), (250, 0.714))])
        self.play(Create(line), FadeIn(dots), run_time=1.4)
        self.play(FadeIn(lep), FadeIn(lepH), run_time=0.8)
        lg = VGroup(VGroup(Dot(color=YELLOW_H, radius=0.07), T("автомат", 21, color=SUB)).arrange(RIGHT, buff=0.12),
                    VGroup(Square(0.14, fill_color=WHITE, fill_opacity=1, stroke_width=0), T("Lepine, MATLAB", 21, color=SUB)).arrange(RIGHT, buff=0.12),
                    VGroup(Square(0.13, stroke_color=WHITE, stroke_width=2.5).rotate(PI / 4), T("Lepine, HAPPy", 21, color=SUB)).arrange(RIGHT, buff=0.12)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(ax.c2p(225, 0.22))
        self.play(FadeIn(lg), run_time=0.5)
        ax2 = axes([18, 46, 4], [0, 150, 50], 4.9, 3.3, [3.75, -0.2, 0], xn=[22, 30, 38, 46], yn=[0, 50, 100, 150])
        xl2 = T(r"наклон базисных полюсов $\chi_0$, $^\circ$", 21, color=SUB).next_to(ax2.x_axis, DOWN, buff=0.4)
        yl2 = T(r"порог (RHF\,=\,0,35), МПа", 21, color=SUB).next_to(ax2.y_axis, UP, buff=0.1).align_to(ax2, LEFT)
        self.play(Create(ax2), FadeIn(xl2), FadeIn(yl2), run_time=1.0)
        pts = [(22, 132, "Zr-1Nb", UR), (30, 67, "Zry-4", UP), (33, 64, r"Э635 внутр.", DL), (38.5, 40, r"Э635 средн.", UR), (41.5, 27, None, UR)]
        curve = VMobject(color=GREY_B, stroke_width=2).set_points_as_corners([ax2.c2p(a, b) for a, b, *_ in pts])
        self.play(Create(curve), run_time=0.8)
        cols = [CYAN_H, WHITE, TEAL_H, RED_H, RED_H]
        for (a, b, lab, dr), col in zip(pts, cols):
            d = Dot(ax2.c2p(a, b), color=col, radius=0.09)
            self.play(FadeIn(d, scale=1.5), run_time=0.3)
            if lab:
                self.play(FadeIn(T(lab, 21, color=col).next_to(d, dr, buff=0.1)), run_time=0.3)
        v45 = DashedLine(ax2.c2p(45, 0), ax2.c2p(45, 150), color=GREY_B, stroke_width=2)
        self.play(Create(v45), FadeIn(T(r"$45^\circ$", 21, color=SUB).next_to(v45, UP, buff=0.05)), run_time=0.5)
        txt = VGroup(T1(r"Чем ближе текстура к $45^\circ$, тем легче переориентация: \textcolor[HTML]{F5C518}{канал Э635 --- легче Zry-4 и Zr-1Nb}", 24),
                     T1(r"абсолютные пороги модель занижает; надёжны порядок и относительный эффект", 23, color=SUB)
                     ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(txt), run_time=1.0)
        self.wait(3.0)
        fade_all(self)


# --------------------------------------------------------------------------- 11. выводы
class S11_Summary(Scene):
    def construct(self):
        head = title_block("Выводы")
        self.play(Write(head[0]), run_time=1.0)
        items = VGroup(
            numbered(1, r"Один упругий механизм объясняет и окружные пакеты, и «колоду карт»: \\ направление цепочки "
                        r"\textcolor[HTML]{F5C518}{переключается при наклоне пластинок $45^\circ$}", 11.0),
            numbered(2, r"Автомат, подобранный по доле радиальных гидридов, сам даёт шаг \\ между радиальными пакетами "
                        r"\textcolor[HTML]{F5C518}{71--75 мкм} (опыт Lepine --- 71 мкм)", 11.0),
            numbered(3, r"Прогноз: \textcolor[HTML]{FF462D}{канал Э635 переориентируется легче Zry-4}; при изгибе \\ "
                        r"радиальные гидриды появляются только в растянутой части стенки", 11.0),
            numbered(4, r"Чего не хватает: \textcolor[HTML]{A8A8A8}{резкого порога}. Нужен коллективный эффект, \\ "
                        r"который усиливает растущий радиальный пакет", 11.0),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        items.move_to([0, -0.55, 0]).to_edge(LEFT, buff=1.0)
        for it in items:
            self.play(FadeIn(it[0], scale=1.3), FadeIn(it[1], shift=0.1 * RIGHT), run_time=1.0)
            self.wait(2.2)
        self.wait(2.5)
        fade_all(self, 1.2)


# --------------------------------------------------------------------------- 8b. рост при пяти напряжениях
class S08b_Series(Scene):
    def construct(self):
        head = title_block("Порог: рост при разных напряжениях", r"Один и тот же материал, охлаждение под напряжением $\sigma$")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        s = load("series.npz")
        S = [0, 100, 150, 200, 250]
        W = 2.5
        xs = np.linspace(-5.6, 5.6, 5)
        trk = ValueTracker(0.0)
        groups = []
        for v, x in zip(S, xs):
            c = np.array([x, -0.35, 0])
            mp = Mapper((240.0, 240.0), c, W)
            p = panel(W, W, c)
            cap = T(rf"$\sigma = {v}$ МПа", 24, color=SUB).next_to(p, UP, buff=0.12)
            P = s[f"s{v}"]
            pl = plates_group(P, mp, width=1.7)
            for ln in pl:
                ln.set_stroke(opacity=0.0)
            num = DecimalNumber(0.0, num_decimal_places=2, font_size=30)
            row = VGroup(T("RHF", 22, color=SUB), num).arrange(RIGHT, buff=0.15).next_to(p, DOWN, buff=0.15)
            self.play(FadeIn(p), FadeIn(cap), FadeIn(row), run_time=0.3)
            self.add(pl)
            groups.append((pl, P, num))

        def upd(_):
            t = trk.get_value()
            for pl, P, num in groups:
                nn = int(t * len(P))
                for i, ln in enumerate(pl):
                    ln.set_stroke(opacity=1.0 if i < nn else 0.0)
                num.set_value(rhf_of(P[:max(nn, 1)]))
        holder = Dot(radius=0.001, fill_opacity=0).add_updater(upd)
        self.add(holder)
        self.play(trk.animate.set_value(1.0), run_time=12, rate_func=linear)
        holder.clear_updaters()
        # полоска «доля радиальных» под панелями
        bars = VGroup()
        for (pl, P, num), x in zip(groups, xs):
            r = rhf_of(P)
            base = np.array([x - W / 2, -3.05, 0])
            bg = Rectangle(width=W, height=0.16, fill_color=PANEL, fill_opacity=1, stroke_color=BORDER, stroke_width=1.5).move_to(base + np.array([W / 2, 0, 0]))
            fg = Rectangle(width=max(W * r, 0.01), height=0.16, fill_color=RED_H, fill_opacity=1, stroke_width=0).align_to(bg, LEFT).move_to(bg, coor_mask=[0, 1, 0])
            bars.add(bg, fg)
        self.play(FadeIn(bars), run_time=0.8)
        txt = T(r"Окружные пакеты сменяются радиальными уже между \textcolor[HTML]{F5C518}{100 и 150 МПа}; "
                r"в опыте переход резче и около 155 МПа", 25).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(txt), run_time=0.8)
        self.wait(3.0)
        fade_all(self)


# --------------------------------------------------------------------------- 8c. крупный план пакетов
class S08c_Zoom(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()
        head = title_block("Из чего сложены пакеты", "Крупный план: отдельные пластинки внутри пакета")
        self.play(Write(head[0]), FadeIn(head[1]), run_time=1.4)
        g0 = load("growth_s0.npz"); g2 = load("growth_s250.npz")
        W = 4.4
        cL, cR = np.array([-3.4, -0.75, 0]), np.array([3.4, -0.75, 0])
        mL, mR = Mapper((200, 200), cL, W), Mapper((200, 200), cR, W)
        pL, pR = panel(W, W, cL), panel(W, W, cR)
        PL, PR = plates_group(g0["plates"], mL, 2.4), plates_group(g2["plates"], mR, 2.4)
        capL = T("0 МПа", 26, color=SUB).next_to(pL, UP, buff=0.12)
        capR = T("250 МПа", 26, color=SUB).next_to(pR, UP, buff=0.12)
        self.play(FadeIn(pL), FadeIn(pR), FadeIn(PL), FadeIn(PR), FadeIn(capL), FadeIn(capR), run_time=1.0)
        self.wait(0.5)
        # радиальный пакет
        fw = 3.6                                                       # ширина кадра при наезде
        zR = mR.pt(145, 125)
        boxR = Square(side_length=44 * mR.s, color=YELLOW_H, stroke_width=3).move_to(zR)
        self.play(Create(boxR), run_time=0.6)
        self.play(self.camera.frame.animate.set(width=fw).move_to(zR + np.array([0.75, 0, 0])),
                  FadeOut(boxR), PR.animate.set_stroke(width=1.1), run_time=2.0)
        lab = TL(r"\textcolor[HTML]{FF462D}{Радиальный пакет}: \\ пластинки наклонены \\ на $45$--$70^\circ$ в обе стороны \\ "
                 r"и уложены друг над другом \\ --- «колода карт»", 6.5).move_to(zR + np.array([0.68, 0, 0]), aligned_edge=LEFT)
        lab = VGroup(BackgroundRectangle(lab, color=PANEL, fill_opacity=0.85, buff=0.04), lab)
        self.play(FadeIn(lab), run_time=0.8)
        self.wait(2.5)
        self.play(FadeOut(lab), Restore(self.camera.frame), PR.animate.set_stroke(width=2.4), run_time=1.6)
        # окружный пакет
        zL = mL.pt(115, 165)
        boxL = Square(side_length=44 * mL.s, color=YELLOW_H, stroke_width=3).move_to(zL)
        self.play(Create(boxL), run_time=0.6)
        self.play(self.camera.frame.animate.set(width=fw).move_to(zL + np.array([-0.75, 0, 0])),
                  FadeOut(boxL), PL.animate.set_stroke(width=1.1), run_time=2.0)
        lab2 = TL(r"\textcolor[HTML]{00AFD6}{Окружный пакет}: \\ пластинки наклонены \\ на $\pm 15$--$30^\circ$ \\ "
                  r"и тянутся зигзагом \\ вдоль дуги", 6.5).move_to(zL + np.array([-1.05, 0.6, 0]))
        lab2 = VGroup(BackgroundRectangle(lab2, color=PANEL, fill_opacity=0.85, buff=0.04), lab2)
        self.play(FadeIn(lab2), run_time=0.8)
        self.wait(2.5)
        self.play(FadeOut(lab2), Restore(self.camera.frame), PL.animate.set_stroke(width=2.4), run_time=1.6)
        self.wait(0.5)
        fade_all(self)
