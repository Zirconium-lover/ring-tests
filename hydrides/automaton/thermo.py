"""Водород в цирконии: пределы растворимости и ход выпадения гидридов при охлаждении.

lg C[ppm] = b − k·1000/T[K] — Плясов и др., Ядерная физика и инжиниринг, 2023, 14(1), 12–21, табл. 2
(ДСК, 10–20 °C/мин). TSSD в пределах погрешности одна для Э635, Э110опт, Zry-2, Zry-4, Zr-1Nb, M5,
ZIRLO; TSSP — кинетическая линия, зависит от скорости охлаждения и максимальной температуры.
В табл. 2 у TSSD Э635 опечатка: D = 2.9 кДж/моль, по k = 2.24 должно быть 42.9.

Выпадение при охлаждении считаем квазиравновесным по линии TSSP: в растворе c(T) = min(C, TSSP(T)).
"""
import numpy as np

TSS = {
    "E635": dict(TSSD=(5.56, 2.24), TSSP=(5.08, 1.68)),
    "E110opt": dict(TSSD=(5.66, 2.32), TSSP=(5.29, 1.85)),
}
H_IN_HYDRIDE_PPM = 18013.0      # δ-ZrH1.66: массовая доля водорода
VOL_HYD_PER_ZR = 1.15           # объём гидрида на объём металла с тем же цирконием (ρ 6.51 / 5.65)
R = 8.314


def c_line(T_C, bk):
    b, k = bk
    return 10.0 ** (b - k * 1000.0 / (np.asarray(T_C, float) + 273.15))


def T_line(c_ppm, bk):
    b, k = bk
    return 1000.0 * k / (b - np.log10(np.asarray(c_ppm, float))) - 273.15


def area_fraction(c_ppm):
    """Доля площади (объёма) гидрида, если выпал водород c_ppm."""
    return np.asarray(c_ppm, float) / H_IN_HYDRIDE_PPM * VOL_HYD_PER_ZR


class Cooling:
    """Ход выпадения: нагрев до T_max (растворяется до TSSD(T_max)), охлаждение до T_end по TSSP."""

    def __init__(self, H_ppm, T_max, T_end=20.0, lines="E635"):
        self.L = TSS[lines] if isinstance(lines, str) else lines
        self.H, self.T_max, self.T_end = float(H_ppm), float(T_max), float(T_end)
        self.dissolved = float(min(self.H, c_line(self.T_max, self.L["TSSD"])))
        self.undissolved = self.H - self.dissolved
        self.c_end = float(min(self.dissolved, c_line(self.T_end, self.L["TSSP"])))
        self.T_start = float(min(self.T_max, T_line(self.dissolved, self.L["TSSP"])))

    @property
    def frac_new(self):
        """Доля площади гидрида, выпадающего при охлаждении."""
        return float(area_fraction(self.dissolved - self.c_end))

    @property
    def frac_left(self):
        """Доля площади нерастворившегося гидрида."""
        return float(area_fraction(self.undissolved))

    def T(self, progress):
        """Температура, при которой выпала доля progress (0..1) охлаждаемого водорода."""
        c = self.dissolved - np.clip(progress, 0, 1) * (self.dissolved - self.c_end)
        return float(min(self.T_start, T_line(max(c, 1e-6), self.L["TSSP"])))

    def supersaturation(self, T_C):
        """Пересыщение на линии выделения S = TSSP/TSSD и Δμ = RT ln S, Дж/моль H."""
        S = c_line(T_C, self.L["TSSP"]) / c_line(T_C, self.L["TSSD"])
        return S, R * (T_C + 273.15) * np.log(S)

    def summary(self):
        return dict(H=self.H, T_max=self.T_max, dissolved=self.dissolved, undissolved=self.undissolved,
                    T_dissolution=float(T_line(self.H, self.L["TSSD"])), T_start=self.T_start,
                    frac_new=self.frac_new, frac_left=self.frac_left)
