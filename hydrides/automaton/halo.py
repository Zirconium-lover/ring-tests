"""Пластические ореолы пластинок для упругого Фурье автомата (вместо обрезки поля соседей потолком σ_cap).

Основание (fe/halo_runs.py, fe/halo_check.py): при одинаковых модулях гидрида и матрицы напряжения
упругопластического решения равны упругому решению с собственной деформацией ε* + ε_p. Ореол ε_p одиночной
пластинки (матрица Мизеса, σ_y 350 МПа, упрочнение 200 МПа; обобщённая плоская деформация) посчитан на Фурье
для полудлин A_TAB, углов α между окружным напряжением и нормалью пластинки и уровней σθ; ореолы соседей
складываются. Проверка на колоде и цепочке из трёх пластинок: g на месте следующей пластинки — ошибка ~60 МПа
против ~170 у потолка, знак у колоды верный (у потолка — нет).

Таблица (halo_tab.npz, собирается fe/halo_table.py): для каждой полудлины a — поле ε_p (tt, nn, zz, tn) в осях
пластинки на сетке 0.1 мкм в окне |s| ≤ a + W_S, |q| ≤ W_Q, для α ∈ AL и σθ ∈ S (σθ = 0 — общий для всех α).
Промежуточные σθ и α — линейно; длина — «растяжением» ближайшей меньшей из таблицы (концевые зоны
раздвигаются, середина заполняется средним столбцом); короче наименьшей — её ореол с множителем a/a_min.
Знак угла: для ψ < 0 ореол отражается q → −q (сдвиг tn меняет знак)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class HaloTable:
    def __init__(self, path=os.path.join(HERE, "data_halo", "halo_tab.npz")):
        z = np.load(path)
        self.A = z["A"]; self.AL = z["AL"]; self.S = z["S"]; self.h = float(z["h"])
        self.WS = float(z["WS"]); self.WQ = float(z["WQ"])
        # F[ia][iS, iα] — (4, ns, nq): tt, nn, zz, tn; iS = 0 — без нагрузки (одно для всех α)
        self.F = [z[f"F{i}"] for i in range(len(self.A))]
        self.sigma = None

    def set_load(self, sigma):
        """Поля для данного σθ (линейно по таблице уровней, ниже первого — от нулевой нагрузки)."""
        S = np.concatenate([[0.0], self.S])
        s = float(np.clip(sigma, 0.0, S[-1]))
        j = int(np.clip(np.searchsorted(S, s) - 1, 0, len(S) - 2))
        w = (s - S[j]) / (S[j + 1] - S[j])
        self.G = [(1 - w) * F[j] + w * F[j + 1] for F in self.F]          # (nα, 4, ns, nq)
        self.sigma = s

    def field(self, a, alpha):
        """ε_p в осях пластинки на сетке h: (4, ns, nq) и полуокно по s; α в градусах, 0…90."""
        ia = int(np.searchsorted(self.A, a, side="right") - 1)
        al = float(np.clip(alpha, self.AL[0], self.AL[-1]))
        k = int(np.clip(np.searchsorted(self.AL, al) - 1, 0, len(self.AL) - 2))
        wa = (al - self.AL[k]) / (self.AL[k + 1] - self.AL[k])
        if ia < 0:                                                         # короче наименьшей
            G = self.G[0]
            return ((1 - wa) * G[k] + wa * G[k + 1]) * (a / self.A[0]), self.A[0] + self.WS
        G = self.G[ia]
        f = (1 - wa) * G[k] + wa * G[k + 1]
        m = int(round((a - self.A[ia]) / self.h))                          # растяжение на 2m столбцов
        if m > 0:
            mid = f.shape[1] // 2
            f = np.concatenate([f[:, :mid], np.repeat(f[:, mid:mid + 1], 2 * m, axis=1), f[:, mid:]], axis=1)
        return f, self.A[ia] + m * self.h + self.WS

    def patch(self, shape, dx, c, psi, half):
        """Ореол пластинки (центр c = (y, x) мкм, угол ψ от TD, полудлина) на сетке автомата: индексы строк и
        столбцов окна и компоненты e11, e22, e12, e33 (глобальные оси, y — вниз, как в plate_eigen)."""
        psi = (psi + np.pi / 2) % np.pi - np.pi / 2
        alpha = 90.0 - abs(np.degrees(psi))
        f, s_max = self.field(half, alpha)
        if psi < 0:
            f = f[:, :, ::-1].copy(); f[3] *= -1.0
        ny, nx = shape
        t = np.array([-np.sin(psi), np.cos(psi)]); n = np.array([t[1], -t[0]])      # (y, x)
        r = s_max * np.abs(t) + self.WQ * np.abs(n) + dx
        y0, y1 = int((c[0] - r[0]) / dx) - 1, int((c[0] + r[0]) / dx) + 2
        x0, x1 = int((c[1] - r[1]) / dx) - 1, int((c[1] + r[1]) / dx) + 2
        ys = np.arange(y0, y1); xs = np.arange(x0, x1)
        k = max(1, int(round(dx / self.h)))
        off = (np.arange(k) + 0.5) / k
        Y = (ys[:, None, None, None] + off[None, None, :, None]) * dx - c[0]
        X = (xs[None, :, None, None] + off[None, None, None, :]) * dx - c[1]
        s = Y * t[0] + X * t[1]; q = Y * n[0] + X * n[1]
        ns, nq = f.shape[1], f.shape[2]
        i = np.rint(s / self.h + (ns - 1) / 2).astype(int); j = np.rint(q / self.h + (nq - 1) / 2).astype(int)
        inside = (i >= 0) & (i < ns) & (j >= 0) & (j < nq)
        i = np.clip(i, 0, ns - 1); j = np.clip(j, 0, nq - 1)
        loc = [np.where(inside, f[m][i, j], 0.0).mean(axis=(2, 3)) for m in range(4)]       # tt, nn, zz, tn
        tx, ty, nx_, ny_ = t[1], t[0], n[1], n[0]
        tt, nn, zz, tn = loc
        e11 = tt * tx * tx + nn * nx_ * nx_ + 2 * tn * tx * nx_
        e22 = tt * ty * ty + nn * ny_ * ny_ + 2 * tn * ty * ny_
        e12 = tt * tx * ty + nn * nx_ * ny_ + tn * (tx * ny_ + ty * nx_)
        return ys % ny, xs % nx, (e11, e22, e12, zz)
