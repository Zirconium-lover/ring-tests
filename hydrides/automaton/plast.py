"""Коллективная пластичность матрицы на сетке автомата — вместо сложения таблиц ореолов (halo.py).

Пластическая деформация матрицы ε_p (e11, e22, e12, e33; e12 — тензорная) — поле состояния. После каждого
решения упругой задачи (новые пластинки, подросшие кончики) пластичность доводится до равновесия:
  пробное напряжение без пластики этого приращения (локальный предиктор): σ_tr = σ + 2μ·Δε_p (девиатор);
  возврат по Мизесу с линейным упрочнением: Δp = ⟨σ_tr,eq − (σ_y + h·p_n)⟩ / (3μ + h), Δε_p = 1.5·Δp·s_tr/σ_tr,eq;
  поправка напряжений — упругое решение Фурье от изменения Δε_p (задача линейна, суперпозиция точна);
  повторять, пока превышение текучести в матрице больше tol (неподвижная точка).
Нагрузка входит в проверку текучести (σ_xx — окружное, σ_zz — осевое у труб), в поле S автомата — нет (как и раньше).
Зоны соседних пластинок сливаются сами, у предела текучести течёт общая область, после растворения ε_p остаётся.

Проверка (fe/proto_collective.py): одиночная пластинка и стопка «бок о бок», Э635 (σ_y 226, h 200), 0 / 110 / 200 МПа —
карта выгоды совпадает с упругопластическим эталоном hill_fft (сетка 0.1 мкм) и с CalculiX; таблицы ореолов на
сетке 0.4 мкм завышают максимум выгоды у гидрида в 1.5–3 раза (у стопки при 200 МПа — втрое)."""
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi


class Plasticity:
    def __init__(self, el, shape, sy, hard, E=90e3, nu=0.34, wmax=160, margin=6):
        self.el, self.sy, self.hard = el, float(sy), float(hard)
        self.mu = E / (2.0 * (1.0 + nu))
        self.shape = shape
        self.Ep = [np.zeros(shape) for _ in range(4)]
        self.p = np.zeros(shape)
        self.its = []
        self.wmax, self.margin = int(wmax), int(margin)
        self._Kw = None                                 # функция влияния: отклик σ_i на единичную ε_p,j в одной клетке
        self._kf = {}                                   # её Фурье для окон данного размера

    def _kernel(self):
        """Функция влияния на смещениях ±wmax: четыре упругих решения от единичной собственной деформации в клетке."""
        if self._Kw is None:
            ny, nx = self.shape
            W = self.wmax
            iy = np.arange(-W, W + 1) % ny; ix = np.arange(-W, W + 1) % nx
            Kw = []
            for j in range(4):
                E = [np.zeros(self.shape) for _ in range(4)]
                E[j][0, 0] = 1.0
                Kw.append([a[np.ix_(iy, ix)] for a in self.el.stress(*E)])
            self._Kw = Kw                               # Kw[j][i] — (2W+1)², центр (W, W)
        return self._Kw

    def _kfft(self, wy, wx):
        key = (wy, wx)
        if key not in self._kf:
            Kw, W = self._kernel(), self.wmax
            Py, Px = 2 * wy, 2 * wx
            dy = np.arange(-(wy - 1), wy); dx = np.arange(-(wx - 1), wx)
            out = []
            for j in range(4):
                row = []
                for i in range(4):
                    pad = np.zeros((Py, Px))
                    pad[np.ix_(dy % Py, dx % Px)] = Kw[j][i][np.ix_(W + dy, W + dx)]
                    row.append(sfft.rfft2(pad))
                out.append(row)
            self._kf[key] = out
        return self._kf[key]

    def _conv(self, dd, wy, wx):
        """Изменение напряжений в окне от изменения ε_p в нём же (точная свёртка с функцией влияния)."""
        KF = self._kfft(wy, wx)
        Py, Px = 2 * wy, 2 * wx
        F = [sfft.rfft2(d, s=(Py, Px)) for d in dd]
        return [sfft.irfft2(sum(F[j] * KF[j][i] for j in range(4)), s=(Py, Px))[:wy, :wx] for i in range(4)]

    @staticmethod
    def _eq(d11, d22, d33, d12):
        return np.sqrt(1.5 * (d11 ** 2 + d22 ** 2 + d33 ** 2 + 2.0 * d12 ** 2))

    def _step(self, S, sxx, dE, Y, mtx, K, szz=0.0):
        """Один шаг локального предиктора: новое Δε_p (за эту релаксацию) и Δp."""
        s11 = S[0] + sxx; s33 = S[3] + szz
        m = (s11 + S[1] + s33) / 3.0
        t11, t22, t33, t12 = s11 - m + K * dE[0], S[1] - m + K * dE[1], s33 - m + K * dE[3], S[2] + K * dE[2]
        eqt = self._eq(t11, t22, t33, t12)
        ndp = np.where(mtx, np.maximum(eqt - Y, 0.0), 0.0) / (1.5 * K + self.hard)
        k = 1.5 * ndp / np.maximum(eqt, 1e-12)
        return [k * t11, k * t22, k * t12, k * t33], ndp

    def _over(self, S, sxx, Y, dp, mtx, szz=0.0):
        s11 = S[0] + sxx; s33 = S[3] + szz
        m = (s11 + S[1] + s33) / 3.0
        eq = self._eq(s11 - m, S[1] - m, s33 - m, S[2])
        return np.where(mtx, eq - (Y + self.hard * dp), -np.inf)

    def relax(self, S, mtx, sxx_app=0.0, tol=5.0, maxit=100, c=0.5, outer=6, szz_app=0.0):
        """Как relax_global, но итерации — в окнах вокруг клеток, где превышена текучесть (функция влияния, точная
        внутри окна), а упругое решение на всём поле — одно на внешний проход (дальнее поле). Окно больше wmax —
        общая текучесть: глобальная итерация. Возвращает число итераций (сумма по окнам и проходам)."""
        mu, h = self.mu, self.hard
        Y = self.sy + h * self.p
        dE = [np.zeros_like(self.p) for _ in range(4)]
        dp = np.zeros_like(self.p)
        sxx = sxx_app if np.ndim(sxx_app) else np.full(self.shape, float(sxx_app))
        szz = szz_app if np.ndim(szz_app) else np.full(self.shape, float(szz_app))   # осевое приложенное (труба)
        applied = [np.zeros_like(self.p) for _ in range(4)]   # Δε_p, уже учтённое в S
        n_it = 0
        for _ in range(outer):
            over = self._over(S, sxx, Y, dp, mtx, szz)
            act = over > tol
            if not act.any():
                break
            lab, nl = ndi.label(ndi.binary_dilation(act, iterations=self.margin))
            boxes = ndi.find_objects(lab)
            if any(b[0].stop - b[0].start > self.wmax or b[1].stop - b[1].start > self.wmax for b in boxes):
                # общая текучесть: окна не помогают — глобальная итерация (с учётом уже сделанного Δε_p)
                n_it += self.relax_global(S, mtx, sxx, tol, maxit, c, dE0=dE, dp0=dp, applied=applied, szz_app=szz)
                return self._finish(dE, dp, n_it)
            for sl in boxes:
                wy, wx = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                wy8, wx8 = -(-wy // 8) * 8, -(-wx // 8) * 8           # округление — меньше размеров окон в кэше
                Sw = [a[sl].copy() for a in S]
                dEw = [a[sl].copy() for a in dE]; dpw = dp[sl].copy()
                mw, Yw, sw, zw = mtx[sl], Y[sl], sxx[sl], szz[sl]
                cc = c
                for it in range(maxit):
                    ow = float(np.max(self._over(Sw, sw, Yw, dpw, mw, zw)))
                    if ow < tol:
                        break
                    if not np.isfinite(ow) or ow > 1e4:                # разошлось — заново, осторожно
                        Sw = [a[sl].copy() for a in S]; dEw = [a[sl].copy() for a in dE]; dpw = dp[sl].copy()
                        cc = 1.0
                        continue
                    new, ndp = self._step(Sw, sw, dEw, Yw, mw, 2 * mu * cc, zw)
                    dd = [n - d for n, d in zip(new, dEw)]
                    dS = self._conv([np.pad(d, ((0, wy8 - wy), (0, wx8 - wx))) for d in dd], wy8, wx8)
                    for a, b in zip(Sw, dS):
                        a += b[:wy, :wx]
                    dEw, dpw = new, ndp
                    n_it += 1
                for a, b in zip(dE, dEw):
                    a[sl] = b
                dp[sl] = dpw
            # дальнее поле: точное упругое решение от всего Δε_p, не учтённого в S
            dS = self.el.stress(*[a - b for a, b in zip(dE, applied)])
            for a, b in zip(S, dS):
                a += b
            applied = [a.copy() for a in dE]
        return self._finish(dE, dp, n_it)

    def _finish(self, dE, dp, n_it):
        for a, d in zip(self.Ep, dE):
            a += d
        self.p += dp
        self.its.append(n_it)
        return n_it

    def relax_global(self, S, mtx, sxx_app=0.0, tol=5.0, maxit=100, c=0.5, dE0=None, dp0=None, applied=None, szz_app=0.0):
        """S = [S11, S22, S12, S33] — внутренние напряжения автомата (с прежней ε_p), правятся на месте.
        mtx — клетки матрицы (пластичны); sxx_app — нагрузка по x (число или поле). c — жёсткость клетки в предикторе
        в долях 2μ: зажатая окружением клетка снимает своё напряжение слабее свободной (0.25–0.43 от 2μ по сетке 0.4),
        c = 0.5 вдвое сокращает итерации при том же решении; меньше 0.45 — расходится (на этот случай откат к c = 1).
        Возвращает число итераций."""
        mu, h = self.mu, self.hard
        Y = self.sy + h * self.p
        own = dE0 is None                               # вызов сам по себе — итог в Ep/p здесь же
        dE = dE0 if dE0 is not None else [np.zeros_like(self.p) for _ in range(4)]
        dp = dp0 if dp0 is not None else np.zeros_like(self.p)
        if applied is not None:                         # довести S до текущего Δε_p
            dS = self.el.stress(*[a - b for a, b in zip(dE, applied)])
            for a, b in zip(S, dS):
                a += b
        S0 = [a.copy() for a in S]; dE00 = [a.copy() for a in dE]; dp00 = dp.copy()
        it = 0
        for it in range(maxit):
            s11 = S[0] + sxx_app; s33 = S[3] + szz_app
            m = (s11 + S[1] + s33) / 3.0
            d11, d22, d33, d12 = s11 - m, S[1] - m, s33 - m, S[2]
            eq = np.sqrt(1.5 * (d11 ** 2 + d22 ** 2 + d33 ** 2 + 2.0 * d12 ** 2))
            over = float(np.max(np.where(mtx, eq - (Y + h * dp), -np.inf))) if mtx.any() else -1.0
            if over < tol:
                break
            if not np.isfinite(over) or over > 1e4:           # разошлось — откат и осторожный предиктор
                for a, b in zip(S, S0):
                    a[...] = b
                dE = [a.copy() for a in dE00]; dp = dp00.copy()
                c = 1.0
                continue
            # пробное: убрать пластику этого приращения локально (Δε_p — девиатор, среднее не меняется)
            K = 2 * mu * c                                  # жёсткость клетки в предикторе
            t11, t22, t12, t33 = d11 + K * dE[0], d22 + K * dE[1], d12 + K * dE[2], d33 + K * dE[3]
            eqt = np.sqrt(1.5 * (t11 ** 2 + t22 ** 2 + t33 ** 2 + 2.0 * t12 ** 2))
            ndp = np.where(mtx, np.maximum(eqt - Y, 0.0), 0.0) / (1.5 * K + h)   # согласовано с K: в точке eq(σ) = Y + h·Δp
            k = 1.5 * ndp / np.maximum(eqt, 1e-12)
            new = [k * t11, k * t22, k * t12, k * t33]
            dS = self.el.stress(*[n - d for n, d in zip(new, dE)])
            for a, b in zip(S, dS):
                a += b
            dE, dp = new, ndp
        if own:
            return self._finish(dE, dp, it)
        for a, b in zip(dE0, dE):                       # вернуть итог вызывающему (relax)
            a[...] = b
        dp0[...] = dp
        return it
