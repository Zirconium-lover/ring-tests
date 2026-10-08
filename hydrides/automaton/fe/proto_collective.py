"""Прототип коллективной пластичности на сетке автомата — вместо сложения таблиц ореолов.

Пластическая деформация матрицы — поле состояния ε_p (e11, e22, e12, e33). На каждое приращение (новые пластинки):
  1) упругое решение Фурье (ca_hydride.Elastic, free_z) с собственной деформацией ε*(гидриды) + ε_p, плюс нагрузка;
  2) в клетках матрицы — пробное напряжение без пластики этого приращения: σ_tr = σ + 2μ·Δε_p(дев.);
     возврат по Мизесу с линейным упрочнением: Δp = ⟨σ_tr,eq − (σ_y + h·p_n)⟩ / (3μ + h), Δε_p = 1.5·Δp·s_tr/σ_tr,eq;
  3) повторять 1–2 до сходимости (локальный предиктор, неподвижная точка).
Пластика сама собирается из суммарного поля: нагрузка + все гидриды + накопленная ε_p; зоны сливаются, у предела
текучести течёт общая область, после растворения ε_p остаётся.

Проверка — эталоны halo_yield_check.py (упругопластика hill_fft, сетка 0.1 мкм, ячейка 20 мкм, σ_y 226, h 200):
одиночная пластинка 1.8 × 0.6 мкм при 0 / 110 / 200 МПа и стопки «бок о бок» (шаг 1.6 мкм по нормали).
Мера — g радиальной пластинки (σ:ε*/ε_n) на месте следующей минус прямая добавка нагрузки, и площадь
пластической зоны (p > 1e-3).
python proto_collective.py папка_эталонов [шаг сетки, мкм]"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from ca_hydride import Elastic, plate_eigen  # noqa: E402
from halo import HaloTable  # noqa: E402
import halo_yield_check as R  # noqa: E402

E_MOD, NU = 90e3, 0.34
MU = E_MOD / (2 * (1 + NU))


class Collective:
    """Матрица Мизеса с линейным упрочнением на сетке автомата; гидрид упругий (клетки с долей ≥ 0.5)."""

    def __init__(self, shape, dx, sy, hard, Sig=(0.0, 0.0, 0.0, 0.0)):
        self.el = Elastic(shape, dx, E_MOD, NU, free_z=True)
        self.el_dx = dx
        self.sy, self.hard = sy, hard
        self.Sig = np.asarray(Sig, float)                  # нагрузка (S11 — окружное, S22, S12, S33)
        self.Es = [np.zeros(shape) for _ in range(4)]      # ε* гидридов
        self.Ep = [np.zeros(shape) for _ in range(4)]      # ε_p матрицы
        self.p = np.zeros(shape)
        self.hyd = np.zeros(shape)
        self.its = []

    def add_plates(self, plates):
        for c, psi, half, th in plates:
            ys, xs, fr, comps = plate_eigen(self.Es[0].shape, self.el_dx, np.asarray(c, float), psi, half, th)
            sel = np.ix_(ys, xs)
            self.hyd[sel] = np.clip(self.hyd[sel] + fr, 0, 1)
            for a, v in zip(self.Es, comps):
                a[sel] += fr * v

    def stress(self, Ep):
        S = self.el.stress(*[a + b for a, b in zip(self.Es, Ep)])
        return [s + g for s, g in zip(S, self.Sig)]

    def solve(self, tol=0.5, maxit=200):
        """Равновесие приращения: Δε_p ≥ 0 по p, локальный предиктор. tol — МПа превышения текучести."""
        mtx = self.hyd < 0.5
        Y = self.sy + self.hard * self.p
        Ep0 = [a.copy() for a in self.Ep]
        dEp = [np.zeros_like(a) for a in self.Ep]
        dp = np.zeros_like(self.p)
        for it in range(maxit):
            S = self.stress([a + b for a, b in zip(Ep0, dEp)])
            # пробное напряжение: убрать пластику этого приращения локально (σ_tr = σ + 2μ·Δε_p)
            T = [s + 2 * MU * d for s, d in zip(S, dEp)]
            m = (T[0] + T[1] + T[3]) / 3.0
            s11, s22, s33, s12 = T[0] - m, T[1] - m, T[3] - m, T[2]
            eq = np.sqrt(1.5 * (s11 ** 2 + s22 ** 2 + s33 ** 2 + 2 * s12 ** 2))
            f = np.where(mtx, eq - Y, -1.0)
            ndp = np.maximum(f, 0.0) / (3 * MU + self.hard)
            k = np.where(eq > 0, 1.5 * ndp / np.maximum(eq, 1e-12), 0.0)
            new = [k * s11, k * s22, k * s12, k * s33]
            # невязка: превышение текучести при текущем Δε_p (σ, а не σ_tr)
            mS = (S[0] + S[1] + S[3]) / 3.0
            eqS = np.sqrt(1.5 * ((S[0] - mS) ** 2 + (S[1] - mS) ** 2 + (S[3] - mS) ** 2 + 2 * S[2] ** 2))
            over = float(np.max(np.where(mtx, eqS - (Y + self.hard * dp), -np.inf))) if mtx.any() else 0.0
            change = max(float(np.abs(n - d).max()) for n, d in zip(new, dEp))
            dEp, dp = new, ndp
            if over < tol and change < 1e-7:
                break
        self.Ep = [a + b for a, b in zip(Ep0, dEp)]
        self.p = self.p + dp
        self.its.append(it + 1)
        return it + 1


def g_rad(S):
    """Выгода радиальной пластинки (нормаль по x — окружное)."""
    return (R.hf.EPS_N * S[0] + R.hf.EPS_T * S[1] + R.hf.EPS_T * S[3]) / R.hf.EPS_N


def site_mask(n, dx, step):
    yy = (np.arange(n) + 0.5) * dx - R.L / 2
    Yr, Xc = np.meshgrid(yy, yy, indexing="ij")
    return (np.abs(Xc - step[1]) < R.TH / 2 + 0.2) & (np.abs(Yr - step[0]) < R.A)


def run_case(conf, s, dx):
    """Стопка (или одиночная): сначала две старые пластинки, затем третья — каждое приращение до равновесия."""
    n = int(round(R.L / dx))
    col = Collective((n, n), dx, R.SY, R.HARD, Sig=(s, 0, 0, 0))
    cs, step = R.centres("side" if conf == "single" else conf)   # одиночная — третья пластинка «бок о бок»
    to_ = lambda c: (R.L / 2 + c[0], R.L / 2 + c[1])            # (вдоль, нормаль) → (строка y, столбец x)
    groups = [[cs[2]]] if conf == "single" else [cs[:2], [cs[2]]]
    t0 = time.time()
    col.solve()                                                 # нагрузка без гидридов
    for grp in groups:
        col.add_plates([(to_(c), np.pi / 2, R.A, R.TH) for c in grp])
        col.solve()
    S = col.stress(col.Ep)
    site = site_mask(n, dx, step) & (col.hyd < 0.2)
    g = float(g_rad(S)[site].mean()) - s * R.E_RAD[0] / R.hf.EPS_N
    return dict(g=g, p_area=float((col.p > 1e-3).sum() * dx * dx), its=col.its, time=time.time() - t0)


def run_halo_sum(conf, s, dx):
    """То, что делает автомат сейчас: упругое + таблица ореолов Э635 (сумма)."""
    return R.automaton(conf, s)[True]


def ref_single(D, s, step):
    """Эталон для одиночной пластинки: g на месте следующей «бок о бок» и площадь пластической зоны."""
    z = np.load(os.path.join(D, f"single_side_{s:g}.npz"))
    X, Y = R.coords()
    site = R.mask_at(*step) & ~R.mask_at(0.0, 0.0)
    g = float(R.g_of(z["sig"].astype(float))[site].mean()) - s * R.E_RAD[0] / R.hf.EPS_N
    return g, float((z["p"] > 1e-3).sum() * R.H * R.H)


if __name__ == "__main__":
    D = sys.argv[1]
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    print(f"сетка {dx} мкм; g — вклад соседей в выгоду радиальной пластинки на месте следующей, МПа")
    for s in R.LOADS:
        # одиночная пластинка: место «бок о бок» через 1.6 мкм по нормали
        gref, aref = ref_single(D, s, (0.0, 1.6))
        r = run_case("single", s, dx)
        print(f"одиночная σ {s:5.0f}: эталон g {gref:6.0f}, зона {aref:5.1f} мкм² | коллективная g {r['g']:6.0f}, "
              f"зона {r['p_area']:5.1f} мкм², итераций {r['its']}, {r['time']:.1f} с")
    for s in R.LOADS:
        fr = os.path.join(D, f"stack_side_{s:g}.npz")
        if not os.path.exists(fr):
            continue
        cs, step = R.centres("side")
        hyd = R.mask_at(*cs[0]) | R.mask_at(*cs[1]) | R.mask_at(*cs[2])
        site = R.mask_at(*step) & ~hyd
        ref = np.load(fr)
        gref = float(R.g_of(ref["sig"].astype(float))[site].mean()) - s * R.E_RAD[0] / R.hf.EPS_N
        r = run_case("side", s, dx)
        au = R.automaton("side", s)
        print(f"бок о бок σ {s:5.0f}: эталон g {gref:6.0f} (зона {float((ref['p'] > 1e-3).sum() * R.H * R.H):5.1f}) | "
              f"коллективная {r['g']:6.0f} (зона {r['p_area']:5.1f}, итераций {r['its']}) | "
              f"сумма ореолов {au[True]:6.0f} | упругое {au[False]:6.0f}")
