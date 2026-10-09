"""Слитые поэлементные ядра (numba) для упругого решения и пластичности: тот же порядок арифметики, что в numpy-коде,
без временных массивов — результат побитно тот же (проверка: opt_check.py). Без numba — None, и вызывающий код
идёт прежним путём."""
import numpy as np

import os
try:
    import numba as nb
except Exception:                                   # numba нет — прежний путь
    nb = None
if os.environ.get("HYD_NO_NUMBA") == "1":           # принудительно прежний путь (проверка)
    nb = None

if nb is not None:
    @nb.njit(cache=True)
    def elastic_spec(s11, s22, s12, kx, ky, c0, a, e11, e22, e12):
        """Спектр деформаций: t = k·τ, u = K⁻¹t, ε = sym(k u); порядок действий как в Elastic.stress."""
        ny, nh = s11.shape
        for i in range(ny):
            kyi = ky[i, 0]
            for j in range(nh):
                kxj = kx[0, j]
                t1 = kxj * s11[i, j] + kyi * s12[i, j]
                t2 = kxj * s12[i, j] + kyi * s22[i, j]
                kt = kxj * t1 + kyi * t2
                aij = a[i, j]; cij = c0[i, j]
                u1 = cij * (t1 - aij * kxj * kt)
                u2 = cij * (t2 - aij * kyi * kt)
                e11[i, j] = kxj * u1
                e22[i, j] = kyi * u2
                e12[i, j] = 0.5 * (kxj * u2 + kyi * u1)
        e11[0, 0] = 0.0; e22[0, 0] = 0.0; e12[0, 0] = 0.0

    @nb.njit(cache=True)
    def elastic_real(E11, E22, E12, e11, e22, e12, e33, tr, m11, m22, m12, E33, lam, mu2, S11, S22, S12, S33):
        """σ = λ tr(ε − ε*) I + 2μ (ε − ε*) в реальном пространстве; порядок действий как в Elastic.stress."""
        ny, nx = E11.shape
        for i in range(ny):
            for j in range(nx):
                a11 = E11[i, j] + m11
                a22 = E22[i, j] + m22
                a12 = E12[i, j] + m12
                ekk = a11 + a22 + E33 - tr[i, j]
                le = lam * ekk
                S11[i, j] = le + mu2 * (a11 - e11[i, j])
                S22[i, j] = le + mu2 * (a22 - e22[i, j])
                S12[i, j] = mu2 * (a12 - e12[i, j])
                S33[i, j] = le + mu2 * (E33 - e33[i, j])

    @nb.njit(cache=True)
    def over_max(S0, S1, S2, S3, sxx, szz, Y, h, dp, mtx):
        """max по матрице превышения текучести (Мизес) — как max(Plasticity._over(...))."""
        ny, nx = S0.shape
        best = -np.inf
        for i in range(ny):
            for j in range(nx):
                if not mtx[i, j]:
                    continue
                s11 = S0[i, j] + sxx[i, j]; s33 = S3[i, j] + szz[i, j]
                m = (s11 + S1[i, j] + s33) / 3.0
                d11 = s11 - m; d22 = S1[i, j] - m; d33 = s33 - m; d12 = S2[i, j]
                eq = np.sqrt(1.5 * (d11 ** 2 + d22 ** 2 + d33 ** 2 + 2.0 * d12 ** 2))
                v = eq - (Y[i, j] + h * dp[i, j])
                if v > best:
                    best = v
        return best

    @nb.njit(cache=True)
    def step(S0, S1, S2, S3, sxx, szz, dE0, dE1, dE2, dE3, Y, mtx, K, hard, n0, n1, n2, n3, ndp):
        """Шаг локального предиктора — как Plasticity._step: новое Δε_p и Δp (в выходные массивы)."""
        ny, nx = S0.shape
        den = 1.5 * K + hard
        for i in range(ny):
            for j in range(nx):
                s11 = S0[i, j] + sxx[i, j]; s33 = S3[i, j] + szz[i, j]
                m = (s11 + S1[i, j] + s33) / 3.0
                t11 = s11 - m + K * dE0[i, j]
                t22 = S1[i, j] - m + K * dE1[i, j]
                t33 = s33 - m + K * dE3[i, j]
                t12 = S2[i, j] + K * dE2[i, j]
                eqt = np.sqrt(1.5 * (t11 ** 2 + t22 ** 2 + t33 ** 2 + 2.0 * t12 ** 2))
                over = eqt - Y[i, j]
                d = (over if over > 0.0 else 0.0) if mtx[i, j] else 0.0
                d = d / den
                k = 1.5 * d / (eqt if eqt > 1e-12 else 1e-12)
                n0[i, j] = k * t11; n1[i, j] = k * t22; n2[i, j] = k * t12; n3[i, j] = k * t33
                ndp[i, j] = d

    @nb.njit(cache=True)
    def over_arr(S0, S1, S2, S3, sxx, szz, Y, h, dp, mtx, out):
        """Превышение текучести по клеткам — как Plasticity._over (вне матрицы −∞)."""
        ny, nx = S0.shape
        for i in range(ny):
            for j in range(nx):
                if not mtx[i, j]:
                    out[i, j] = -np.inf
                    continue
                s11 = S0[i, j] + sxx[i, j]; s33 = S3[i, j] + szz[i, j]
                m = (s11 + S1[i, j] + s33) / 3.0
                d11 = s11 - m; d22 = S1[i, j] - m; d33 = s33 - m; d12 = S2[i, j]
                eq = np.sqrt(1.5 * (d11 ** 2 + d22 ** 2 + d33 ** 2 + 2.0 * d12 ** 2))
                out[i, j] = eq - (Y[i, j] + h * dp[i, j])

else:
    elastic_spec = elastic_real = over_max = step = over_arr = None
