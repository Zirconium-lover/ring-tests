"""Упругопластичность с критерием Хилла на Фурье: одна пластинка гидрида в периодической ячейке.

Неизвестные — перемещения; деформация ε = E + sym(D u), D — разность вперёд (символ (e^{ikh} − 1)/h).
Равновесие решается методом Ньютона, линейная задача на каждом шаге — сопряжёнными градиентами
с упругим предобуславливателем (точный обратный оператор однородной упругой среды в Фурье).
Среднее напряжение задано полностью: E подправляется по упругой податливости до совпадения.

Оси: x — окружное (θ), y — радиальное (r), z — осевое. Сетка (nx, ny, nz); nz = 1 — обобщённая
плоская деформация (всё однородно по z, ε_zz = E_zz). Тензоры в нотации Мандела:
[xx, yy, zz, √2·yz, √2·zx, √2·xy] = [θθ, rr, zz, √2·rz, √2·zθ, √2·rθ].

Хилл (1 = r, 2 = θ, 3 = z): σ̄² = F(σθ−σz)² + G(σz−σr)² + H(σr−σθ)² + 2Lσθz² + 2Mσzr² + 2Nσrθ²,
нормировка — σ̄ = σθ при одноосном окружном растяжении. Изотропия (Мизес): F = G = H = 1/2, L = M = N = 3/2.
Изотропное линейное упрочнение σ_y(p) = σ_y0 + h·p, ассоциированный закон течения.
Гидрид упругий, модули — как у матрицы (как в fe/biax_inh.py, отношение 1).

Работа превращения: W = ∫₀¹ ⟨σ⟩_пластинки : ε* dt (на единицу объёма пластинки), g = W/ε_n —
в тех же единицах, что выгода g = σ:ε*/ε_n в автомате (в упругости W = σ_прил:ε* − собственная энергия).
"""
import numpy as np
import scipy.fft as sfft

EPS_N, EPS_T = 0.0720, 0.0458
R2 = np.sqrt(2.0)
WORKERS = 4


def hill_P(F=0.5, G=0.5, H=0.5, L=None, M=None, N=None):
    """Матрица Хилла в Манделе для [θθ, rr, zz, √2rz, √2zθ, √2rθ]; сдвиговые по умолчанию F + G + H
    (как у Мизеса: 3/2 при F = G = H = 1/2)."""
    s = F + G + H
    L = s if L is None else L; M = s if M is None else M; N = s if N is None else N
    P = np.zeros((6, 6))
    P[0, 0] = F + H; P[1, 1] = G + H; P[2, 2] = F + G
    P[0, 2] = P[2, 0] = -F; P[1, 2] = P[2, 1] = -G; P[0, 1] = P[1, 0] = -H
    P[3, 3] = M; P[4, 4] = L; P[5, 5] = N
    return P / (F + H)


def misfit(normal):
    """Несоответствие пластинки в Манделе: нормаль по оси normal (0 — x/θ, 1 — y/r), в плоскости ε_t."""
    e = np.zeros(6); e[:3] = EPS_T; e[normal] = EPS_N
    return e


class Cell:
    def __init__(self, n, h, E=90e3, nu=0.34, sy=350.0, hard=200.0, P=None):
        self.n = tuple(n); self.h = h
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu)); self.mu = E / (2 * (1 + nu))
        C = np.zeros((6, 6)); C[:3, :3] = self.lam; C += 2 * self.mu * np.eye(6)
        self.C = C; self.Cinv = np.linalg.inv(C)
        self.sy, self.hard = sy, hard
        self.P = hill_P() if P is None else P
        w, Q = np.linalg.eigh(self.P)
        w[np.abs(w) < 1e-12] = 0.0
        self.pw, self.pQ = w, Q
        ks = [2 * np.pi * sfft.fftfreq(n[0], h), 2 * np.pi * sfft.fftfreq(n[1], h), 2 * np.pi * sfft.rfftfreq(n[2], h)]
        K = np.meshgrid(*ks, indexing="ij")
        self.d = [(np.exp(1j * k * h) - 1.0) / h for k in K]
        # упругий оператор D^H C sym(D ·) в Фурье: 3×3 на частоту, обратный — предобуславливатель
        Mk = np.zeros(K[0].shape + (3, 3), complex)
        for j in range(3):
            e = [np.zeros(K[0].shape, complex) for _ in range(3)]; e[j][:] = 1.0
            r = self._div_hat(self._mul_C_hat(self._grad_hat(e)))
            for i in range(3):
                Mk[..., i, j] = r[i]
        Mk[0, 0, 0] = np.eye(3)
        self.Minv = np.linalg.inv(Mk); self.Minv[0, 0, 0] = 0.0

    # --- дифференциальные операторы (в Фурье) ---
    def _grad_hat(self, u):
        d = self.d
        return [d[0] * u[0], d[1] * u[1], d[2] * u[2], (d[1] * u[2] + d[2] * u[1]) / R2,
                (d[2] * u[0] + d[0] * u[2]) / R2, (d[0] * u[1] + d[1] * u[0]) / R2]

    def _div_hat(self, s):
        c = [np.conj(v) for v in self.d]
        return [c[0] * s[0] + c[1] * s[5] / R2 + c[2] * s[4] / R2,
                c[0] * s[5] / R2 + c[1] * s[1] + c[2] * s[3] / R2,
                c[0] * s[4] / R2 + c[1] * s[3] / R2 + c[2] * s[2]]

    def _mul_C_hat(self, e):
        tr = e[0] + e[1] + e[2]
        return [self.lam * tr + 2 * self.mu * e[i] if i < 3 else 2 * self.mu * e[i] for i in range(6)]

    def rfft(self, a):
        return sfft.rfftn(a, axes=(-3, -2, -1), workers=WORKERS)

    def irfft(self, a):
        return sfft.irfftn(a, s=self.n, axes=(-3, -2, -1), workers=WORKERS)

    def grad(self, u):
        uh = self.rfft(u)
        return self.irfft(np.array(self._grad_hat(list(uh))))

    def div(self, s):
        sh = self.rfft(s)
        return self.irfft(np.array(self._div_hat(list(sh))))

    def precond(self, r):
        rh = self.rfft(r)
        z = np.einsum("...ij,j...->i...", self.Minv, rh)
        return self.irfft(z)

    # --- материал ---
    def return_map(self, sig_tr, p_n):
        """Возврат на поверхность Хилла: σ = σ_tr − 2μΔp·Pσ/σ̄, σ̄ = σ_y0 + h(p_n + Δp). Столбцы — точки."""
        s = self.pQ.T @ sig_tr
        lam = self.pw[:, None]
        mu, hard = self.mu, self.hard
        dp = np.zeros(sig_tr.shape[1])
        for _ in range(60):
            Y = self.sy + hard * (p_n + dp)
            a = 2 * mu * dp / Y
            den = 1.0 + a * lam
            S = (lam * s ** 2 / den ** 2).sum(0)
            sb = np.sqrt(S)
            phi = sb - Y
            dS = (-2 * lam ** 2 * s ** 2 / den ** 3).sum(0)
            dphi = dS * (2 * mu / Y) * (1 - dp * hard / Y) / (2 * sb) - hard
            new = dp - phi / dphi
            dp = np.where(new < 0, 0.5 * dp, new)
            if np.abs(phi).max() < 1e-10 * self.sy:
                break
        Y = self.sy + hard * (p_n + dp)
        den = 1.0 + (2 * mu * dp / Y) * lam
        sig = self.pQ @ (s / den)
        return sig, dp

    def constitutive(self, eps, ep, p, eig, matrix, tangent=True):
        """eps, ep, eig — (6, N); p — (N,); matrix — (N,) bool. Возвращает σ, индексы пластических точек,
        касательную в них (m, 6, 6), Δp и Δε_p."""
        sig = self.C @ (eps - ep - eig)
        Y = self.sy + self.hard * p
        f = np.sqrt(np.maximum((sig * (self.P @ sig)).sum(0), 0)) - Y
        idx = np.nonzero((f > 1e-9 * self.sy) & matrix)[0]
        dp = np.zeros(len(p)); dep = np.zeros_like(ep); Kt = None
        if len(idx):
            st = sig[:, idx]
            sg, d = self.return_map(st, p[idx])
            sig[:, idx] = sg; dp[idx] = d
            Yn = self.sy + self.hard * (p[idx] + d)
            dep[:, idx] = d * (self.P @ sg) / Yn
            if tangent:
                delta = 1e-7
                Kt = np.empty((len(idx), 6, 6))
                for j in range(6):
                    sj, _ = self.return_map(st + delta * self.C[:, j:j + 1], p[idx])
                    Kt[:, :, j] = ((sj - sg) / delta).T
                Kt = 0.5 * (Kt + Kt.transpose(0, 2, 1))
        return sig, idx, Kt, dp, dep

    def apply_K(self, e, idx, Kt):
        """σ = K:e (упругая всюду, касательная в пластических точках), e — (6, N)."""
        s = self.C @ e
        if idx is not None and len(idx):
            s[:, idx] = np.einsum("mij,jm->im", Kt, e[:, idx])
        return s


class Plate:
    """Расчёт одной пластинки: нагрузка Σ, затем несоответствие нарастает до 1 за nt шагов."""

    def __init__(self, cell, mask, eig):
        self.c = cell
        N = int(np.prod(cell.n))
        self.mask = mask.reshape(N)
        self.matrix = ~self.mask
        self.eig = eig[:, None] * self.mask[None, :]
        self.u = np.zeros((3,) + cell.n)
        self.E = np.zeros(6)
        self.ep = np.zeros((6, N)); self.p = np.zeros(N)
        self.N = N

    def eps_of(self, u, E):
        return E[:, None] + self.c.grad(u).reshape(6, self.N)

    def solve(self, Sig, t, tol=1e-7, tol_s=0.01, maxit=40, verbose=False):
        c = self.c
        eig = t * self.eig
        for it in range(maxit):
            eps = self.eps_of(self.u, self.E)
            sig, idx, Kt, dp, dep = c.constitutive(eps, self.ep, self.p, eig, self.matrix)
            m = Sig - sig.mean(1)
            r = -c.div(sig.reshape((6,) + c.n))
            # ГС для A δu = r, A = D^H K sym(D ·)
            A = lambda v: c.div(c.apply_K(c.grad(v).reshape(6, self.N), idx, Kt).reshape((6,) + c.n))
            du = np.zeros_like(self.u)
            rr = r.copy(); z = c.precond(rr); pp = z.copy(); rz = (rr * z).sum()
            r0 = np.sqrt((rr * rr).sum())
            k = -1
            for k in range(500 if r0 > 1e-12 * (np.abs(sig).max() + 1.0) / c.h else 0):
                Ap = A(pp)
                al = rz / (pp * Ap).sum()
                du += al * pp; rr -= al * Ap
                if np.sqrt((rr * rr).sum()) < 1e-4 * r0:
                    break
                z = c.precond(rr); rz_new = (rr * z).sum()
                pp = z + (rz_new / rz) * pp; rz = rz_new
            self.u += du
            self.E += c.Cinv @ m
            de = np.abs(c.grad(du)).max()
            if verbose:
                print(f"  it {it} пласт. точек {len(idx)} |δε| {de:.2e} |Σ−⟨σ⟩| {np.abs(m).max():.3f} ГС {k + 1}", flush=True)
            if not np.isfinite(de) or dp.max() > 0.5:
                return None, None, None, it
            if de < tol and np.abs(m).max() < tol_s:
                break
        else:
            return None, None, None, it
        eps = self.eps_of(self.u, self.E)
        sig, idx, Kt, dp, dep = c.constitutive(eps, self.ep, self.p, eig, self.matrix, tangent=False)
        return sig, dp, dep, it

    def run(self, Sig, nt=20, verbose=False, dt_min=1e-3):
        """Нагрузка, затем несоответствие от 0 до 1: шаг 1/nt, при расхождении Ньютона — делится пополам,
        начальное приближение — продолжение решения с прошлого шага. Возвращает историю:
        t, ⟨σ⟩ в пластинке, работа превращения W (на единицу объёма пластинки)."""
        Sig = np.asarray(Sig, float)
        sig, dp, dep, _ = self.solve(Sig, 0.0, verbose=verbose)
        self.ep += dep; self.p += dp
        W = 0.0; hist = []
        s_old = sig[:, self.mask].mean(1)
        e_pl = self.eig[:, self.mask][:, 0]
        t, dt = 0.0, 1.0 / nt
        du_prev, dE_prev, dt_prev = np.zeros_like(self.u), np.zeros(6), dt
        while t < 1.0 - 1e-12:
            dt = min(dt, 1.0 - t)
            u0, E0 = self.u.copy(), self.E.copy()
            self.u += du_prev * (dt / dt_prev); self.E += dE_prev * (dt / dt_prev)
            sig, dp, dep, it = self.solve(Sig, t + dt, verbose=verbose)
            if sig is None:
                self.u, self.E = u0, E0
                dt *= 0.5
                if verbose:
                    print(f"  нет сходимости, шаг {dt:.4f}", flush=True)
                if dt < dt_min:
                    raise RuntimeError("шаг слишком мал")
                continue
            t += dt
            self.ep += dep; self.p += dp
            du_prev, dE_prev, dt_prev = self.u - u0, self.E - E0, dt
            s_new = sig[:, self.mask].mean(1)
            W += 0.5 * (s_old + s_new) @ e_pl * dt
            s_old = s_new
            hist.append(dict(t=t, s_plate=s_new.tolist(), W=W, it=it, npl=int((self.p > 1e-4).sum())))
            if verbose:
                print(f"t {t:.3f} W {W:.3f} ⟨σ⟩пл {np.round(s_new[:3], 1)} пласт. {hist[-1]['npl']}", flush=True)
            if it < 6:
                dt = min(dt * 1.5, 1.0 / nt)
        self.sig = sig
        return hist


def rect_mask(n, h, a, th, normal):
    """Прямоугольная пластинка в центре ячейки: полудлина a, толщина th, нормаль по оси normal (0 — x, 1 — y);
    по z — на всю ячейку (обобщённая плоская деформация) или диск при nz > 1."""
    x = (np.arange(n[0]) + 0.5) * h - n[0] * h / 2
    y = (np.arange(n[1]) + 0.5) * h - n[1] * h / 2
    z = (np.arange(n[2]) + 0.5) * h - n[2] * h / 2
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nrm, tan = (X, Y) if normal == 0 else (Y, X)
    if n[2] == 1:
        return (np.abs(nrm) < th / 2) & (np.abs(tan) < a)
    return (np.abs(nrm) < th / 2) & (tan ** 2 + Z ** 2 < a ** 2)
