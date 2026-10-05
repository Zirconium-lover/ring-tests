"""Трёхмерный автомат зарождения гидридов (пилот): куб с периодическими границами.

Оси: x — TD (по окружности), y — ND (радиус), z — L (ось трубы). Массивы [ix, iy, iz].
• Зёрна — трёхмерные ячейки Вороного, вытянутые по осям (TD, ND, L) = grain_um.
• Ось c зерна: от радиуса к TD на α = ±χ0 + N(0, s), от плоскости r–θ к оси трубы на γ ~ N(0, s_L).
  Факторы Кернса: f_R = ⟨c_y²⟩, f_T = ⟨c_x²⟩, f_L = ⟨c_z²⟩ (s_L ≈ 21° даёт f_L ≈ 0.12).
• Пластинка — диск толщиной h в базисной плоскости (нормаль = c), растёт в своей плоскости, пока
  край не выйдет из зерна, не упрётся в другую пластинку или радиус не дойдёт до R_max.
• Несоответствие: ε* = ε_t·I + (ε_n − ε_t)·n⊗n (7.2 % по нормали, 4.6 % в плоскости).
• Поле — микроупругость Хачатуряна в 3D (изотропно, периодично, среднее напряжение 0).
• Выбор места — как в 2D (ca_hydride): вес ∝ exp(β·g)·c_H, g = σ:ε*_новой/ε_n, потолок только на
  ближнее поле, среднее напряжение в металле — без потолка; приложено окружное σ_xx.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

EPS_N, EPS_T = 0.0720, 0.0458


@dataclass
class P3:
    size_um: float = 64.0
    dx: float = 0.5
    grain_um: tuple = (4.5, 2.5, 6.0)       # TD, ND, L
    chi0: float = 30.0
    chi_s: float = 26.0
    chi_sL: float = 21.0                    # разброс наклона оси c к оси трубы (0 — ось c в плоскости r–θ)
    beta: float = 0.12
    sigma_cap: float = 90.0
    cap_local_only: bool = True
    capture_um: float = 35.0
    sigma_app: float = 0.0                  # окружное (x)
    h_um: float = 0.6
    R_max: float = 3.0
    R_min: float = 0.5
    frac: float = 0.0115
    E: float = 90e3
    nu: float = 0.34
    seed: int = 1


def make_grains3(p, rng):
    n = int(round(p.size_um / p.dx))
    L = p.size_um
    ng = int(L ** 3 / np.prod(p.grain_um))
    seeds = rng.uniform(0, L, size=(ng, 3))
    sc = 1.0 / np.array(p.grain_um)
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]) * L
    reps = (seeds[None] + shifts[:, None]).reshape(-1, 3)
    ids = np.tile(np.arange(ng), 27)
    tree = cKDTree(reps * sc)
    g = (np.arange(n) + 0.5) * p.dx
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    _, j = tree.query(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) * sc)
    grains = ids[j].reshape(n, n, n).astype(np.int32)
    sign = rng.choice([-1, 1], size=ng)
    alpha = np.radians(sign * p.chi0 + rng.normal(0, p.chi_s, ng))
    gam = np.radians(rng.normal(0, p.chi_sL, ng)) if p.chi_sL > 0 else np.zeros(ng)
    c = np.stack([np.sin(alpha) * np.cos(gam), np.cos(alpha) * np.cos(gam), np.sin(gam)], 1)   # (x=TD, y=ND, z=L)
    return grains, c


def eps_star(nv):
    """ε* для нормали nv (…, 3) → компоненты (xx, yy, zz, xy, xz, yz)."""
    d = EPS_N - EPS_T
    return (EPS_T + d * nv[..., 0] ** 2, EPS_T + d * nv[..., 1] ** 2, EPS_T + d * nv[..., 2] ** 2,
            d * nv[..., 0] * nv[..., 1], d * nv[..., 0] * nv[..., 2], d * nv[..., 1] * nv[..., 2])


class Elastic3:
    def __init__(self, n, dx, E, nu):
        self.n = n
        self.mu = E / (2 * (1 + nu)); self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        k = 2 * np.pi * np.fft.fftfreq(n, dx); kz = 2 * np.pi * np.fft.rfftfreq(n, dx)
        self.kx = k[:, None, None]; self.ky = k[None, :, None]; self.kz = kz[None, None, :]
        k2 = self.kx ** 2 + self.ky ** 2 + self.kz ** 2
        k2[0, 0, 0] = 1.0
        self.k2 = k2
        self.a = (self.lam + self.mu) / (self.lam + 2 * self.mu)

    def stress_of(self, fhat, e):
        """Напряжение от собственной деформации e (xx, yy, zz, xy, xz, yz), умноженной на поле доли f (fhat — его rfftn)."""
        lam, mu = self.lam, self.mu
        exx, eyy, ezz, exy, exz, eyz = e
        tr = exx + eyy + ezz
        s = np.array([[lam * tr + 2 * mu * exx, 2 * mu * exy, 2 * mu * exz],
                      [2 * mu * exy, lam * tr + 2 * mu * eyy, 2 * mu * eyz],
                      [2 * mu * exz, 2 * mu * eyz, lam * tr + 2 * mu * ezz]])
        kx, ky, kz = self.kx, self.ky, self.kz
        t = [s[i, 0] * kx + s[i, 1] * ky + s[i, 2] * kz for i in range(3)]
        kt = (kx * t[0] + ky * t[1] + kz * t[2]) / self.k2
        inv = 1.0 / (mu * self.k2)
        u = [inv * (t[0] - self.a * kx * kt), inv * (t[1] - self.a * ky * kt), inv * (t[2] - self.a * kz * kt)]
        A = [kx * u[0], ky * u[1], kz * u[2], 0.5 * (kx * u[1] + ky * u[0]), 0.5 * (kx * u[2] + kz * u[0]),
             0.5 * (ky * u[2] + kz * u[1])]
        D = [A[0] - exx, A[1] - eyy, A[2] - ezz, A[3] - exy, A[4] - exz, A[5] - eyz]
        trD = D[0] + D[1] + D[2]
        out = []
        for i, Di in enumerate(D):
            sh = (lam * trD + 2 * mu * Di) if i < 3 else 2 * mu * Di
            sh = sh * fhat
            sh[0, 0, 0] = 0.0
            out.append(np.fft.irfftn(sh, s=(self.n,) * 3))
        return out                         # xx, yy, zz, xy, xz, yz


def disc_fraction(n, dx, c0, nv, R, h, sub=2):
    """Доля диска (центр c0, нормаль nv, радиус R, толщина h) в клетках; возвращает индексы окна и доли."""
    r = R + h
    lo = np.floor((c0 - r) / dx).astype(int) - 1
    hi = np.floor((c0 + r) / dx).astype(int) + 2
    ax = [np.arange(lo[i], hi[i]) for i in range(3)]
    off = (np.arange(sub) + 0.5) / sub
    X = (ax[0][:, None, None, None, None, None] + off[None, None, None, :, None, None]) * dx - c0[0]
    Y = (ax[1][None, :, None, None, None, None] + off[None, None, None, None, :, None]) * dx - c0[1]
    Z = (ax[2][None, None, :, None, None, None] + off[None, None, None, None, None, :]) * dx - c0[2]
    q = X * nv[0] + Y * nv[1] + Z * nv[2]
    rho2 = X ** 2 + Y ** 2 + Z ** 2 - q ** 2
    fr = ((np.abs(q) <= h / 2) & (rho2 <= R * R)).mean(axis=(3, 4, 5))
    return [a % n for a in ax], fr


def grow_disc(ci, nv, grains, occ, p):
    """Диск растёт в своей плоскости; упираясь краем в границу зерна или соседа, сдвигает центр
    от препятствия (как пластинка в 2D растёт в обе стороны), пока зародыш внутри диска."""
    n = grains.shape[0]
    g0 = grains[ci]
    nuc = (np.array(ci) + 0.5) * p.dx
    c0 = nuc.copy()
    u = np.cross(nv, [0, 0, 1.0] if abs(nv[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(nv, u)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    dirs = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    R = 0.0
    step = p.dx * 0.5

    def blocked(c, r):
        idx = (np.floor((c + r * dirs) / p.dx).astype(int)) % n
        return (grains[idx[:, 0], idx[:, 1], idx[:, 2]] != g0) | occ[idx[:, 0], idx[:, 1], idx[:, 2]]

    for _ in range(int(4 * p.R_max / step)):
        if R + step > p.R_max:
            break
        b = blocked(c0, R + step)
        if not b.any():
            R += step
            continue
        if b.mean() > 0.4:
            break
        shift = -dirs[b].mean(0)
        shift /= max(np.linalg.norm(shift), 1e-9)
        c1 = c0 + 0.5 * step * shift
        if np.linalg.norm(c1 - nuc) > R + 0.5 * step or blocked(c1, R).any():
            break
        c0 = c1
        if not blocked(c0, R + 0.5 * step).any():
            R += 0.5 * step
    return c0, R


def run3d(p: P3, verbose=False):
    rng = np.random.default_rng(p.seed)
    grains, cax = make_grains3(p, rng)
    n = grains.shape[0]
    el = Elastic3(n, p.dx, p.E, p.nu)
    ncell = cax[grains]                                       # нормаль пластинки = ось c, по клеткам
    e_new = [a.astype(np.float32) for a in eps_star(ncell)]  # ε* возможной пластинки в клетке
    del ncell
    e_mean_xx = EPS_T + (EPS_N - EPS_T) / 3.0                 # средняя по направлениям часть ε*_xx
    g_app = (p.sigma_app * (e_new[0] - e_mean_xx) / EPS_N).astype(np.float32)
    S = [np.zeros((n, n, n), np.float32) for _ in range(6)]
    hyd = np.zeros((n, n, n), np.float32)
    occ = np.zeros((n, n, n), bool)
    tried = np.zeros((n, n, n), bool)
    cH = np.ones((n, n, n))
    if p.capture_um > 0:
        Gk = np.exp(-0.5 * p.capture_um ** 2 * (el.kx ** 2 + el.ky ** 2 + el.kz ** 2))
    plates = []
    target = p.frac * n ** 3
    w2 = np.array([1, 1, 1, 2, 2, 2], np.float32)              # сдвиговые компоненты — дважды
    while hyd.sum() < target:
        g = e_new[0] * S[0]
        for i in range(1, 6):
            g += w2[i] * e_new[i] * S[i]
        g /= EPS_N
        if p.cap_local_only:
            m = hyd < 0.2
            Sm = [float(a[m].mean()) for a in S]
            gm = sum(w2[i] * e_new[i] * Sm[i] for i in range(6)) / EPS_N
            g = np.clip(g - gm, -p.sigma_cap, p.sigma_cap) + gm
        else:
            g = np.clip(g, -p.sigma_cap, p.sigma_cap)
        expo = p.beta * (g + g_app)
        allowed = ~occ & ~tried
        if p.capture_um > 0:
            allowed &= cH > 1e-3
            expo = expo + np.log(np.clip(cH, 1e-12, None))
        if not allowed.any():
            break
        expo = np.where(allowed, expo, -np.inf)
        w = np.exp(expo - expo[allowed].max()).ravel()
        ok = False
        for _ in range(40):                                 # отказы — без пересчёта поля: пачка кандидатов
            cdf = np.cumsum(w, dtype=np.float64)
            if cdf[-1] <= 0:
                break
            ks = np.searchsorted(cdf, rng.random(64) * cdf[-1])
            for k in ks:
                if w[k] == 0.0:
                    continue
                ci = np.unravel_index(k, (n, n, n))
                nv = cax[grains[ci]]
                c0, R = grow_disc(ci, nv, grains, occ, p)
                if R >= p.R_min:
                    ok = True
                    break
                w[k] = 0.0
                tried[ci] = True
            if ok:
                break
        if not ok:
            break
        ax, fr = disc_fraction(n, p.dx, c0, nv, R, p.h_um)
        sel = np.ix_(*ax)
        f = np.zeros((n, n, n))
        f[sel] = fr
        new = np.clip(hyd[sel] + fr, 0, 1) - hyd[sel]
        hyd[sel] += new
        fhat = np.fft.rfftn(f)
        for i, a in enumerate(el.stress_of(fhat, eps_star(nv))):
            S[i] += a
        if p.capture_um > 0:
            cH -= np.fft.irfftn(fhat * Gk, s=(n,) * 3) / p.frac
        occ[sel] |= fr > 0.2
        # зазор в клетку вокруг пластинки
        o = occ[sel]
        for d in range(3):
            o |= np.roll(occ[sel], 1, d) | np.roll(occ[sel], -1, d)
        occ[sel] = o
        tried[:] = False
        plates.append(dict(c=c0, n=nv.copy(), R=float(R), grain=int(grains[ci])))
        if verbose and len(plates) % 50 == 0:
            print(len(plates), f"{hyd.mean() * 100:.2f} %", flush=True)
    return dict(params=p, plates=plates, hyd=hyd, grains=grains, cax=cax, S=S)


def simon_w(dev):
    return np.where(dev <= 40, 0.0, np.where(dev < 65, 0.5, 1.0))


def section_rhf(res, n_sec=16):
    """RHF по следам пластинок в сечениях ⊥ оси трубы (как на шлифе r–θ): вес — длина хорды."""
    p = res["params"]
    zs = (np.arange(n_sec) + 0.5) / n_sec * p.size_um
    num = den = 0.0
    for q in res["plates"]:
        nv, c0, R = q["n"], q["c"], q["R"]
        d = np.cross(nv, [0, 0, 1.0])                         # след плоскости пластинки в сечении z = const
        if np.linalg.norm(d) < 1e-6:
            continue                                          # пластинка параллельна сечению
        d /= np.linalg.norm(d)
        dev = np.degrees(np.arccos(min(1.0, abs(d[0]))))       # угол следа к TD
        sin_t = np.sqrt(max(1e-12, 1 - nv[2] ** 2))            # наклон плоскости к оси z
        for z in zs:
            dz = (z - c0[2] + p.size_um / 2) % p.size_um - p.size_um / 2
            rr = abs(dz) / sin_t                                # расстояние от центра диска до линии сечения в его плоскости
            if rr < R:
                L = 2 * np.sqrt(R * R - rr * rr)
                num += L * simon_w(dev); den += L
    return num / den if den else np.nan


def section_images(res, n_sec=8):
    """Сечения ⊥ z: массивы (ND строки, TD столбцы) для обработки как снимков."""
    n = res["hyd"].shape[0]
    return [res["hyd"][:, :, int((k + 0.5) * n / n_sec)].T for k in range(n_sec)]
