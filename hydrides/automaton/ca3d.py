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
import os
import scipy.fft as sfft
from scipy.spatial import cKDTree

EPS_N, EPS_T = 0.0720, 0.0458
WORKERS = int(os.environ.get("CA3D_WORKERS", "1"))      # потоки FFT по всему кубу (при пуле процессов — 1)


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
    # кристаллография микрогидрида (shape="needle"): вытянутая пластинка в базисной плоскости,
    # длинная ось вдоль одного из трёх ⟨11-20⟩ зерна (Carpenter; Perovic; Patel 2021)
    shape: str = "disc"                     # "disc" — круглый диск (пилот); "needle" — пластинка вдоль ⟨11-20⟩
    prism: str = "10-10"                    # призматическая текстура: ⟨10-10⟩ ∥ оси трубы (CWSR), "11-20" (RX), "random"
    prism_s: float = 10.0                   # разброс поворота вокруг оси c, град
    A_max: float = 3.0                      # наибольшая полудлина вдоль ⟨11-20⟩, мкм (длина до 6 мкм, как в 2D)
    B_max: float = 0.5                      # наибольшая полуширина поперёк, мкм
    A_min: float = 0.5                      # меньше — зародыш не растёт


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


def basal_axes(c, p):
    """Направление a1 = ⟨11-20⟩ в базисной плоскости каждого зерна. Отсчёт — от проекции оси трубы
    на базисную плоскость: при ⟨10-10⟩ ∥ оси (CWSR) ⟨11-20⟩ повёрнуты на 30°, 90°, 150°; при ⟨11-20⟩ ∥ оси
    (RX) — на 0°, 60°, 120°. Свой генератор случайных чисел, чтобы не менять зёрна дисковых прогонов."""
    rng = np.random.default_rng(p.seed + 7919)
    ng = len(c)
    ref = np.array([0, 0, 1.0])[None] - c[:, 2:3] * c
    bad = np.linalg.norm(ref, axis=1) < 1e-6
    ref[bad] = np.array([1.0, 0, 0])[None] - c[bad, 0:1] * c[bad]
    ref /= np.linalg.norm(ref, axis=1)[:, None]
    w = np.cross(c, ref)
    if p.prism == "random":
        phi = rng.uniform(0, np.pi / 3, ng)
    else:
        phi = np.radians((30.0 if p.prism == "10-10" else 0.0) + rng.normal(0, p.prism_s, ng))
    return np.cos(phi)[:, None] * ref + np.sin(phi)[:, None] * w


def needle_dirs(c, a1):
    """Три оси ⟨11-20⟩ зерна (через 120° в базисной плоскости)."""
    w = np.cross(c, a1)
    return [np.cos(t) * a1 + np.sin(t) * w for t in np.radians((0.0, 120.0, 240.0))]


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
        self.k32 = tuple(a.astype(np.float32) for a in (self.kx, self.ky, self.kz, self.k2))

    def stress_of(self, fhat, e, single=False):
        """Напряжение от собственной деформации e (xx, yy, zz, xy, xz, yz), умноженной на поле доли f (fhat — его rfftn)."""
        lam, mu = self.lam, self.mu
        e = [float(v) for v in e]
        exx, eyy, ezz, exy, exz, eyz = e
        tr = exx + eyy + ezz
        s = np.array([[lam * tr + 2 * mu * exx, 2 * mu * exy, 2 * mu * exz],
                      [2 * mu * exy, lam * tr + 2 * mu * eyy, 2 * mu * eyz],
                      [2 * mu * exz, 2 * mu * eyz, lam * tr + 2 * mu * ezz]])
        kx, ky, kz, k2 = self.k32 if single else (self.kx, self.ky, self.kz, self.k2)
        s = s.tolist()
        t = [s[i][0] * kx + s[i][1] * ky + s[i][2] * kz for i in range(3)]
        kt = (kx * t[0] + ky * t[1] + kz * t[2]) / k2
        inv = 1.0 / (mu * k2)
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
            out.append(sfft.irfftn(sh, s=(self.n,) * 3))
        return out                         # xx, yy, zz, xy, xz, yz


def stress_of_field(el, E):
    """Напряжение от поля собственной деформации E = (xx, yy, zz, xy, xz, yz) по всему кубу (периодично,
    среднее 0). Одинарная точность; выходные компоненты считаются по одной."""
    lam, mu, n = el.lam, el.mu, el.n
    kx, ky, kz, k2 = el.k32
    Eh = [sfft.rfftn(a.astype(np.float32, copy=False), workers=WORKERS) for a in E]
    tr = Eh[0] + Eh[1] + Eh[2]
    # t = (C:Ê)·k
    t0 = (lam * tr + 2 * mu * Eh[0]) * kx + 2 * mu * (Eh[3] * ky + Eh[4] * kz)
    t1 = (lam * tr + 2 * mu * Eh[1]) * ky + 2 * mu * (Eh[3] * kx + Eh[5] * kz)
    t2 = (lam * tr + 2 * mu * Eh[2]) * kz + 2 * mu * (Eh[4] * kx + Eh[5] * ky)
    kt = (kx * t0 + ky * t1 + kz * t2) / k2
    inv = 1.0 / (mu * k2)
    u0 = inv * (t0 - el.a * kx * kt); del t0
    u1 = inv * (t1 - el.a * ky * kt); del t1
    u2 = inv * (t2 - el.a * kz * kt); del t2, kt
    trD = kx * u0 + ky * u1 + kz * u2 - tr
    del tr
    out = []
    for i, (A, Ei) in enumerate(((kx * u0, Eh[0]), (ky * u1, Eh[1]), (kz * u2, Eh[2]))):
        sh = lam * trD + 2 * mu * (A - Ei)
        sh[0, 0, 0] = 0.0
        out.append(sfft.irfftn(sh, s=(n,) * 3, workers=WORKERS).astype(np.float32, copy=False))
    for A, Ei in ((0.5 * (kx * u1 + ky * u0), Eh[3]), (0.5 * (kx * u2 + kz * u0), Eh[4]), (0.5 * (ky * u2 + kz * u1), Eh[5])):
        sh = 2 * mu * (A - Ei)
        sh[0, 0, 0] = 0.0
        out.append(sfft.irfftn(sh, s=(n,) * 3, workers=WORKERS).astype(np.float32, copy=False))
    return out


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


def ellipse_fraction(n, dx, c0, nv, u, A, B, h, sub=2):
    """Доля пластинки-эллипса (центр c0, нормаль nv, длинная ось u, полуоси A вдоль u и B поперёк, толщина h)."""
    w = np.cross(nv, u)
    r = max(A, B) + h
    lo = np.floor((c0 - r) / dx).astype(int) - 1
    hi = np.floor((c0 + r) / dx).astype(int) + 2
    ax = [np.arange(lo[i], hi[i]) for i in range(3)]
    off = (np.arange(sub) + 0.5) / sub
    X = (ax[0][:, None, None, None, None, None] + off[None, None, None, :, None, None]) * dx - c0[0]
    Y = (ax[1][None, :, None, None, None, None] + off[None, None, None, None, :, None]) * dx - c0[1]
    Z = (ax[2][None, None, :, None, None, None] + off[None, None, None, None, None, :]) * dx - c0[2]
    q = X * nv[0] + Y * nv[1] + Z * nv[2]
    s = X * u[0] + Y * u[1] + Z * u[2]
    t = X * w[0] + Y * w[1] + Z * w[2]
    fr = ((np.abs(q) <= h / 2) & ((s / A) ** 2 + (t / B) ** 2 <= 1)).mean(axis=(3, 4, 5))
    return [a % n for a in ax], fr


def plate_fraction(n, dx, c0, q, h):
    """Доля пластинки q (диск или эллипс) в клетках вокруг центра c0."""
    if "u" in q:
        return ellipse_fraction(n, dx, c0, q["n"], q["u"], q["A"], q["B"], h)
    return disc_fraction(n, dx, c0, q["n"], q["R"], h)


def grow_needle(ci, nv, u, grains, occ, p):
    """Пластинка растёт вдоль ⟨11-20⟩ (u) в обе стороны, пока то же зерно и свободно (как в 2D), затем
    в ширину поперёк (w = n × u) до B_max. Возвращает (центр, A, B)."""
    n = grains.shape[0]
    g0 = grains[ci]
    nuc = (np.array(ci) + 0.5) * p.dx
    w = np.cross(nv, u)
    step = p.dx * 0.5

    def free(pts):
        i = np.floor(pts / p.dx).astype(int) % n
        return bool(np.all((grains[i[..., 0], i[..., 1], i[..., 2]] == g0) & ~occ[i[..., 0], i[..., 1], i[..., 2]]))

    def reach(c, e, lim, side_pts):
        d = 0.0
        while d + step <= lim and free(c[None] + side_pts + (d + step) * e[None]):
            d += step
        return d

    zero = np.zeros((1, 3))
    a, b = reach(nuc, u, p.A_max, zero), reach(nuc, -u, p.A_max, zero)
    A = 0.5 * (a + b)
    c0 = nuc + 0.5 * (a - b) * u
    side = np.array([-0.5 * A, 0.0, 0.5 * A])[:, None] * u[None]       # ширину проверяем по трём линиям
    a, b = reach(c0, w, p.B_max, side), reach(c0, -w, p.B_max, side)
    B = max(0.5 * (a + b), 0.25)
    c0 = c0 + 0.5 * (a - b) * w
    return c0, A, B


def plate_chord(c, nv, u, A, B, zc, L):
    """След пластинки (эллипс A × B по осям u, n × u; диск — A = B = R) в сечении z = zc периодического куба L:
    (середина следа (x, y), направление следа d, полудлина) или None."""
    d = np.cross(nv, [0, 0, 1.0])
    nd = np.linalg.norm(d)
    if nd < 1e-6:
        return None
    d = d / nd
    dz = (zc - c[2] + L / 2) % L - L / 2
    wv = np.array([0, 0, 1.0]) - nv[2] * nv
    wv /= np.linalg.norm(wv)
    p0 = wv * (dz / wv[2])                                    # относительно центра пластинки
    w = np.cross(nv, u)
    s0, su, r0, rw = p0 @ u, d @ u, p0 @ w, d @ w
    qa = su * su / (A * A) + rw * rw / (B * B)
    qb = 2 * (s0 * su / (A * A) + r0 * rw / (B * B))
    qc = s0 * s0 / (A * A) + r0 * r0 / (B * B) - 1
    disc = qb * qb - 4 * qa * qc
    if disc <= 0:
        return None
    tm, half = -qb / (2 * qa), np.sqrt(disc) / (2 * qa)
    mid = c + p0 + tm * d
    return mid[:2], d, half


def plate_axes(q):
    """(u, A, B) пластинки: у диска u — любое направление в его плоскости, A = B = R."""
    if "u" in q:
        return q["u"], q["A"], q["B"]
    nv = q["n"]
    u = np.cross(nv, [0, 0, 1.0] if abs(nv[2]) < 0.9 else [1.0, 0, 0])
    return u / np.linalg.norm(u), q["R"], q["R"]


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
                q = make_plate(ci, nv)
                if q is not None:
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
        u, A, B = plate_axes(q)
        for z in zs:
            tr = plate_chord(q["c"], q["n"], u, A, B, z, p.size_um)
            if tr is not None:
                dev = np.degrees(np.arccos(min(1.0, abs(tr[1][0]))))   # угол следа к TD
                num += 2 * tr[2] * simon_w(dev); den += 2 * tr[2]
    return num / den if den else np.nan


def section_images(res, n_sec=8):
    """Сечения ⊥ z: массивы (ND строки, TD столбцы) для обработки как снимков."""
    n = res["hyd"].shape[0]
    return [res["hyd"][:, :, int((k + 0.5) * n / n_sec)].T for k in range(n_sec)]


# ------------------------------------------------------------------ быстрый вариант для больших объёмов
def _contract(e, S):
    g = e[0] * S[0] + e[1] * S[1] + e[2] * S[2]
    g += 2 * (e[3] * S[3] + e[4] * S[4] + e[5] * S[5])
    return g / EPS_N


_MWIN = {}


def window_offset_tensor(win, dx, E, nu):
    """Решение в периодическом окне имеет нулевое среднее по окну, а настоящее поле малого включения
    имеет среднее по окну (V/W³)·M:ε* (лемма Танаки–Мори для области-окна). M (6×6, МПа) — один раз,
    по включению 2³ клеток в периодическом кубе 4·win."""
    key = (win, E, nu)
    if key not in _MWIN:
        nB = 4 * win
        el = Elastic3(nB, dx, E, nu)
        c = nB // 2
        f = np.zeros((nB,) * 3); f[c - 1:c + 1, c - 1:c + 1, c - 1:c + 1] = 1.0
        fh = np.fft.rfftn(f)
        sel = tuple(slice(c - win // 2, c + win // 2) for _ in range(3))
        M = np.zeros((6, 6))
        for j in range(6):
            e = [0.0] * 6; e[j] = 1.0
            S = el.stress_of(fh, e)
            M[:, j] = [S[i][sel].mean() / (8.0 * (1.0 / win ** 3 - 1.0 / nB ** 3)) for i in range(6)]
        _MWIN[key] = M
    return _MWIN[key]


def run3d_fast(p: P3, win=32, resync=100, B=8, tol_mpa=1.0, verbose=False):
    """Тот же автомат для больших объёмов. Раз в resync пластинок поле считается точно (Фурье по всему
    кубу от накопленной собственной деформации), обеднение водородом вычитается, веса пересчитываются.
    Между пересчётами поле новой пластинки берётся из Фурье в окне win³ клеток вокруг неё (16 мкм) с
    поправкой на среднее по окну (window_offset_tensor); ошибка — единицы МПа, только у последних
    resync пластинок. Выбор места — по суммам весов
    в блоках B³ (обновляются только блоки окна). Среднее напряжение в металле (для потолка) — на каждом
    шаге по клеткам гидрида (сумма поля по кубу — ноль); веса по всему кубу пересчитываются и тогда,
    когда оно сдвинулось больше чем на tol_mpa."""
    rng = np.random.default_rng(p.seed)
    grains, cax = make_grains3(p, rng)
    a1 = basal_axes(cax, p) if p.shape == "needle" else None
    n = grains.shape[0]
    assert n % B == 0 and win <= n
    nb = n // B
    el = Elastic3(win, p.dx, p.E, p.nu)
    elF = Elastic3(n, p.dx, p.E, p.nu)
    Mw = window_offset_tensor(win, p.dx, p.E, p.nu)
    ncell = cax[grains]
    e_new = [a.astype(np.float32) for a in eps_star(ncell)]
    del ncell
    e_mean_xx = EPS_T + (EPS_N - EPS_T) / 3.0
    g_app = (p.sigma_app * (e_new[0] - e_mean_xx) / EPS_N).astype(np.float32)
    S = [np.zeros((n, n, n), np.float32) for _ in range(6)]
    Eacc = [np.zeros((n, n, n), np.float32) for _ in range(6)]   # собственная деформация всех пластинок
    graw = np.zeros((n, n, n), np.float32)                   # σ:ε*/ε_n без потолка
    hyd = np.zeros((n, n, n), np.float32)
    occ = np.zeros((n, n, n), bool)
    cap = p.sigma_cap
    capture = p.capture_um > 0
    if capture:
        cH = np.ones((n, n, n), np.float32)
        pend = np.zeros((n, n, n), np.float32)               # доля гидрида, ещё не вычтенная из cH
        k = 2 * np.pi * np.fft.fftfreq(n, p.dx); kz = 2 * np.pi * np.fft.rfftfreq(n, p.dx)
        Gk = np.exp(-0.5 * p.capture_um ** 2 * (k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)).astype(np.float32)
    w = np.zeros((n, n, n))
    w6 = w.reshape(nb, B, nb, B, nb, B)
    ar = np.arange(B)
    st = dict(Sm=np.zeros(6), E0=0.0, T=np.zeros(6))      # T — сумма поля по кубу (после точного пересчёта 0)
    hcells = np.zeros(0, np.int64)                            # клетки гидрида (hyd ≥ 0.2), плоские индексы
    Sf = [a.reshape(-1) for a in S]

    def expo_of(sl):
        e = [a[sl] for a in e_new]
        g = graw[sl]
        if p.cap_local_only:
            gm = _contract(e, st["Sm"])
            g = np.clip(g - gm, -cap, cap) + gm
        else:
            g = np.clip(g, -cap, cap)
        ex = p.beta * (g + g_app[sl])
        ok = ~occ[sl]
        if capture:
            c = cH[sl]
            ok &= c > 1e-3
            ex = ex + np.log(np.clip(c, 1e-12, None))
        return np.where(ok, ex, -np.inf)

    def weights(sl):
        return np.exp(expo_of(sl) - st["E0"])

    def exact_field():
        new = stress_of_field(elF, Eacc)
        for i in range(6):
            S[i][:] = new[i]
        del new
        st["T"] = np.zeros(6)
        for x0 in range(0, n, 16):
            sl = np.s_[x0:x0 + 16]
            graw[sl] = _contract([a[sl] for a in e_new], [a[sl] for a in S])

    def resync_all(full=True):
        if full and plates:
            exact_field()
        if full and capture and pend.any():
            cH[:] -= (sfft.irfftn(sfft.rfftn(pend, workers=WORKERS) * Gk, s=(n,) * 3, workers=WORKERS) / p.frac).astype(np.float32)
            pend[:] = 0
        st["Sm"] = Sm_now()
        mx = -np.inf
        for x0 in range(0, n, 16):
            sl = np.s_[x0:x0 + 16]
            w[sl] = expo_of(sl)
            mx = max(mx, float(w[sl].max()))
        if not np.isfinite(mx):                               # мест не осталось
            w[:] = 0.0
            return np.zeros((nb,) * 3)
        st["E0"] = mx
        for x0 in range(0, n, 16):
            sl = np.s_[x0:x0 + 16]
            np.exp(w[sl] - mx, out=w[sl])
        return w6.sum(axis=(1, 3, 5))

    def update_blocks(bsum, cells):                         # прямоугольное окно: ix по осям
        bx, by, bz = (np.unique(c // B) for c in cells)
        bsum[np.ix_(bx, by, bz)] = w6[np.ix_(bx, ar, by, ar, bz, ar)].sum(axis=(1, 3, 5))

    def Sm_now():
        nm = n ** 3 - hcells.size
        return np.array([(st["T"][i] - float(a[hcells].sum(dtype=np.float64))) / nm for i, a in enumerate(Sf)])

    def gcap_mean(pts):
        """Средняя выгода (с потолком на ближнее поле) вдоль точек pts, мкм."""
        i = np.floor(pts / p.dx).astype(int) % n
        t = (i[:, 0], i[:, 1], i[:, 2])
        g = graw[t]
        if p.cap_local_only:
            gm = _contract([a[t] for a in e_new], st["Sm"])
            g = np.clip(g - gm, -cap, cap) + gm
        else:
            g = np.clip(g, -cap, cap)
        return float(g.mean())

    def make_plate(ci, nv):
        """Пластинка из зародыша ci или None. Игла: из трёх ⟨11-20⟩ выбирается вариант с вероятностью
        ∝ exp(β·ḡ), ḡ — средняя выгода вдоль её длины (взаимодействие формы с полем соседей)."""
        if p.shape != "needle":
            c0, R = grow_disc(ci, nv, grains, occ, p)
            return dict(c=c0, n=nv.copy(), R=float(R)) if R >= p.R_min else None
        opts = []
        for u in needle_dirs(nv, a1[grains[ci]]):
            c0, Aq, Bq = grow_needle(ci, nv, u, grains, occ, p)
            if Aq >= p.A_min:
                sg = np.linspace(-Aq, Aq, max(3, int(2 * Aq / p.dx) + 1))
                opts.append((gcap_mean(c0[None] + sg[:, None] * u[None]), c0, u, Aq, Bq))
        if not opts:
            return None
        sc = p.beta * np.array([o[0] for o in opts])
        pr = np.exp(sc - sc.max())
        _, c0, u, Aq, Bq = opts[rng.choice(len(opts), p=pr / pr.sum())]
        return dict(c=c0, n=nv.copy(), u=u, A=float(Aq), B=float(Bq), R=float(np.sqrt(Aq * Bq)))

    plates = []
    bsum = resync_all()
    target = p.frac * n ** 3
    total = 0.0
    while total < target:
        ok = False
        tried = []
        for _ in range(40):
            cdf = np.cumsum(np.clip(bsum, 0, None).ravel())
            if cdf[-1] <= 0:
                break
            for bk in np.searchsorted(cdf, rng.random(64) * cdf[-1]):
                bi, bj, bl = np.unravel_index(min(bk, nb ** 3 - 1), (nb, nb, nb))
                wb = w6[bi, :, bj, :, bl, :]
                cs = np.cumsum(wb.ravel())
                if cs[-1] <= 0:
                    continue
                a, b, c = np.unravel_index(min(np.searchsorted(cs, rng.random() * cs[-1]), B ** 3 - 1), (B, B, B))
                ci = (bi * B + a, bj * B + b, bl * B + c)
                nv = cax[grains[ci]]
                q = make_plate(ci, nv)
                if q is not None:
                    ok = True
                    break
                bsum[bi, bj, bl] -= w[ci]
                w[ci] = 0.0
                tried.append(ci)
            if ok:
                break
        if not ok:
            break
        c0 = q["c"]
        o = np.floor(c0 / p.dx).astype(int) - win // 2
        ix = [(o[d] + np.arange(win)) % n for d in range(3)]
        sel = np.ix_(*ix)
        ax, fr = plate_fraction(win, p.dx, c0 - o * p.dx, q, p.h_um)
        f = np.zeros((win,) * 3, np.float32)
        f[np.ix_(*ax)] = fr
        es = eps_star(nv)
        corr = float(fr.sum()) / win ** 3 * (Mw @ np.array(es))
        for i, a in enumerate(el.stress_of(sfft.rfftn(f), es, single=True)):
            S[i][sel] += a + np.float32(corr[i])
        st["T"] = st["T"] + corr * win ** 3
        graw[sel] = _contract([a[sel] for a in e_new], [a[sel] for a in S])
        dsel = np.ix_(*[(o[d] + ax[d]) % n for d in range(3)])
        was = hyd[dsel] >= 0.2
        new = np.clip(hyd[dsel] + fr, 0, 1) - hyd[dsel]
        hyd[dsel] += new
        gi = np.ravel_multi_index(np.meshgrid(*[(o[d] + ax[d]) % n for d in range(3)], indexing="ij"), (n,) * 3)
        hcells = np.concatenate([hcells, gi[(hyd[dsel] >= 0.2) & ~was]])
        total += float(new.sum())
        if capture:
            pend[dsel] += fr.astype(np.float32)
        for i in range(6):
            Eacc[i][dsel] += (fr * es[i]).astype(np.float32)
        oc = occ[dsel] | (fr > 0.2)
        o2 = oc.copy()
        for d in range(3):
            o2 |= np.roll(oc, 1, d) | np.roll(oc, -1, d)
        occ[dsel] = o2
        plates.append(dict(q, c=c0 % (n * p.dx), grain=int(grains[ci])))
        if len(plates) % resync == 0:
            bsum = resync_all()
        elif p.cap_local_only and np.abs(Sm_now() - st["Sm"]).sum() > tol_mpa:
            bsum = resync_all(full=False)
        else:
            w[sel] = weights(sel)
            update_blocks(bsum, ix)
            if tried:
                t = tuple(np.array(tried).T)
                w[t] = weights(t)
                for bi, bj, bl in set(zip(*(c // B for c in t))):
                    bsum[bi, bj, bl] = w6[bi, :, bj, :, bl, :].sum()
        if verbose and len(plates) % 500 == 0:
            print(len(plates), f"{total / n ** 3 * 100:.2f} %", flush=True)
    return dict(params=p, plates=plates, hyd=hyd, grains=grains, cax=cax, S=S)
