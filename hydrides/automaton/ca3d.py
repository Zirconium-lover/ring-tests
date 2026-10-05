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
from scipy import ndimage
from scipy.spatial import cKDTree

EPS_N, EPS_T = 0.0720, 0.0458
C_OVER_A = 1.593                                         # α-Zr
HABIT_DEG = float(np.degrees(np.arctan(2.0 * C_OVER_A / (7.0 * np.sqrt(3.0)))))   # угол (10-17) к (0001): 14.7°
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
    shape: str = "disc"                     # "disc" — круглый диск (пилот); "needle" — пластинка вдоль ⟨11-20⟩;
                                            # "segment" — отрезок макрогидрида в зерне на плоскости {10-17}
    prism: str = "10-10"                    # призматическая текстура: ⟨10-10⟩ ∥ оси трубы (CWSR), "11-20" (RX), "random"
    prism_s: float = 10.0                   # разброс поворота вокруг оси c, град
    A_max: float = 3.0                      # наибольшая полудлина вдоль ⟨11-20⟩, мкм (длина до 6 мкм, как в 2D)
    B_max: float = 0.5                      # наибольшая полуширина поперёк, мкм
    A_min: float = 0.5                      # меньше — зародыш не растёт
    R_seg: float = 4.0                      # отрезок {10-17}: наибольшее удаление от зародыша в плоскости, мкм
    S_min: float = 0.8                      # наименьшая площадь отрезка, мкм² (≈ диск R = 0.5)
    gb: bool = False                        # межзёренные отрезки: на грани двух зёрен, несоответствие 7.2 % по нормали к грани
    w_gb: float = 1.0                       # во сколько раз зарождение на грани выгоднее, чем в теле зерна (1 — так же)
    gb_eps: str = "facet"                   # несоответствие межзёренного: "facet" — 7.2 % по нормали к грани ({111}δ ∥ границе);
                                            # "grain" — в осях того из двух зёрен, чья ось c ближе к нормали грани (Fang)
    cross_deg: float = 0.0                  # переход отрезка в соседнее зерно, если у того есть вариант {10-17}
                                            # с нормалью ближе этого угла (0 — не переходит)


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


def habit_normals(c, a1, tilt_deg=HABIT_DEG):
    """Шесть вариантов габитуса {10-17} зерна: ось c, повёрнутая на ±14.7° вокруг каждой из трёх осей
    ⟨11-20⟩ (стопка игл вдоль u со ступеньками в одну или другую сторону). Список (нормаль, u)."""
    t = np.radians(tilt_deg)
    out = []
    for u in needle_dirs(c, a1):
        for sg in (1.0, -1.0):
            out.append((np.cos(t) * c + sg * np.sin(t) * np.cross(u, c), u))
    return out


def grow_segment(ci, nh, grains, occ, p, sub=2, allowed=None, q0=0.0, cross=None):
    """Отрезок макрогидрида: часть плоскости (нормаль nh, на расстоянии q0 от центра клетки ci) в
    разрешённых зёрнах (по умолчанию — зерно зародыша), свободная и связная с зародышем, не дальше R_seg;
    толщина h. Форма — сечение зерна плоскостью. cross(g) → True, если отрезок может перейти в зерно g.
    Возвращает (lo — угол окна в клетках, fr — доли в окне, центр, площадь мкм², зёрна окна) или None."""
    n = grains.shape[0]
    g0 = grains[ci]
    m = int(np.ceil((p.R_seg + p.h_um) / p.dx)) + 1
    lo = np.array(ci) - m
    rng_ = np.arange(-m, m + 1)
    idx = np.ix_(*[(ci[d] + rng_) % n for d in range(3)])
    X = rng_[:, None, None] * p.dx
    Y = rng_[None, :, None] * p.dx
    Z = rng_[None, None, :] * p.dx
    q = X * nh[0] + Y * nh[1] + Z * nh[2] - q0
    rho2 = X * X + Y * Y + Z * Z - (q + q0) ** 2
    halfw = p.h_um / 2 + 0.5 * p.dx * np.abs(nh).sum()
    gw = grains[idx]
    slab = (np.abs(q) <= halfw) & (rho2 <= p.R_seg ** 2) & ~occ[idx]
    ok_g = np.array([g0] if allowed is None else allowed)
    if cross is not None:
        ok_g = np.array([g for g in np.unique(gw[slab]) if g == g0 or cross(g)])
    cand = slab & np.isin(gw, ok_g)
    lab, _ = ndimage.label(cand, structure=np.ones((3, 3, 3), bool))
    k = lab[m, m, m]
    if k == 0:
        return None
    comp = lab == k
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * p.dx
    fr = np.zeros(q.shape)
    for ox in off:
        for oy in off:
            for oz in off:
                fr += np.abs(q + ox * nh[0] + oy * nh[1] + oz * nh[2]) <= p.h_um / 2
    fr = fr / sub ** 3 * comp
    tot = fr.sum()
    if tot <= 0:
        return None
    cen = (np.array(ci) + 0.5) * p.dx + np.array([(fr * X).sum(), (fr * Y).sum(), (fr * Z).sum()]) / tot
    return lo, fr, cen, tot * p.dx ** 3 / p.h_um, gw


def grain_facets(grains, dx, min_links=4):
    """Грани между зёрнами: пары (a < b), центр, нормаль (по главным осям точек на границе), площадь и
    клетки по обе стороны грани (для зарождения). Учитывается периодичность."""
    n = grains.shape[0]
    L = n * dx
    ng = int(grains.max()) + 1
    keys, pts, cells, dirs = [], [], [], []
    for d in range(3):
        g2 = np.roll(grains, -1, axis=d)
        i = np.nonzero(grains != g2)
        a, b = grains[i], g2[i]
        keys.append(np.minimum(a, b).astype(np.int64) * ng + np.maximum(a, b))
        P = (np.stack(i, 1) + 0.5) * dx
        P[:, d] += 0.5 * dx
        pts.append(P % L)
        j = list(i); j[d] = (j[d] + 1) % n
        cells.append(np.stack([np.ravel_multi_index(i, (n,) * 3), np.ravel_multi_index(tuple(j), (n,) * 3)], 1))
        dirs.append(np.full(len(a), d))
    keys, pts, cells, dirs = map(np.concatenate, (keys, pts, cells, dirs))
    o = np.argsort(keys, kind="stable")
    keys, pts, cells, dirs = keys[o], pts[o], cells[o], dirs[o]
    uk, start, cnt = np.unique(keys, return_index=True, return_counts=True)
    ref = np.repeat(pts[start], cnt, axis=0)
    dp = (pts - ref + L / 2) % L - L / 2
    s1 = np.add.reduceat(dp, start, axis=0)
    s2 = np.add.reduceat(dp[:, :, None] * dp[:, None, :], start, axis=0)
    mean = s1 / cnt[:, None]
    cov = s2 / cnt[:, None, None] - mean[:, :, None] * mean[:, None, :]
    ev, vec = np.linalg.eigh(cov)
    nrm = vec[:, :, 0]
    nd = np.stack([np.add.reduceat((dirs == d).astype(float), start) for d in range(3)], 1)
    area = dx * dx * np.sqrt((nd ** 2).sum(1))
    keep = cnt >= min_links
    cen = (pts[start] + mean) % L
    ptr = np.concatenate([[0], np.cumsum(2 * cnt)])
    allc = cells.reshape(-1)
    sel = np.nonzero(keep)[0]
    fc = [np.unique(allc[ptr[k]:ptr[k + 1]]) for k in sel]
    fptr = np.concatenate([[0], np.cumsum([len(c) for c in fc])])
    return dict(a=(uk[sel] // ng).astype(int), b=(uk[sel] % ng).astype(int), cen=cen[sel], n=nrm[sel], area=area[sel],
                cells=np.concatenate(fc), ptr=fptr)


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
        if "area" in q:                                       # отрезок: Σ хорд по сечениям = площадь·sin
            d = np.cross(q["n"], [0, 0, 1.0]); s = np.linalg.norm(d)
            if s > 1e-6:
                dev = np.degrees(np.arccos(min(1.0, abs(d[0]) / s)))
                wgt = q["area"] * s * n_sec / p.size_um
                num += wgt * simon_w(dev); den += wgt
            continue
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
    a1 = basal_axes(cax, p) if p.shape in ("needle", "segment") else None
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
    occ_f = occ.reshape(-1)
    W2 = np.array([1.0, 1, 1, 2, 2, 2])
    if p.shape == "segment" and p.cross_deg > 0:              # нормали {10-17} всех зёрен — для перехода через границу
        HN = np.array([[h for h, _ in habit_normals(cax[g], a1[g])] for g in range(len(cax))])
        cos_x = np.cos(np.radians(p.cross_deg))
    if p.gb:                                                   # грани зёрен — места межзёренного зарождения
        F = grain_facets(grains, p.dx)
        nfac = len(F["a"])
        if p.gb_eps == "grain":                               # ось c соседнего зерна, ближайшая к нормали грани
            ca, cb = cax[F["a"]], cax[F["b"]]
            pick_a = np.abs((ca * F["n"]).sum(1)) >= np.abs((cb * F["n"]).sum(1))
            F_ax = np.where(pick_a[:, None], ca, cb)
        else:
            F_ax = F["n"]
        F_eps = np.stack(eps_star(F_ax), 1)
        F_gapp = p.sigma_app * (F_eps[:, 0] - e_mean_xx) / EPS_N
        F_tree = cKDTree(F["cen"], boxsize=n * p.dx * (1 + 1e-9))
        F_cnt = np.diff(F["ptr"])
        wf = np.zeros(nfac)

    def facet_weights(ids):
        """Веса зарождения по клеткам граней ids (несоответствие 7.2 % по нормали к грани) × w_gb:
        (сумма по каждой грани, клетки, веса клеток, число клеток граней)."""
        ids = np.asarray(ids, int)
        cnt = F_cnt[ids]
        offs = np.repeat(F["ptr"][ids] - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
        cells = F["cells"][np.arange(cnt.sum()) + offs]
        e = F_eps[ids][np.repeat(np.arange(len(ids)), cnt)]
        g = sum(W2[i] * e[:, i] * Sf[i][cells] for i in range(6)) / EPS_N
        if p.cap_local_only:
            gm = (e @ (W2 * st["Sm"])) / EPS_N
            g = np.clip(g - gm, -cap, cap) + gm
        else:
            g = np.clip(g, -cap, cap)
        ex = p.beta * (g + np.repeat(F_gapp[ids], cnt))
        ok = ~occ_f[cells]
        if capture:
            c = cH.reshape(-1)[cells]
            ok &= c > 1e-3
            ex = ex + np.log(np.clip(c, 1e-12, None))
        wv = np.where(ok, np.exp(np.minimum(ex - st["E0"], 700.0)), 0.0) * p.w_gb
        tot = np.add.reduceat(wv, np.concatenate([[0], np.cumsum(cnt)[:-1]])) if len(wv) else np.zeros(len(ids))
        return tot, cells, wv, cnt

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
        if p.gb:
            wf[:] = facet_weights(np.arange(nfac))[0]
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

    def gcap_cells(t, wgt):
        """Средняя выгода (с потолком на ближнее поле) по клеткам t с весами wgt."""
        g = graw[t]
        if p.cap_local_only:
            gm = _contract([a[t] for a in e_new], st["Sm"])
            g = np.clip(g - gm, -cap, cap) + gm
        else:
            g = np.clip(g, -cap, cap)
        return float((g * wgt).sum() / wgt.sum())

    def make_segment(ci, nv):
        """Отрезок макрогидрида: из шести вариантов {10-17} зерна выбирается вариант с вероятностью
        ∝ exp(β·ḡ), ḡ — средняя выгода по его клеткам. Несоответствие — δ в осях кристалла (nv = ось c)."""
        opts = []
        for nh, u in habit_normals(nv, a1[grains[ci]]):
            cross = (lambda g, nh=nh: np.abs(HN[g] @ nh).max() >= cos_x) if p.cross_deg > 0 else None
            sg = grow_segment(ci, nh, grains, occ, p, cross=cross)
            if sg is None or sg[3] < p.S_min:
                continue
            lo, fr, cen, area, gw = sg
            nz = np.nonzero(fr)
            t = tuple((lo[d] + nz[d]) % n for d in range(3))
            opts.append((gcap_cells(t, fr[nz]), lo, fr, cen, area, nh, u, gw))
        if not opts:
            return None
        sc = p.beta * np.array([o[0] for o in opts])
        pr = np.exp(sc - sc.max())
        _, lo, fr, cen, area, nh, u, gw = opts[rng.choice(len(opts), p=pr / pr.sum())]
        gs_ = np.unique(gw[fr > 0])
        parts = [(fr * (gw == g), eps_star(cax[g])) for g in gs_]   # несоответствие — в осях кристалла каждого зерна
        return dict(c=cen, n=nh, cax=nv.copy(), u=u, area=float(area), R=float(np.sqrt(area / np.pi)),
                    lo=lo, fr=fr, anchor=np.array(ci), parts=parts, ngr=int(len(gs_)))

    def make_gb(kf, ci):
        """Межзёренный отрезок на грани kf через клетку ci: плоскость грани, оба зерна; несоответствие —
        по p.gb_eps (по нормали к грани или в осях соседнего зерна)."""
        nf = F["n"][kf]
        L_ = n * p.dx
        dc = (F["cen"][kf] - (np.array(ci) + 0.5) * p.dx + L_ / 2) % L_ - L_ / 2
        sg = grow_segment(ci, nf, grains, occ, p, allowed=[F["a"][kf], F["b"][kf]], q0=float(dc @ nf))
        if sg is None or sg[3] < p.S_min:
            return None
        lo, fr, cen, area, _ = sg
        uu = np.cross(nf, [0, 0, 1.0] if abs(nf[2]) < 0.9 else [1.0, 0, 0])
        return dict(c=cen, n=nf.copy(), cax=F_ax[kf].copy(), u=uu / np.linalg.norm(uu), area=float(area), R=float(np.sqrt(area / np.pi)),
                    lo=lo, fr=fr, anchor=np.array(ci), parts=[(fr, eps_star(F_ax[kf]))], gb=True,
                    pair=(int(F["a"][kf]), int(F["b"][kf])), ngr=2)

    def make_plate(ci, nv):
        """Пластинка из зародыша ci или None. Игла: из трёх ⟨11-20⟩ выбирается вариант с вероятностью
        ∝ exp(β·ḡ), ḡ — средняя выгода вдоль её длины (взаимодействие формы с полем соседей)."""
        if p.shape == "segment":
            return make_segment(ci, nv)
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
        tried_f = []
        for _ in range(40):
            cdf = np.cumsum(np.clip(bsum, 0, None).ravel())
            Wg = float(wf.sum()) if p.gb else 0.0
            if cdf[-1] + Wg <= 0:
                break
            if Wg > 0:
                cdf_f = np.cumsum(wf)
            for bk in np.searchsorted(cdf, rng.random(64) * cdf[-1]):
                if Wg > 0 and rng.random() * (cdf[-1] + Wg) < Wg:     # межзёренное зарождение
                    kf = int(min(np.searchsorted(cdf_f, rng.random() * cdf_f[-1]), nfac - 1))
                    _, fcells, fwv, _ = facet_weights([kf])
                    if fwv.sum() <= 0:
                        wf[kf] = 0.0
                        continue
                    cell = fcells[min(np.searchsorted(np.cumsum(fwv), rng.random() * fwv.sum()), len(fwv) - 1)]
                    ci = np.unravel_index(cell, (n,) * 3)
                    nv = cax[grains[ci]]
                    q = make_gb(kf, ci)
                    if q is not None:
                        ok = True
                        break
                    wf[kf] = 0.0
                    tried_f.append(kf)
                    continue
                if cdf[-1] <= 0:
                    continue
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
        if "lo" in q:                                         # отрезок: окно вокруг зародыша
            o = q["anchor"] - win // 2
            fr, lo = q.pop("fr"), q.pop("lo")
            ax = [lo[d] - o[d] + np.arange(fr.shape[d]) for d in range(3)]
            assert all(a[0] >= 0 and a[-1] < win for a in ax)
        else:
            o = np.floor(c0 / p.dx).astype(int) - win // 2
            ax, fr = plate_fraction(win, p.dx, c0 - o * p.dx, q, p.h_um)
        ix = [(o[d] + np.arange(win)) % n for d in range(3)]
        sel = np.ix_(*ix)
        parts = q.pop("parts", None) or [(fr, eps_star(q.get("cax", nv)))]
        for frp, es in parts:
            f = np.zeros((win,) * 3, np.float32)
            f[np.ix_(*ax)] = frp
            corr = float(frp.sum()) / win ** 3 * (Mw @ np.array(es))
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
        for frp, es in parts:
            for i in range(6):
                Eacc[i][dsel] += (frp * es[i]).astype(np.float32)
        oc = occ[dsel] | (fr > 0.2)
        o2 = oc.copy()
        for d in range(3):
            o2 |= np.roll(oc, 1, d) | np.roll(oc, -1, d)
        occ[dsel] = o2
        if "anchor" in q:                                     # клетки отрезка — для 3D-геометрии
            nzm = fr > 0
            q["cells"] = gi.reshape(fr.shape)[nzm].astype(np.int64)
            q["cfr"] = fr[nzm].astype(np.float32)
            del q["anchor"]
        plates.append(dict(q, c=c0 % (n * p.dx), grain=int(grains[ci])))
        if len(plates) % resync == 0:
            bsum = resync_all()
        elif p.cap_local_only and np.abs(Sm_now() - st["Sm"]).sum() > tol_mpa:
            bsum = resync_all(full=False)
        else:
            w[sel] = weights(sel)
            update_blocks(bsum, ix)
            if p.gb:
                near = np.array(F_tree.query_ball_point(c0 % (n * p.dx), r=0.9 * win * p.dx) + tried_f, int)
                if near.size:
                    near = np.unique(near)
                    wf[near] = facet_weights(near)[0]
            if tried:
                t = tuple(np.array(tried).T)
                w[t] = weights(t)
                for bi, bj, bl in set(zip(*(c // B for c in t))):
                    bsum[bi, bj, bl] = w6[bi, :, bj, :, bl, :].sum()
        if verbose and len(plates) % 500 == 0:
            print(len(plates), f"{total / n ** 3 * 100:.2f} %", flush=True)
    return dict(params=p, plates=plates, hyd=hyd, grains=grains, cax=cax, S=S)
