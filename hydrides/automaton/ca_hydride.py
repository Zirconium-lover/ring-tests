"""Клеточный автомат зарождения гидридов в сечении стенки (плоскость ND–TD = r–θ).

Идея
----
Гидрид δ-ZrH₁.₆₆ больше по объёму, чем цирконий: несоответствие ≈ 7.2 % по нормали
к пластинке и ≈ 4.6 % в её плоскости. Каждая выросшая пластинка создаёт вокруг
себя поле напряжений. Новая пластинка зарождается охотнее там, где это поле
(вместе с приложенным напряжением) совершает над её собственным расширением
положительную работу: g = σ : ε*_новой.

Модель (одна итерация = одна новая пластинка):
  1. Зёрна — ячейки Вороного; в каждом зерне плоскость пластинки задана текстурой
     (габитус близок к базисной плоскости, угол следа ψ от TD = наклон базисного
     полюса от ND).
  2. Поле напряжений от всех пластинок — микроупругость Хачатуряна: собственная
     деформация пластинок → решение уравнений равновесия в пространстве Фурье
     (изотропная упругость, плоская деформация, периодическая ячейка).
  3. Пластическая релаксация: «эффективное» напряжение от соседей ограничено
     пределом σ_cap (у вершин упругое решение даёт гигапаскали, реальный металл
     при 300–400 °C их не держит).
  4. Вероятность зарождения в клетке ∝ exp(β · g / ε*_nn), запрещены клетки на
     пластинках и вплотную к ним. Выбирается одна клетка (кинетический Монте-Карло).
  5. Пластинка растёт от точки зарождения в обе стороны по своей плоскости до
     границы зерна, до другой пластинки или до L_max.
  6. Повтор, пока доля гидрида не достигнет заданной (водородный баланс).

Параметры: β — избирательность (1/МПа); σ_app — окружное напряжение при
охлаждении; текстура (χ0, s); размер зёрен; σ_cap.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

EPS_N, EPS_T = 0.0720, 0.0458        # несоответствие δ-гидрида: по нормали, в плоскости (Carpenter 1973)


@dataclass
class Params:
    size_um: tuple = (240.0, 240.0)   # (ND, TD) — строки, столбцы
    dx: float = 0.4                   # мкм на клетку
    grain_um: tuple = (2.5, 4.5)      # размер зерна по ND и по TD
    chi0: float = 30.0                # наклон базисного полюса от ND к TD, град
    chi_s: float = 26.0               # разброс наклона, град
    chi0_profile: tuple = ()          # (наружный, средний, внутренний) χ0 по толщине; пусто — везде chi0
    beta: float = 0.05                # избирательность, 1/МПа
    sigma_app: float = 0.0            # окружное (по TD) напряжение при охлаждении, МПа
    sigma_app_grad: float = 0.0       # изменение σ_app по толщине: σ(y) = σ_app + grad·(y/H − 0.5)
    sigma_cap: float = 200.0          # предел напряжения от соседних пластинок, МПа
    kappa: float = 1.0                # доля несоответствия, не снятая пластичностью
    beta_h: float = 0.0               # вклад гидростатики (накопление водорода), 1/МПа
    h_um: float = 0.6                 # толщина пластинки
    L_max: float = 6.0                # наибольшая длина пластинки
    L_min: float = 1.2                # меньше — зародыш не растёт
    frac: float = 0.011               # доля площади гидрида (≈ 180 wppm)
    E: float = 90e3                   # МПа
    nu: float = 0.34
    v_h_over_rt: float = 1.01e-4      # V_H/(RT) при 400 °C, 1/МПа (перераспределение водорода, Горский)
    capture_um: float = 0.0           # длина сбора водорода пластинкой, мкм (0 — водород общий на всё поле)
    eps_n: float = EPS_N              # несоответствие по нормали (для опытов «без» — менять)
    eps_t: float = EPS_T              # несоответствие в плоскости пластинки
    seed: int = 0
    record_every: int = 0             # сохранять снимки по ходу (0 — нет)
    g_extra: object = None            # добавка к выгоде по клеткам, МПа (напряжения несовместности из МКЭ), массив (ny, nx)
    init_plates: object = None        # пластинки, оставшиеся с прошлого цикла (не растворились): [(cy, cx, ψ, полудлина), ...]
    schedule: object = None           # schedule(progress 0..1) → dict(T, beta, sigma_cap, sigma_app, g_extra) — ход охлаждения
    cap_local_only: bool = False      # потолок только на ближнее поле; среднее напряжение в металле — без потолка
    free_z: bool = False              # обобщённая плоская деформация: среднее ε_zz = среднему ε*_zz (σ̄_zz = 0);
                                      # False — ε_zz = 0, от осевого несоответствия гидрида остаётся однородное
                                      # сжатие σ̄ = −λ⟨ε*_zz⟩ в плоскости и −(λ + 2μ)⟨ε*_zz⟩ по оси (fe/stack_auto.py)
    screen: bool = False              # экранирование приложенного напряжения у гидридов (пластическая зона, МКЭ)
    screen_scale: float = 1.0         # масштаб расстояний профиля экранирования (1 — как у пластинки 5 × 0.6 мкм в МКЭ)
    screen_d: tuple = (0.0, 0.75, 1.25, 1.75, 2.5, 3.5)        # расстояние до гидрида, мкм
    screen_s: tuple = (0.05, 0.08, 0.45, 0.45, 0.70, 1.0)      # доля действия приложенного напряжения (fe/biax_inh.py)
    app_dg: object = None             # (σθ, Δg): выгода радиальной пластинки над окружной от нагрузки по МКЭ
                                      # (fe/hill_runs.py, пластичность Мизеса или Хилла); None — упругая 0.36·σθ


# ------------------------------------------------------------------ зёрна и текстура
def make_grains(p, rng, gb=False):
    """Зёрна (номер зерна по клеткам) и углы следа базисной плоскости. gb=True — ещё и границы зёрен:
    dict(cells — индексы клеток на границе (тонкая цифровая линия вдоль грани Вороного), pair — пары
    зёрен (меньший, больший номер), psi — угол следа грани от TD). Случайные числа расходуются так же,
    как без gb, поэтому зёрна одинаковые."""
    ny, nx = int(p.size_um[0] / p.dx), int(p.size_um[1] / p.dx)
    area = p.size_um[0] * p.size_um[1]
    n = int(area / (p.grain_um[0] * p.grain_um[1]))
    seeds = rng.uniform([0, 0], p.size_um, size=(n, 2))
    # анизотропная метрика: зерно вытянуто вдоль TD; периодичность — 9 копий семян
    sc = np.array([1.0 / p.grain_um[0], 1.0 / p.grain_um[1]])
    reps = np.concatenate([seeds + np.array([a, b]) * p.size_um for a in (-1, 0, 1) for b in (-1, 0, 1)])
    ids = np.tile(np.arange(n), 9)
    tree = cKDTree(reps * sc)
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.stack([(yy + 0.5) * p.dx, (xx + 0.5) * p.dx], -1).reshape(-1, 2)
    _, j = tree.query(pts * sc)
    grains = ids[j].reshape(ny, nx)
    # угол следа пластинки от TD = наклон базисного полюса от ND: ±χ0 со случайным разбросом
    sign = rng.choice([-1, 1], size=n)
    if p.chi0_profile:
        yf = seeds[:, 0] / p.size_um[0]                    # 0 — наружная поверхность (верх), 1 — внутренняя
        chi = np.interp(yf, [0.0, 0.5, 1.0], list(p.chi0_profile))
    else:
        chi = p.chi0
    psi = sign * chi + rng.normal(0, p.chi_s, size=n)
    psi = (psi + 90) % 180 - 90                         # в (−90, 90]
    if not gb:
        return grains, np.radians(psi)
    # граница двух ближайших семян (в растянутой метрике) — прямая n·x = b, n = S²(x2 − x1);
    # клетка на границе, если её центр ближе к прямой, чем dx/2·max(|n_y|, |n_x|)/|n| (по клетке на столбец/строку)
    _, j2 = tree.query(pts * sc, k=2)
    r1, r2 = reps[j2[:, 0]], reps[j2[:, 1]]
    nvec = (r2 - r1) * sc ** 2
    b = 0.5 * (((r2 * sc) ** 2).sum(1) - ((r1 * sc) ** 2).sum(1))
    nn = np.linalg.norm(nvec, axis=1)
    dist = np.abs((pts * nvec).sum(1) - b) / nn
    on = dist <= 0.5 * p.dx * np.abs(nvec).max(1) / nn
    g1, g2 = ids[j2[on, 0]], ids[j2[on, 1]]
    # след грани: t ⟂ n; в осях (строка вниз, столбец) t = (−sin ψ, cos ψ) → ψ = atan2(n_x, n_y)
    psi_f = np.arctan2(nvec[on, 1], nvec[on, 0])
    psi_f = (psi_f + np.pi / 2) % np.pi - np.pi / 2
    gbd = dict(cells=np.flatnonzero(on), pair=np.stack([np.minimum(g1, g2), np.maximum(g1, g2)], 1), psi=psi_f)
    return grains, np.radians(psi), gbd


# ------------------------------------------------------------------ микроупругость (Фурье)
class Elastic:
    """Поле напряжений от собственных деформаций в периодической ячейке, плоская деформация."""

    def __init__(self, shape, dx, E, nu, free_z=False):
        self.shape = shape
        self.free_z = free_z
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.nu = nu
        ky = 2 * np.pi * np.fft.fftfreq(shape[0], dx)[:, None]
        kx = 2 * np.pi * np.fft.rfftfreq(shape[1], dx)[None, :]
        k2 = kx ** 2 + ky ** 2
        k2[0, 0] = 1.0
        self.kx, self.ky, self.k2 = kx, ky, k2
        a = (self.lam + self.mu) / (self.lam + 2 * self.mu)
        # K⁻¹ = (1/(μk²)) [I − a k̂k̂ᵀ]
        self.Ki = (1.0 / (self.mu * k2), a / k2)

    def stress(self, e11, e22, e12, e33):
        lam, mu = self.lam, self.mu
        tr = e11 + e22 + e33
        s11 = sfft.rfft2(lam * tr + 2 * mu * e11)
        s22 = sfft.rfft2(lam * tr + 2 * mu * e22)
        s12 = sfft.rfft2(2 * mu * e12)
        kx, ky = self.kx, self.ky
        t1 = kx * s11 + ky * s12
        t2 = kx * s12 + ky * s22
        c0, a = self.Ki
        kt = (kx * t1 + ky * t2)
        u1 = c0 * (t1 - a * kx * kt)            # (K⁻¹ τ), множитель −i учтён ниже
        u2 = c0 * (t2 - a * ky * kt)
        eps11 = kx * u1
        eps22 = ky * u2
        eps12 = 0.5 * (kx * u2 + ky * u1)
        for arr in (eps11, eps22, eps12):
            arr[0, 0] = 0.0
        E11 = np.fft.irfft2(eps11, s=self.shape)
        E22 = np.fft.irfft2(eps22, s=self.shape)
        E12 = np.fft.irfft2(eps12, s=self.shape)
        # средняя деформация в плоскости = средней собственной (свободное макрорасширение); σ̄ = 0 при free_z
        E11 += e11.mean(); E22 += e22.mean(); E12 += e12.mean()
        E33 = float(np.mean(e33)) if self.free_z else 0.0          # иначе ε_zz = 0 (плоская деформация)
        ekk = E11 + E22 + E33 - tr
        S11 = lam * ekk + 2 * mu * (E11 - e11)
        S22 = lam * ekk + 2 * mu * (E22 - e22)
        S12 = 2 * mu * (E12 - e12)
        S33 = lam * ekk + 2 * mu * (E33 - e33)
        return S11, S22, S12, S33


def plate_eigen(shape, dx, c, psi, half, h, sub=3):
    """Собственная деформация одной пластинки (центр c=(y, x) мкм, угол ψ от TD, полудлина)."""
    t = np.array([np.sin(psi), np.cos(psi)])           # (y, x): ψ откладываем вверх от TD
    t[0] = -t[0]                                       # строки идут вниз
    n = np.array([t[1], -t[0]])
    r = half + h
    ny, nx = shape
    y0, y1 = int((c[0] - r) / dx) - 1, int((c[0] + r) / dx) + 2
    x0, x1 = int((c[1] - r) / dx) - 1, int((c[1] + r) / dx) + 2
    ys = np.arange(y0, y1); xs = np.arange(x0, x1)
    off = (np.arange(sub) + 0.5) / sub
    Y = (ys[:, None, None, None] + off[None, None, :, None]) * dx
    X = (xs[None, :, None, None] + off[None, None, None, :]) * dx
    s = (Y - c[0]) * t[0] + (X - c[1]) * t[1]
    q = (Y - c[0]) * n[0] + (X - c[1]) * n[1]
    frac = ((np.abs(s) <= half) & (np.abs(q) <= h / 2)).mean(axis=(2, 3))
    # компоненты ε* в осях (y, x) → (11 = x, 22 = y)
    tx, ty = t[1], t[0]
    nx_, ny_ = n[1], n[0]
    e11 = EPS_N * nx_ ** 2 + EPS_T * tx ** 2
    e22 = EPS_N * ny_ ** 2 + EPS_T * ty ** 2
    e12 = EPS_N * nx_ * ny_ + EPS_T * tx * ty
    return ys % ny, xs % nx, frac, (e11, e22, e12, EPS_T)


def eigen_components_for(psi):
    """ε* гипотетической пластинки с углом ψ (в глобальных осях x = TD, y = ND вверх)."""
    s, c = np.sin(psi), np.cos(psi)
    # t = (c, s) в осях (x, y↑), n = (−s, c). В массиве ось y направлена вниз: s → −s
    tx, ty = c, -s
    nx_, ny_ = s, c
    e11 = EPS_N * nx_ ** 2 + EPS_T * tx ** 2
    e22 = EPS_N * ny_ ** 2 + EPS_T * ty ** 2
    e12 = EPS_N * nx_ * ny_ + EPS_T * tx * ty
    return e11, e22, e12


# ------------------------------------------------------------------ рост пластинки
def grow(c_idx, psi, grains, occ, p):
    """Растим от клетки c_idx в обе стороны по следу ψ, пока то же зерно и свободно."""
    ny, nx = grains.shape
    g0 = grains[c_idx]
    ty, tx = -np.sin(psi), np.cos(psi)
    step = p.dx * 0.5
    ext = []
    for sgn in (1, -1):
        d = 0.0
        while d < p.L_max / 2:
            d2 = d + step
            yy = (c_idx[0] + 0.5) * p.dx + sgn * d2 * ty
            xx = (c_idx[1] + 0.5) * p.dx + sgn * d2 * tx
            iy, ix = int(np.floor(yy / p.dx)) % ny, int(np.floor(xx / p.dx)) % nx
            if grains[iy, ix] != g0 or occ[iy, ix]:
                break
            d = d2
        ext.append(d)
    a, b = ext
    half = 0.5 * (a + b)
    shift = 0.5 * (a - b)
    cy = (c_idx[0] + 0.5) * p.dx + shift * ty
    cx = (c_idx[1] + 0.5) * p.dx + shift * tx
    return np.array([cy, cx]), half


# ------------------------------------------------------------------ прогон
def run(p: Params, verbose=False, callback=None):
    """callback(plates, expo, hyd, cH) вызывается перед выбором каждой новой пластинки
    (expo — логарифм веса зарождения по клеткам; −inf там, где нельзя).
    p.init_plates — пластинки, которые уже есть до охлаждения (нерастворившиеся);
    p.schedule(progress) — параметры шага по ходу выпадения (температура, β, потолок, напряжение)."""
    global EPS_N, EPS_T
    EPS_N, EPS_T = p.eps_n, p.eps_t
    rng = np.random.default_rng(p.seed)
    grains, gpsi = make_grains(p, rng)
    ny, nx = grains.shape
    el = Elastic((ny, nx), p.dx, p.E, p.nu, free_z=p.free_z)
    psi_map = gpsi[grains]
    e11n, e22n, e12n = eigen_components_for(psi_map)
    e33n = EPS_T
    yfrac = (np.arange(ny) + 0.5) / ny
    # приложенное напряжение выбирает ориентацию пластинки: работа над той частью несоответствия,
    # что зависит от ориентации (ε11 − среднее по ориентациям). Средняя часть (σ·ε̄) от ориентации
    # не зависит; при неоднородном σ она лишь гнала бы весь водород в растянутую зону, а реальное
    # перераспределение водорода по Горскому мало: c ∝ exp(V_H·σ_h/RT), V_H ≈ 1.7 см³/моль.
    e_mean = 0.5 * (EPS_N + EPS_T)

    def applied(sig):
        """σ_xx = sig (вдоль TD), возможно с градиентом по толщине (ND) → (g_app, вклад Горского)."""
        sapp = (sig + p.sigma_app_grad * (0.5 - yfrac))[:, None] * np.ones((1, nx))
        gor = p.v_h_over_rt * sapp / 3.0 if p.v_h_over_rt else 0.0
        if p.app_dg is not None:      # Δg(σθ) по МКЭ; между окружной и радиальной — как sin²ψ (как в упругости)
            return np.interp(sapp, *p.app_dg) * (np.sin(psi_map) ** 2 - 0.5), gor
        return sapp * (e11n - e_mean) / EPS_N, gor

    g_app, gorsky = applied(p.sigma_app)
    S11 = np.zeros((ny, nx)); S22 = np.zeros_like(S11); S12 = np.zeros_like(S11); S33 = np.zeros_like(S11)
    E11 = np.zeros_like(S11); E22 = np.zeros_like(S11); E12 = np.zeros_like(S11); E33 = np.zeros_like(S11)
    occ = np.zeros((ny, nx), bool)          # занято пластинками (с зазором)
    hyd = np.zeros((ny, nx))                # доля гидрида в клетке
    plates = []
    tried = np.zeros((ny, nx), bool)
    snaps = []
    # поле доступного водорода: 1 — исходное пересыщение; пластинка истощает окрестность ~capture_um
    cH = np.ones((ny, nx))
    if p.capture_um > 0:
        ky = np.fft.fftfreq(ny, p.dx)[:, None]; kx = np.fft.rfftfreq(nx, p.dx)[None, :]
        Gk = np.exp(-2 * (np.pi * p.capture_um) ** 2 * (kx ** 2 + ky ** 2))

    scr = np.ones((ny, nx))                 # экранирование приложенного напряжения

    def update_screen():
        """Доля действия приложенного напряжения на выбор ориентации в зависимости от расстояния до гидрида:
        в пластической зоне у гидрида матрица на пределе текучести и лишнего девиатора не держит (МКЭ)."""
        m = int(np.ceil(max(p.screen_d) * p.screen_scale / p.dx)) + 2
        solid = np.pad(hyd > 0.2, m, mode="wrap")
        d = ndi.distance_transform_edt(~solid)[m:m + ny, m:m + nx] * p.dx
        scr[:] = np.interp(d / p.screen_scale, p.screen_d, p.screen_s)

    def add_plate(c, psi, half, deplete=True):
        nonlocal cH
        ys, xs, fr, (a11, a22, a12, a33) = plate_eigen((ny, nx), p.dx, c, psi, half, p.h_um)
        sel = np.ix_(ys, xs)
        hyd[sel] = np.clip(hyd[sel] + fr, 0, 1)
        if deplete and p.capture_um > 0:
            add = np.zeros((ny, nx)); add[sel] = fr
            cH -= np.fft.irfft2(np.fft.rfft2(add) * Gk, s=(ny, nx)) / p.frac
        E11[:] = 0; E22[:] = 0; E12[:] = 0; E33[:] = 0
        E11[sel] = fr * a11; E22[sel] = fr * a22; E12[sel] = fr * a12; E33[sel] = fr * a33
        d11, d22, d12, d33 = el.stress(E11, E22, E12, E33)
        S11[:] += d11; S22[:] += d22; S12[:] += d12; S33[:] += d33
        occ[:] |= ndi.binary_dilation(hyd > 0.2, iterations=1)
        if p.screen:
            update_screen()

    # нерастворившиеся пластинки: поле и место есть, водород вокруг уже выровнялся
    if p.init_plates is not None:
        for cy, cx, psi, half in p.init_plates:
            add_plate(np.array([cy, cx]), psi, half, deplete=False)
            iy, ix = int(cy / p.dx) % ny, int(cx / p.dx) % nx
            plates.append(dict(c=np.array([cy, cx]), psi=float(psi), half=float(half), grain=int(grains[iy, ix]),
                               T=None, init=True))
    hyd0 = hyd.sum()
    target = hyd0 + p.frac * ny * nx
    beta, cap, sig, T, g_extra = p.beta, p.sigma_cap, p.sigma_app, None, p.g_extra
    while hyd.sum() < target:
        if p.schedule is not None:
            st = p.schedule((hyd.sum() - hyd0) / (target - hyd0))
            beta = st.get("beta", beta); cap = st.get("sigma_cap", cap); T = st.get("T", T)
            g_extra = st.get("g_extra", g_extra)
            if "sigma_app" in st and st["sigma_app"] != sig:
                sig = st["sigma_app"]; g_app, gorsky = applied(sig)
        g_int = (e11n * S11 + e22n * S22 + 2 * e12n * S12 + e33n * S33) / EPS_N * p.kappa
        if p.cap_local_only:
            # среднее напряжение в металле (отклик на все гидриды) — без потолка, ближнее поле — с потолком
            mtx = hyd < 0.2
            Sm = [float(a[mtx].mean()) for a in (S11, S22, S12, S33)]
            g_mean = (e11n * Sm[0] + e22n * Sm[1] + 2 * e12n * Sm[2] + e33n * Sm[3]) / EPS_N * p.kappa
            g_int = np.clip(g_int - g_mean, -cap, cap) + g_mean
        else:
            g_int = np.clip(g_int, -cap, cap)
        expo = beta * (g_int + (g_app * scr if p.screen else g_app)) + gorsky
        if g_extra is not None:
            expo = expo + beta * g_extra
        if p.beta_h:
            expo = expo + p.beta_h * p.kappa * (1 + p.nu) * (S11 + S22) / 3
        allowed = ~occ & ~tried
        if p.capture_um > 0:
            allowed &= cH > 1e-3
            expo = expo + np.log(np.clip(cH, 1e-12, None))
        if not allowed.any():
            break
        expo = np.where(allowed, expo, -np.inf)
        if callback is not None:
            callback(plates, expo, hyd, cH)
        w = np.exp(expo - expo[allowed].max())
        w /= w.sum()
        k = rng.choice(w.size, p=w.ravel())
        c_idx = np.unravel_index(k, w.shape)
        psi = psi_map[c_idx]
        c, half = grow(c_idx, psi, grains, occ, p)
        if 2 * half < p.L_min:
            tried[c_idx] = True                 # зародыш не вырос: в эту клетку больше не пробуем
            continue
        add_plate(c, psi, half)
        tried[:] = False
        plates.append(dict(c=c, psi=float(psi), half=float(half), grain=int(grains[c_idx]), T=T, init=False))
        if p.record_every and len(plates) % p.record_every == 0:
            snaps.append(hyd.copy())
        if verbose and len(plates) % 50 == 0:
            print(len(plates), f"{hyd.sum() / (ny * nx) * 100:.2f} %")
    return dict(params=p, plates=plates, hyd=hyd, grains=grains, gpsi=gpsi,
                S=(S11, S22, S12, S33), snaps=snaps)
