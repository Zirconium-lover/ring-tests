"""Спектральный анализ гидридов на снимке поперечного сечения стенки трубы/кольца.

Что делает (по шагам):
  1. Находит две поверхности стенки (наружную и внутреннюю) и подгоняет к ним
     концентрические окружности.
  2. Разворачивает дугу в прямую полосу: по горизонтали — дуга (окружное
     направление), по вертикали — глубина от наружной поверхности.
  3. Выделяет тёмные линии (гидриды) «чёрным цилиндром» — сигнал не зависит
     от неравномерной подсветки.
  4. Ориентация по масштабам. Изображение сглаживается гауссом σ (в Фурье это
     низкочастотный фильтр), затем считается матрица вторых производных
     (в Фурье — умножение на −k·kᵀ). Её собственные векторы дают направление
     линии в каждой точке на этом масштабе. Малое σ — отдельные пластинки,
     большое σ — цепочки и сетка, которые из них сложены.
     Доля радиальных Fn(σ) — доля длины осевых линий, отклонённых от окружного
     направления больше чем на 45°. Длина — по скелету, как в RHF.
  5. Профили по толщине: доля площади гидридов и Fn(σ) по слоям.
  6. Связность: какой зазор матрицы приходится «перепрыгивать», чтобы пройти
     по гидридам через стенку (и вдоль дуги — для сравнения), и доля матрицы
     на лучшем пути трещины через стенку (аналог идеи RHCP).
  7. Проверка спектром: Fn по углу энергии двумерного спектра в каждом слое.

Запуск:
  python hydride_spec.py снимок.jpg --um-per-px 0.862 --out папка
  python hydride_spec.py снимок.jpg --bar-um 200 --out папка   # масштаб по линейке

Ориентация снимка: наружная поверхность сверху (иначе --outer bottom).
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal.windows import tukey
from skimage.filters import threshold_otsu
from skimage.graph import MCP_Geometric
from skimage.morphology import disk, skeletonize
from skimage.measure import label as sk_label, regionprops


# ---------------------------------------------------------------- ввод
def read_gray(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("L")).astype(float) / 255.0


def find_scale_bar(g, bar_um):
    """Самая длинная горизонтальная светлая или тёмная линия в нижних 18 % кадра."""
    H, W = g.shape
    best = 0
    for y in range(int(H * 0.82), H):
        for mask in (g[y] > 0.8, g[y] < 0.15):
            idx = np.where(mask)[0]
            if idx.size < 20:
                continue
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            L = max(len(r) for r in runs)
            if W * 0.04 < L < W * 0.5 and L > best:
                best = L
    if best == 0:
        raise SystemExit("масштабная линейка не найдена — задайте --um-per-px")
    return bar_um / best


# ---------------------------------------------------------------- стенка
def wall_mask(g, um):
    """Металл стенки: светлее заливки (оптика) или текстурнее её (СЭМ после фильтра)."""
    s = 6.0 / um
    bright = ndi.gaussian_filter(g, s)
    tex = np.sqrt(np.clip(ndi.gaussian_filter(g * g, s) - ndi.gaussian_filter(g, s) ** 2, 0, None))
    best = None
    for feat in (bright, tex):
        f = (feat - feat.mean()) / (feat.std() + 1e-9)
        t = threshold_otsu(f)
        m = f > t
        # бимодальность: доля дисперсии между классами
        w1 = m.mean()
        sep = w1 * (1 - w1) * (f[m].mean() - f[~m].mean()) ** 2 / f.var()
        if best is None or sep > best[0]:
            best = (sep, m)
    m = best[1]
    lab, n = ndi.label(m)
    if n == 0:
        raise SystemExit("стенка не найдена")
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (1 + np.argmax(sizes))
    return ndi.binary_fill_holes(ndi.binary_closing(m, iterations=int(10 / um) + 1))


def surface_points(mask, outer_top=True):
    H, W = mask.shape
    xs, top, bot = [], [], []
    for x in range(0, W, 2):
        idx = np.where(mask[:, x])[0]
        if idx.size < 10:
            continue
        xs.append(x); top.append(idx[0]); bot.append(idx[-1])
    xs, top, bot = map(np.asarray, (xs, top, bot))
    # отбросить столбцы, где край упирается в рамку кадра
    ok = (top > 1) & (bot < H - 2)
    xs, top, bot = xs[ok], top[ok], bot[ok]
    if not outer_top:
        top, bot = bot, top
    return xs, top.astype(float), bot.astype(float)


def surfaces_from_outline(g, um, outer_top=True):
    """Поверхности как две самые крайние длинные тёмные линии (контур шлифа на СЭМ после фильтра,
    когда заливка по яркости и текстуре не отличается от металла)."""
    H, W = g.shape
    d = 1 - ndi.gaussian_filter(g, (0.6 / um, 12.0 / um))      # длинные горизонтальные линии усилены
    step = max(2, W // 120)
    xs, top, bot = [], [], []
    for x in range(int(W * 0.02), int(W * 0.98), step):
        col = d[:, x]
        base = ndi.median_filter(col, size=int(30 / um) | 1)
        peak = col - base
        thr = 0.35 * peak.max()
        cand = np.where((peak > thr) & (peak >= np.roll(peak, 1)) & (peak >= np.roll(peak, -1)))[0]
        cand = cand[(cand > 1) & (cand < H - 2)]
        if cand.size >= 2 and cand[-1] - cand[0] > 0.3 * H:
            xs.append(x); top.append(cand[0]); bot.append(cand[-1])
    xs, top, bot = map(lambda v: np.asarray(v, float), (xs, top, bot))
    if xs.size < 10:
        raise SystemExit("контур стенки не найден — попробуйте --surfaces mask")
    # робастно: медианное сглаживание, затем квадратичная подгонка с отбрасыванием выбросов
    for arr in (top, bot):
        arr[:] = ndi.median_filter(arr, size=9, mode="nearest")
        ok = np.ones_like(arr, bool)
        for _ in range(4):
            p = np.polyfit(xs[ok], arr[ok], 2)
            r = arr - np.polyval(p, xs)
            ok = np.abs(r) < max(2.0, 2.0 * np.std(r[ok]))
        arr[:] = np.polyval(p, xs)
    if not outer_top:
        top, bot = bot, top
    return xs, top, bot


def surfaces_near(g, um, y_top, y_bot, tol_px=20, outer_top=True):
    """Поверхности по подсказке: в каждом столбце — самая тёмная длинная линия рядом с y_top и y_bot."""
    H, W = g.shape
    d = 1 - ndi.gaussian_filter(g, (0.6 / um, 8.0 / um))
    step = max(2, W // 160)
    xs = np.arange(int(W * 0.01), int(W * 0.99), step)
    out = []
    for y0 in (y_top, y_bot):
        lo, hi = max(0, int(y0 - tol_px)), min(H, int(y0 + tol_px) + 1)
        ys = np.array([lo + np.argmax(d[lo:hi, x]) for x in xs], float)
        ys = ndi.median_filter(ys, size=9, mode="nearest")
        ok = np.ones_like(ys, bool)
        for _ in range(4):
            p = np.polyfit(xs[ok], ys[ok], 2)
            r = ys - np.polyval(p, xs)
            ok = np.abs(r) < max(2.0, 2.0 * np.std(r[ok]))
        out.append(np.polyval(p, xs))
    top, bot = out
    if not outer_top:
        top, bot = bot, top
    return xs.astype(float), top, bot


def fit_concentric(xs, yo, yi):
    """Общий центр (xc, yc) и радиусы Ro, Ri по точкам двух краёв (линеаризованная задача)."""
    # (x-xc)^2 + (y-yc)^2 = R^2  ->  2x·xc + 2y·yc + (R^2 - xc^2 - yc^2) = x^2 + y^2
    n = len(xs)
    A = np.zeros((2 * n, 4)); b = np.zeros(2 * n)
    A[:n, 0] = 2 * xs; A[:n, 1] = 2 * yo; A[:n, 2] = 1; b[:n] = xs ** 2 + yo ** 2
    A[n:, 0] = 2 * xs; A[n:, 1] = 2 * yi; A[n:, 3] = 1; b[n:] = xs ** 2 + yi ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, co, ci = sol
    Ro = np.sqrt(co + xc ** 2 + yc ** 2); Ri = np.sqrt(ci + xc ** 2 + yc ** 2)
    res = np.concatenate([np.hypot(xs - xc, yo - yc) - Ro, np.hypot(xs - xc, yi - yc) - Ri])
    return xc, yc, Ro, Ri, float(np.sqrt(np.mean(res ** 2)))


def fit_center_known_r(xs, yo, yi, Ro, guess):
    """Центр при известном наружном радиусе; внутренний радиус — по точкам."""
    from scipy.optimize import least_squares

    def f(p):
        xc, yc, Ri = p
        return np.concatenate([np.hypot(xs - xc, yo - yc) - Ro, np.hypot(xs - xc, yi - yc) - Ri])
    sol = least_squares(f, guess)
    xc, yc, Ri = sol.x
    return xc, yc, Ro, Ri, float(np.sqrt(np.mean(sol.fun ** 2)))


def unwrap(g, xs, yo, yi, um, margin_um=3.0, ro_px=None):
    """Полоса «глубина × дуга». Строка 0 — наружная поверхность."""
    H, W = g.shape
    xc, yc, Ro, Ri, rms = fit_concentric(xs, yo, yi)
    if ro_px is not None:
        sgn = 1.0 if np.median(yi - yo) > 0 else -1.0
        t = np.median(np.abs(yi - yo))
        guess = (np.mean(xs), np.median(yo) + sgn * ro_px, ro_px - t)
        xc, yc, Ro, Ri, rms = fit_center_known_r(xs, yo, yi, ro_px, guess)
    straight = (not np.isfinite(Ro)) or Ro > 50 * W or rms > 6
    m = margin_um / um
    if straight:                       # почти прямая стенка: по столбцам между краями
        po = np.polyval(np.polyfit(xs, yo, 2), np.arange(W))
        pi = np.polyval(np.polyfit(xs, yi, 2), np.arange(W))
        x0, x1 = xs.min(), xs.max()
        cols = np.arange(int(x0), int(x1) + 1)
        nr = int(round(np.median(np.abs(pi - po)[cols]) - 2 * m))
        t = np.linspace(0, 1, nr)
        sgn = np.sign(np.median(pi - po))
        Y = po[cols][None, :] + sgn * (m + t[:, None] * (np.abs(pi - po)[cols][None, :] - 2 * m))
        X = np.broadcast_to(cols[None, :], Y.shape)
        geo = dict(kind="straight", rms_px=rms)
    else:
        # центр со стороны внутренней поверхности; угол φ отсчитываем от направления «вверх» к центру
        sgn = 1.0 if Ro > Ri else -1.0
        r_out, r_in = Ro - sgn * m, Ri + sgn * m
        phi_e = np.arctan2(xs - xc, -(yo - yc) * sgn)
        p0, p1 = phi_e.min(), phi_e.max()
        Rm = 0.5 * (Ro + Ri)
        ncol = int((p1 - p0) * Rm)
        phi = np.linspace(p0, p1, ncol)
        nr = int(round(abs(r_out - r_in)))
        r = np.linspace(r_out, r_in, nr)
        X = xc + r[:, None] * np.sin(phi)[None, :]
        Y = yc - sgn * r[:, None] * np.cos(phi)[None, :]
        geo = dict(kind="circle", xc=float(xc), yc=float(yc), Ro_um=float(Ro * um), Ri_um=float(Ri * um), rms_px=rms)
    S = ndi.map_coordinates(g, [Y, X], order=1, mode="nearest")
    inside = (X >= 0) & (X <= W - 1) & (Y >= 0) & (Y <= H - 1)
    geo["thickness_um"] = float((S.shape[0] + 2 * m) * um)
    return S, inside, X, Y, geo


# ---------------------------------------------------------------- сигнал гидридов
def hydride_signal(S, um, line_um=2.5):
    """Чёрный цилиндр: насколько точка темнее своего светлого окружения."""
    r = max(2, int(round(1.5 * line_um / um)))
    bg = ndi.grey_closing(S, footprint=disk(r))
    bg = ndi.gaussian_filter(bg, r)
    h = np.clip(bg - S, 0, None) / (bg + 1e-3)
    return h


# ---------------------------------------------------------------- Фурье-фильтры
def _fft_pad(img, pad):
    p = np.pad(img, pad, mode="reflect")
    ky = np.fft.fftfreq(p.shape[0])[:, None]
    kx = np.fft.fftfreq(p.shape[1])[None, :]
    return np.fft.fft2(p), kx, ky


def hessian_fft(img, sigma_px):
    """Гаусс σ и вторые производные — одним умножением в пространстве Фурье."""
    pad = int(4 * sigma_px) + 2
    F, kx, ky = _fft_pad(img, pad)
    G = np.exp(-2 * (np.pi * sigma_px) ** 2 * (kx ** 2 + ky ** 2))
    w = 2 * np.pi
    out = []
    for m in (-(w * kx) ** 2, -(w * kx) * (w * ky), -(w * ky) ** 2):
        a = np.real(np.fft.ifft2(F * G * m))
        out.append(a[pad:-pad, pad:-pad])
    return out  # Hxx, Hxy, Hyy


def ridges(h, sigma_px):
    """Сила линии и её направление на масштабе σ.
    Угол: 0 — вдоль дуги (окружное), +90 — по радиусу; знак «вверх» положительный."""
    Hxx, Hxy, Hyy = hessian_fft(h, sigma_px)
    tr = Hxx + Hyy
    d = np.sqrt(((Hxx - Hyy) / 2) ** 2 + Hxy ** 2)
    l1 = tr / 2 - d                      # самая отрицательная кривизна — поперёк линии
    l2 = tr / 2 + d
    strength = sigma_px ** 2 * np.clip(-l1, 0, None) * np.clip(1 - np.abs(l2) / (np.abs(l1) + 1e-12), 0, 1)
    # направление поперёк линии — собственный вектор для l1; вдоль — перпендикуляр
    theta_n = 0.5 * np.arctan2(2 * Hxy, Hxx - Hyy)          # для l2 (большее значение)…
    ang_along = np.degrees(theta_n)                          # …совпадает с направлением вдоль линии
    ang = (-ang_along) % 180                                 # строки идут вниз → меняем знак
    return strength, ang


def drop_small(mask, min_len_px):
    """Убрать точки и короткие обрывки: компоненты с длиной (по главной оси) меньше порога."""
    lab = sk_label(mask, connectivity=2)
    keep = np.zeros(lab.max() + 1, bool)
    for r in regionprops(lab):
        keep[r.label] = r.axis_major_length >= min_len_px
    return keep[lab]


def ridge_mask(strength, rel=0.25):
    """Линии там, где линия сильная (относительно сильнейших 1 %)."""
    return strength > rel * np.percentile(strength, 99.0)


def ridge_skeleton(strength, min_len_px=4, rel=0.25):
    return skeletonize(drop_small(ridge_mask(strength, rel), min_len_px))


def fn_from(angles, weights=None):
    dev = np.minimum(angles, 180 - angles)
    w = np.ones_like(dev) if weights is None else weights
    if w.sum() == 0:
        return np.nan, np.nan
    fn = w[dev > 45].sum() / w.sum()
    simon = np.where(dev <= 40, 0, np.where(dev < 65, 0.5, 1.0))
    return float(fn), float((simon * w).sum() / w.sum())


def spectral_fn(h, um, pmin_um, pmax_um, alpha=0.25):
    """Fn по углу энергии двумерного спектра (энергия — от квадрата площади, это другой вес!)."""
    if min(h.shape) < 16:
        return np.nan
    f = (h - h.mean()) * np.outer(tukey(h.shape[0], alpha), tukey(h.shape[1], alpha))
    P = np.abs(np.fft.fft2(f)) ** 2
    ky = np.fft.fftfreq(h.shape[0], um)[:, None]
    kx = np.fft.fftfreq(h.shape[1], um)[None, :]
    k = np.hypot(kx, ky)
    band = (k > 1 / pmax_um) & (k < 1 / pmin_um)
    ang_k = np.degrees(np.arctan2(np.abs(ky), np.abs(kx)) + 0 * kx)   # 0 — волна вдоль дуги
    nb = 18
    edges = np.linspace(0, 90, nb + 1)
    prof = np.array([P[band & (ang_k >= a) & (ang_k < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    prof = np.clip(prof - prof.min(), 0, None)                          # ровный фон шума
    mid = 0.5 * (edges[1:] + edges[:-1])
    # волна вдоль дуги (ang_k≈0) — это структура поперёк дуги, т. е. радиальная
    return float(prof[mid < 45].sum() / prof.sum())


# ---------------------------------------------------------------- связность
def critical_gap(mask, axis, um, rmax_um=60):
    dist = ndi.distance_transform_edt(~mask)
    for r in np.arange(0, rmax_um / um, 0.5):
        c = dist <= r
        if axis == 0:
            c[0, :] = True; c[-1, :] = True
        else:
            c[:, 0] = True; c[:, -1] = True
        lab, _ = ndi.label(c)
        a, b = (lab[0, :], lab[-1, :]) if axis == 0 else (lab[:, 0], lab[:, -1])
        if np.intersect1d(a[a > 0], b[b > 0]).size:
            return float(2 * r * um)
    return np.nan


def best_path(mask):
    cost = np.where(mask, 0.01, 1.0)
    mcp = MCP_Geometric(cost)
    H, W = mask.shape
    cum, _ = mcp.find_costs([(0, j) for j in range(W)], [(H - 1, j) for j in range(W)])
    j = int(np.argmin(cum[-1]))
    return float(cum[-1, j] / H), np.asarray(mcp.traceback((H - 1, j)))


# ---------------------------------------------------------------- всё вместе
def analyse(path, um=None, bar_um=None, outer="top", scales_um=(1.5, 3, 6, 12, 24), n_layers=10,
            line_um=2.5, crop=None, ro_mm=None, surfaces="mask", guess=None):
    g = read_gray(path)
    if crop:
        x0, x1, y0, y1 = crop
        g = g[y0:y1, x0:x1]
    if um is None:
        um = find_scale_bar(g, bar_um)
    if guess is not None:
        xs, yo, yi = surfaces_near(g, um, guess[0], guess[1], outer_top=(outer == "top"))
        mask = None
    elif surfaces == "outline":
        xs, yo, yi = surfaces_from_outline(g, um, outer_top=(outer == "top"))
        mask = None
    else:
        mask = wall_mask(g, um)
        xs, yo, yi = surface_points(mask, outer_top=(outer == "top"))
    S, inside, X, Y, geo = unwrap(g, xs, yo, yi, um, ro_px=None if ro_mm is None else ro_mm * 1000 / um)
    h = hydride_signal(S, um, line_um)
    h[~inside] = 0
    nr, nc = h.shape
    depth = (np.arange(nr) + 0.5) * um + 3.0                 # от наружной поверхности
    # маска гидридов: тёмные линии (Оцу по сигналу) без точек травления и обрывков короче 6 мкм;
    # на неё опираются доля площади и связность
    thr = threshold_otsu(h[inside])
    st0, _ = ridges(h, max(1.0, 1.0 / um))
    lines = drop_small(ridge_mask(st0) | ((h > thr) & ndi.binary_dilation(ridge_mask(st0), iterations=1)),
                       6.0 / um)
    hm = lines & (h > 0.5 * thr) & inside
    res = dict(file=os.path.basename(path), um_per_px=um, geometry=geo,
               strip_um=[nr * um, nc * um], area_fraction=float(hm.mean()))
    # ориентация по масштабам
    layers = np.array_split(np.arange(nr), n_layers)
    scale_rows = []
    sk_maps = {}
    for s_um in scales_um:
        s_px = s_um / um
        if s_px < 1.0 or 6 * s_px > min(nr, nc):
            continue
        st, ang = ridges(h, s_px)
        core = ndi.binary_erosion(inside, iterations=int(np.ceil(2.5 * s_px)), border_value=0)
        core[:, :int(np.ceil(2.5 * s_px))] = False; core[:, -int(np.ceil(2.5 * s_px)):] = False
        sk = ridge_skeleton(st, min_len_px=max(4, 2 * s_px)) & core
        sk_maps[s_um] = (sk, ang)
        fn, rhf = fn_from(ang[sk])
        prof = [fn_from(ang[rows][:, :][sk[rows]])[0] for rows in layers]
        scale_rows.append(dict(scale_um=s_um, Fn=fn, RHF_simon=rhf, length_mm_per_mm2=float(sk.sum() * um / (inside.sum() * um * um) * 1e3),
                               Fn_layers=prof))
    res["scales"] = scale_rows
    res["depth_layers_um"] = [float(depth[rows].mean()) for rows in layers]
    res["area_fraction_layers"] = [float(hm[rows].mean()) for rows in layers]
    # спектральная проверка по трём слоям (как у МИФИ: 1 — у наружной поверхности)
    thirds = np.array_split(np.arange(nr), 3)
    res["spectral_Fn_thirds"] = [spectral_fn(h[rows], um, 2.0, 30.0) for rows in thirds]
    res["skeleton_Fn_thirds"] = {str(s): [fn_from(sk_maps[s][1][rows][sk_maps[s][0][rows]])[0] for rows in thirds]
                                 for s in sk_maps}
    # связность
    sq = min(nr, nc)
    gaps_r, gaps_t = [], []
    for c0 in range(0, nc - sq + 1, max(sq // 2, 1)):
        win = hm[:, c0:c0 + sq]
        gaps_r.append(critical_gap(win, 0, um))
        gaps_t.append(critical_gap(win, 1, um))
    res["gap_through_wall_um"] = gaps_r
    res["gap_along_arc_um"] = gaps_t
    frac, path = best_path(hm)
    res["best_path_matrix_fraction"] = frac
    # то же, но «дешёвыми» считаются только радиальные участки (на мелком масштабе):
    # окружный гидрид не раскрывается окружным напряжением, по нему трещина толщину не проходит
    s_f = min(sk_maps) if sk_maps else 1.5
    _, ang_f = ridges(h, s_f / um)
    dev_f = np.minimum(ang_f, 180 - ang_f)
    hm_rad = hm & (dev_f > 45)
    frac_r, path_r = best_path(hm_rad)
    res["best_path_matrix_fraction_radial_only"] = frac_r
    res["gap_through_wall_radial_only_um"] = [critical_gap(hm_rad[:, c0:c0 + sq], 0, um)
                                              for c0 in range(0, nc - sq + 1, max(sq // 2, 1))]
    return res, dict(g=g, mask=mask, S=S, h=h, hm=hm, inside=inside, sk=sk_maps, path=path, path_r=path_r,
                     hm_rad=hm_rad, depth=depth,
                     xs=xs, yo=yo, yi=yi, um=um)


# ---------------------------------------------------------------- рисунок
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
LIGHT = "#a9a8a2"


def _hex(c):
    return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def overlay(S, sk, ang, inside):
    rgb = np.dstack([S] * 3) * 0.55 + 0.45
    rgb[~inside] = 1
    dev = np.minimum(ang, 180 - ang)
    thick = ndi.binary_dilation(sk, iterations=1)
    near = ndi.distance_transform_edt(~sk, return_indices=True)[1]
    dev_t = dev[near[0], near[1]]
    rgb[thick & (dev_t <= 45)] = _hex(BLUE)
    rgb[thick & (dev_t > 45)] = _hex(ORANGE)
    return rgb


def figure(res, D, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    um = D["um"]; S, inside = D["S"], D["inside"]
    nr, nc = S.shape
    ext = (0, nc * um, nr * um, 0)
    sc = sorted(D["sk"].keys())
    fine, coarse = sc[0], sc[-2] if len(sc) > 2 else sc[-1]
    fig = plt.figure(figsize=(14, 10.5), facecolor=BG)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], width_ratios=[1, 1, 0.9], hspace=0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(D["g"], cmap="gray")
    ax.plot(D["xs"], D["yo"], ".", ms=1, color=ORANGE); ax.plot(D["xs"], D["yi"], ".", ms=1, color=BLUE)
    ax.set_title("а) Исходный снимок и найденные поверхности", loc="left", fontsize=10, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(S, cmap="gray", extent=ext)
    ax.set_title("б) Стенка, развёрнутая в полосу", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("вдоль дуги, мкм"); ax.set_ylabel("от наружной поверхности, мкм")
    for k, s in enumerate((fine, coarse)):
        ax = fig.add_subplot(gs[1, k])
        sk, ang = D["sk"][s]
        ax.imshow(overlay(S, sk, ang, inside), extent=ext)
        fn = [r for r in res["scales"] if r["scale_um"] == s][0]["Fn"]
        ax.set_title(f"{'вг'[k]}) Масштаб {s:g} мкм: радиальных {fn:.2f}\n"
                     f"оранжевый — радиальные, синий — окружные", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("вдоль дуги, мкм")
        if k == 0: ax.set_ylabel("от наружной поверхности, мкм")
    # профили по толщине
    ax = fig.add_subplot(gs[0, 2])
    dl = np.array(res["depth_layers_um"])
    ax.plot(np.array(res["area_fraction_layers"]) * 100, dl, "o-", color=INK, lw=1.8, ms=4)
    ax.set_ylim(nr * um + 6, 0); ax.set_xlabel("доля площади гидридов, %"); ax.set_ylabel("от наружной поверхности, мкм")
    ax.set_title("д) Сколько гидридов по толщине", loc="left", fontsize=10, color=INK)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax = fig.add_subplot(gs[1, 2])
    cols = [BLUE, AQUA, YELLOW, ORANGE, INK]
    styles = ["-", (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1, 2)), "-"]
    for r, c, ls in zip(res["scales"], cols, styles):
        ax.plot(np.array(r["Fn_layers"]) * 100, dl, color=c, lw=2, ls=ls, label=f"{r['scale_um']:g} мкм")
    ax.set_ylim(nr * um + 6, 0); ax.set_xlim(0, 100)
    ax.axvline(50, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("доля радиальных, %"); ax.set_ylabel("от наружной поверхности, мкм")
    ax.set_title("е) Доля радиальных по толщине", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, title="масштаб", title_fontsize=8.5, loc="lower right")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax = fig.add_subplot(gs[2, 0])
    ss = [r["scale_um"] for r in res["scales"]]
    ax.plot(ss, [r["Fn"] * 100 for r in res["scales"]], "o-", color=BLUE, lw=2, ms=6)
    ax.set_xscale("log"); ax.set_xticks(ss, [f"{s:g}" for s in ss]); ax.minorticks_off()
    ax.set_ylim(0, 100); ax.axhline(50, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("масштаб σ, мкм"); ax.set_ylabel("доля радиальных, %")
    ax.set_title("ж) Доля радиальных по масштабу (вся стенка)", loc="left", fontsize=10, color=INK)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax = fig.add_subplot(gs[2, 1])
    base = np.where(D["hm"], 0.62, 0.98)
    base[D["hm_rad"]] = 0.12
    ax.imshow(base, cmap="gray", vmin=0, vmax=1, extent=ext)
    p = D["path"]; pr = D["path_r"]
    ax.plot(p[:, 1] * um, p[:, 0] * um, color=BLUE, lw=1.3, label=f"по любым гидридам: матрица {res['best_path_matrix_fraction'] * 100:.0f} %")
    ax.plot(pr[:, 1] * um, pr[:, 0] * um, color=ORANGE, lw=1.6,
            label=f"только по радиальным: матрица {res['best_path_matrix_fraction_radial_only'] * 100:.0f} %")
    ax.legend(frameon=True, fontsize=8, loc="lower left", framealpha=0.9)
    ax.set_title("з) Лучший путь трещины через стенку\n(чёрные — радиальные гидриды, серые — окружные)",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlabel("вдоль дуги, мкм"); ax.set_ylabel("от наружной поверхности, мкм")
    ax = fig.add_subplot(gs[2, 2])
    gr, gt = res["gap_through_wall_um"], res["gap_along_arc_um"]
    x = np.arange(len(gr))
    ax.bar(x - 0.18, gt, 0.34, color=BLUE, label="вдоль дуги")
    ax.bar(x + 0.18, gr, 0.34, color=ORANGE, label="через стенку")
    ax.set_xticks(x, [f"{i + 1}" for i in x]); ax.set_xlabel("квадратное окно №")
    ax.set_ylabel("зазор, мкм")
    ax.set_title("и) Какой зазор матрицы надо\nперепрыгнуть (меньше — связнее)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(out_png, dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--um-per-px", type=float)
    ap.add_argument("--bar-um", type=float, help="длина масштабной линейки, мкм (длина в пикселях ищется сама)")
    ap.add_argument("--outer", choices=["top", "bottom"], default="top")
    ap.add_argument("--scales", default="1.5,3,6,12,24")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--crop", help="x0,x1,y0,y1 — вырезать область до анализа (убрать подписи)")
    ap.add_argument("--ro-mm", type=float, help="известный наружный радиус, мм (иначе — по снимку)")
    ap.add_argument("--surface-guess", help="y_нар,y_вн — примерные строки поверхностей (пиксели); уточняются сами")
    ap.add_argument("--surfaces", choices=["mask", "outline"], default="mask",
                    help="mask — металл светлее/текстурнее заливки (оптика); outline — по тёмному контуру (СЭМ)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    if a.um_per_px is None and a.bar_um is None:
        raise SystemExit("нужен --um-per-px или --bar-um")
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    res, D = analyse(a.image, um=a.um_per_px, bar_um=a.bar_um, outer=a.outer,
                     scales_um=tuple(float(s) for s in a.scales.split(",")), n_layers=a.layers, crop=crop,
                     ro_mm=a.ro_mm, surfaces=a.surfaces,
                     guess=tuple(float(v) for v in a.surface_guess.split(",")) if a.surface_guess else None)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    with open(os.path.join(a.out, stem + "_metrics.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    figure(res, D, os.path.join(a.out, stem + "_report.png"), a.title or stem)
    print(json.dumps({k: res[k] for k in ("file", "um_per_px", "area_fraction", "best_path_matrix_fraction",
                                          "best_path_matrix_fraction_radial_only")}, ensure_ascii=False))
    for r in res["scales"]:
        print(f"  σ={r['scale_um']:>5g} мкм: Fn={r['Fn']:.2f}  RHF(Simon)={r['RHF_simon']:.2f}")


if __name__ == "__main__":
    main()
