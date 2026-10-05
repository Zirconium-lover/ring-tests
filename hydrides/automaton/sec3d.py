"""Сечения 3D-автомата ⊥ оси трубы как шлиф r–θ: следы дисков → картинка → RHF по изображению
(тот же рендер и анализ, что для 2D: 0.65 мкм/пикс, размытие 1 мкм, масштаб 3 мкм)."""
import os
import sys
import warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "fe"), os.path.join(HERE, "..")):
    if d not in sys.path:
        sys.path.insert(0, d)
import hydride_spec as hs  # noqa: E402
from gaps import render_plates  # noqa: E402

FIELD_UM = 144.0          # периодическое сечение размножается до поля не меньше этого


def section_plates(c, nv, R, zc, L, u=None, A=None, B=None):
    """Следы пластинок в сечении z = zc: (ND, TD, ψ, полудлина) в координатах автомата (строки вниз).
    Диски — c, nv, R; эллипсы (иглы вдоль ⟨11-20⟩) — ещё u, A, B."""
    from ca3d import plate_chord, plate_axes
    out = []
    for k in range(len(c)):
        q = dict(n=nv[k], R=R[k]) if u is None else dict(n=nv[k], u=u[k], A=A[k], B=B[k])
        uu, aa, bb = plate_axes(q)
        tr = plate_chord(c[k], nv[k], uu, aa, bb, zc, L)
        if tr is None:
            continue
        (x, y), d, half = tr
        psi = np.arctan2(-d[1], d[0])                       # строки вниз: угол следа от TD «вверх»
        out.append((y % L, x % L, psi, half))
    return np.array(out) if out else np.zeros((0, 4))


def rhf_image(P, L, tile=None):
    """RHF по картинке следов P (ND, TD, ψ, полудлина) периодического сечения L × L."""
    tile = tile or int(np.ceil(FIELD_UM / L - 1e-9))
    if len(P) and tile > 1:
        P = np.concatenate([P + np.array([a * L, b * L, 0, 0]) for a in range(tile) for b in range(tile)])
    g = render_plates(P, (L * tile, L * tile), um=0.65, h_um=0.6, blur_um=1.0, noise=0.03) / 255.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = hs.analyse("ca", um=0.65, g=g, field=True, scales_um=(3.0,), line_um=5.0, n_layers=3)
    return r["scales"][0]["RHF_simon"] if r["scales"] else np.nan


def rhf_sections(c, nv, R, L, n_sec=32, se=False, u=None, A=None, B=None):
    """Среднее RHF по изображениям n_sec равноотстоящих сечений (по одному сечению разброс ±0.15–0.25:
    следы дисков короче 2 мкм, поэтому сечений нужно много). se=True — ещё и стандартная ошибка."""
    v = np.array([rhf_image(section_plates(c, nv, R, zc, L, u, A, B), L) for zc in (np.arange(n_sec) + 0.5) * L / n_sec])
    m = float(np.nanmean(v))
    return (m, float(np.nanstd(v) / np.sqrt(np.isfinite(v).sum()))) if se else m


def hyd_from_cells(n, cells, cfr):
    """Поле доли гидрида по сохранённым клеткам отрезков."""
    h = np.bincount(cells, weights=cfr, minlength=n ** 3)[: n ** 3]
    return np.clip(h, 0, 1).reshape(n, n, n)


def rhf_voxel(hyd, dx, n_sec=32, se=False):
    """RHF по изображениям сечений z = const поля гидрида (как image_rhf в 2D: масштабирование до
    0.65 мкм/пикс, размытие 1 мкм, шум 0.03). Периодическое сечение размножается до поля ≥ FIELD_UM."""
    from scipy import ndimage as ndi
    n = hyd.shape[0]
    L = n * dx
    tile = int(np.ceil(FIELD_UM / L - 1e-9))
    rng = np.random.default_rng(0)
    v = []
    for k in ((np.arange(n_sec) + 0.5) * n / n_sec).astype(int):
        sl = np.tile(hyd[:, :, k].T, (tile, tile))              # строки — ND, столбцы — TD
        im = ndi.gaussian_filter(ndi.zoom(sl, dx / 0.65, order=1), 1.0 / 0.65)
        if im.max() <= 0:
            v.append(np.nan)
            continue
        g = np.clip(1 - 0.85 * np.clip(im / im.max() * 1.2, 0, 1) + rng.normal(0, 0.03, im.shape), 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = hs.analyse("ca", um=0.65, g=g, field=True, scales_um=(3.0,), line_um=5.0, n_layers=3)
        v.append(r["scales"][0]["RHF_simon"] if r["scales"] else np.nan)
    v = np.array(v)
    m = float(np.nanmean(v))
    return (m, float(np.nanstd(v) / np.sqrt(np.isfinite(v).sum()))) if se else m


def section_traces(z, zc, L, dx=0.5):
    """Следы пластинок в сечении z = zc как 2D-пластинки (ND, TD, ψ, полудлина). z — словарь/npz прогона:
    отрезки {10-17} — по их клеткам (cells, cfr, cptr), диски и иглы — аналитически."""
    if "cells" not in z:
        ell = {k: z[k] for k in ("u", "A", "B") if k in z}
        return section_plates(z["c"], z["n"], z["R"], zc, L, **ell)
    n = int(round(L / dx))
    k = int(zc / dx) % n
    cells, ptr = z["cells"], z["cptr"]
    ix, iy, iz = np.unravel_index(cells, (n, n, n))
    out = []
    for m in range(len(ptr) - 1):
        sl = slice(ptr[m], ptr[m + 1])
        sel = iz[sl] == k
        if sel.sum() == 0:
            continue
        nv = z["n"][m]
        d = np.cross(nv, [0, 0, 1.0]); nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        d = d[:2] / nd
        c = z["c"][m][:2]
        pts = (np.stack([ix[sl][sel], iy[sl][sel]], 1) + 0.5) * dx
        pts = c + ((pts - c + L / 2) % L - L / 2)             # развернуть относительно центра
        s = (pts - c) @ d
        perp = (pts - c) @ np.array([-d[1], d[0]])
        smin, smax = s.min() - dx / 2, s.max() + dx / 2
        mid = c + 0.5 * (smin + smax) * d + perp.mean() * np.array([-d[1], d[0]])
        psi = np.arctan2(-d[1], d[0])
        out.append((mid[1] % L, mid[0] % L, psi, 0.5 * (smax - smin)))
    return np.array(out) if out else np.zeros((0, 4))


def packet_stats(z, L, dx=0.5, n_sec=16, h_um=0.6, min_ext=15.0, gap=1.0):
    """Пакеты на сечениях r–θ тем же способом, что в 2D (ca_analysis.clusters): следы ближе gap мкм —
    один пакет; пакет — от min_ext мкм. RHF пакетов (вес — длина пакета), RHF следов, медианный
    угол следов в радиальных (> 65°) и окружных (< 25°) пакетах, длина пакетов."""
    from types import SimpleNamespace
    from ca_analysis import clusters, simon_w
    ext, dev, plate_L, plate_dev, d_rad, d_cir = [], [], [], [], [], []
    for zc in (np.arange(n_sec) + 0.5) * L / n_sec:
        P = section_traces(z, zc, L, dx)
        if len(P) < 2:
            continue
        res = dict(plates=[dict(c=np.array(p[:2]), psi=p[2], half=p[3]) for p in P],
                   params=SimpleNamespace(size_um=(L, L), h_um=h_um))
        C, T = clusters(res, gap=gap)
        plate_L.append(T["L"]); plate_dev.append(T["dev"])
        for c in C:
            if c["ext"] >= min_ext:
                ext.append(c["ext"]); dev.append(c["dev"])
                if c["dev"] > 65:
                    d_rad.append(T["dev"][c["idx"]])
                elif c["dev"] < 25:
                    d_cir.append(T["dev"][c["idx"]])
    plate_L, plate_dev = np.concatenate(plate_L), np.concatenate(plate_dev)
    ext, dev = np.array(ext), np.array(dev)
    return dict(packet_RHF=float((ext * simon_w(dev)).sum() / ext.sum()) if len(ext) else np.nan,
                plate_RHF=float((plate_L * simon_w(plate_dev)).sum() / plate_L.sum()),
                packet_ext_median=float(np.median(ext)) if len(ext) else np.nan,
                n_packets=int(len(ext)), frac_in_packets=np.nan,
                dev_in_radial=float(np.median(np.concatenate(d_rad))) if d_rad else np.nan,
                dev_in_circ=float(np.median(np.concatenate(d_cir))) if d_cir else np.nan)


def packets3d(hyd, dx, level=0.2, gap_um=1.0, min_vol_um3=5.0):
    """Пакеты в 3D: связные области гидрида, где соседние пластинки ближе gap_um (как в 2D), с учётом
    периодичности. Для каждого — протяжённость по TD (x), ND (y) и оси трубы (z), мкм, объём и средний
    угол следа пластинок не считается (это делает packet_stats на сечениях)."""
    from scipy import ndimage as ndi
    n = hyd.shape[0]
    m = hyd > level
    r = max(1, int(round(gap_um / 2 / dx)))
    md = ndi.binary_dilation(m, iterations=r)
    lab, k = ndi.label(md, structure=np.ones((3, 3, 3), bool))
    # склейка меток через периодические грани
    parent = np.arange(k + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for ax in range(3):
        a = np.take(lab, 0, axis=ax).ravel(); b = np.take(lab, n - 1, axis=ax).ravel()
        for x, y in set(zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)])):
            fx, fy = find(x), find(y)
            if fx != fy:
                parent[fy] = fx
    root = np.array([find(i) for i in range(k + 1)])
    lab = root[lab] * m
    out = []
    idx = np.nonzero(lab)
    labs = lab[idx]
    order = np.argsort(labs)
    labs = labs[order]
    pts = np.stack(idx, 1)[order]
    w = hyd[idx][order]
    cuts = np.nonzero(np.diff(labs))[0] + 1
    for P, W in zip(np.split(pts, cuts), np.split(w, cuts)):
        vol = W.sum() * dx ** 3
        if vol < min_vol_um3:
            continue
        d = (P - P[0] + n // 2) % n - n // 2                    # развернуть относительно первой клетки
        ext = (d.max(0) - d.min(0) + 1) * dx
        out.append(dict(vol=float(vol), ext_TD=float(ext[0]), ext_ND=float(ext[1]), ext_L=float(ext[2])))
    return out


def hyd_of(z, L, dx=0.5, h_um=0.6):
    """Поле доли гидрида прогона: по клеткам отрезков или по геометрии дисков и игл."""
    from ca3d import plate_fraction
    n = int(round(L / dx))
    if "cells" in z:
        return hyd_from_cells(n, z["cells"], z["cfr"])
    h = np.zeros((n, n, n))
    for k in range(len(z["c"])):
        q = dict(n=z["n"][k], R=z["R"][k])
        if "u" in z:
            q.update(u=z["u"][k], A=z["A"][k], B=z["B"][k])
        ax, fr = plate_fraction(n, dx, z["c"][k], q, h_um)
        h[np.ix_(*ax)] += fr
    return np.clip(h, 0, 1)
