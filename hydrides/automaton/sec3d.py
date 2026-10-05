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
