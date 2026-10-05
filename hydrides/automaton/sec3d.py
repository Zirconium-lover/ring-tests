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


def section_plates(c, nv, R, zc, L):
    """Следы дисков в сечении z = zc: (ND, TD, ψ, полудлина) в координатах автомата (строки вниз)."""
    out = []
    for ci, ni, ri in zip(c, nv, R):
        sin_t = np.sqrt(max(1e-12, 1 - ni[2] ** 2))
        dz = (zc - ci[2] + L / 2) % L - L / 2
        rr = abs(dz) / sin_t
        if rr >= ri:
            continue
        half = np.sqrt(ri * ri - rr * rr)
        d = np.cross(ni, [0, 0, 1.0]); d /= np.linalg.norm(d)
        w = np.array([0, 0, 1.0]) - ni[2] * ni; w /= max(np.linalg.norm(w), 1e-9)
        p0 = ci + w * (dz / max(w[2], 1e-9))
        psi = np.arctan2(-d[1], d[0])                       # строки вниз: угол следа от TD «вверх»
        out.append((p0[1] % L, p0[0] % L, psi, half))
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


def rhf_sections(c, nv, R, L, n_sec=32, se=False):
    """Среднее RHF по изображениям n_sec равноотстоящих сечений (по одному сечению разброс ±0.15–0.25:
    следы дисков короче 2 мкм, поэтому сечений нужно много). se=True — ещё и стандартная ошибка."""
    v = np.array([rhf_image(section_plates(c, nv, R, zc, L), L) for zc in (np.arange(n_sec) + 0.5) * L / n_sec])
    m = float(np.nanmean(v))
    return (m, float(np.nanstd(v) / np.sqrt(np.isfinite(v).sum()))) if se else m
