"""Разрывы между пластинками вдоль пакетов: снимки Cinbiz и картинки автомата, одним способом.

Ось пакета — хребет на масштабе 4 мкм (как при поиске пакетов); гидрид — хребты на масштабе
0.6 мкм выше порога фазовых суррогатов. Точка оси «покрыта», если гидрид ближе r_tol.
Разрыв — непрерывный непокрытый участок оси; его длина ≈ видимый промежуток минус 2·r_tol.
Ореол из МКЭ (σ_y = 150–400 МПа) запрещает зарождение ближе ~1–3 мкм к кончику и даёт
самое выгодное место в 2.5–5 мкм перед ним — значит, в пакетах были бы разрывы в несколько мкм.
"""
import os
import sys
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, disk
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import hydride_spec as hs  # noqa: E402


def render_plates(P, size_um, um=0.2, h_um=0.6, shrink_um=0.0, blur_um=0.3, noise=0.03, seed=0):
    """Картинка «как на шлифе» прямо по таблице пластинок (cy, cx, ψ, полудлина); shrink_um
    укорачивает каждую пластинку с обоих концов — искусственные разрывы +2·shrink."""
    ny, nx = int(size_um[0] / um), int(size_um[1] / um)
    img = np.zeros((ny, nx))
    for cy, cx, psi, half in P:
        half = half - shrink_um
        if half <= 0.2:
            continue
        t = np.array([-np.sin(psi), np.cos(psi)])           # (строка, столбец)
        n = np.array([t[1], -t[0]])
        r = half + h_um
        y0, y1 = int((cy - r) / um), int((cy + r) / um) + 2
        x0, x1 = int((cx - r) / um), int((cx + r) / um) + 2
        Y, X = np.mgrid[y0:y1, x0:x1]
        s = ((Y + 0.5) * um - cy) * t[0] + ((X + 0.5) * um - cx) * t[1]
        q = ((Y + 0.5) * um - cy) * n[0] + ((X + 0.5) * um - cx) * n[1]
        m = (np.abs(s) <= half) & (np.abs(q) <= h_um / 2)
        img[Y[m] % ny, X[m] % nx] = 1.0
    if blur_um:
        img = ndi.gaussian_filter(img, blur_um / um)
    img = np.clip(img / max(img.max(), 1e-9), 0, 1)
    g = 1 - 0.7 * img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(g, 0, 1) * 255.0


def measure(g, um, r_tol_um=0.6):
    """Возвращает список (длина разрыва, класс пакета) и покрытие осей окружных и радиальных пакетов."""
    h = hs.hydride_signal(g / 255.0, um, 2.0, denoise_um=0.25)
    surr = hs.phase_surrogates(h)
    sp = 4.0 / um
    stP, angP = hs.ridges(h, sp)
    pk = hs.drop_small(stP > hs.null_threshold(surr, sp, 99.0), 3 * sp)
    skP = skeletonize(pk)
    sf = 0.6 / um
    stF, _ = hs.ridges(h, sf)
    pl = hs.drop_small(stF > hs.null_threshold(surr, sf, 99.0), 1.5 / um)
    near = ndi.binary_dilation(pl, structure=disk(max(1, int(round(r_tol_um / um)))))
    pdev = np.minimum(angP, 180 - angP)
    out = {}
    for cls, sel in (("окр", pdev < 30), ("рад", pdev > 60)):
        ax_ = skP & sel
        if ax_.sum() < 20:
            out[cls] = dict(cov=np.nan, L=0.0, gaps=np.zeros(0))
            continue
        lab, n = ndi.label(ax_ & ~near, structure=np.ones((3, 3)))
        gaps = ndi.sum(np.ones(lab.shape), lab, index=np.arange(1, n + 1)) * um * 1.1 if n else np.zeros(0)
        out[cls] = dict(cov=float(near[ax_].mean()), L=float(ax_.sum() * um * 1.1), gaps=np.asarray(gaps))
    return out


def measure_dark(g, um, r_um=0.6, pct=97.0):
    """То же, но «покрыто» — по потемнению: среднее затемнение в круге r_um на оси пакета выше
    pct-го процентиля затемнения матрицы вдали (> 6 мкм) от пакетов. Плотные стопки, которые
    детектор тонких линий не разделяет, так считаются сплошными."""
    h = hs.hydride_signal(g / 255.0, um, 2.0, denoise_um=0.25)
    surr = hs.phase_surrogates(h)
    sp = 4.0 / um
    stP, angP = hs.ridges(h, sp)
    pk = hs.drop_small(stP > hs.null_threshold(surr, sp, 99.0), 3 * sp)
    skP = skeletonize(pk)
    dark = g.mean() - g
    dark = dark - ndi.gaussian_filter(dark, 30.0 / um)
    k = disk(max(1, int(round(r_um / um)))).astype(float); k /= k.sum()
    dl = ndi.convolve(dark, k, mode="reflect")
    far = ~ndi.binary_dilation(pk, structure=disk(int(6.0 / um)))
    thr = np.percentile(dl[far], pct)
    near = dl > thr
    pdev = np.minimum(angP, 180 - angP)
    out = {}
    for cls, sel in (("окр", pdev < 30), ("рад", pdev > 60)):
        ax_ = skP & sel
        if ax_.sum() < 20:
            out[cls] = dict(cov=np.nan, L=0.0, gaps=np.zeros(0))
            continue
        lab, n = ndi.label(ax_ & ~near, structure=np.ones((3, 3)))
        gaps = ndi.sum(np.ones(lab.shape), lab, index=np.arange(1, n + 1)) * um * 1.1 if n else np.zeros(0)
        out[cls] = dict(cov=float(near[ax_].mean()), L=float(ax_.sum() * um * 1.1), gaps=np.asarray(gaps))
    out["_maps"] = (skP, near, pdev)
    return out


def summary(out):
    s = {}
    for cls, d in out.items():
        if cls.startswith("_"):
            continue
        gp = d["gaps"]
        s[cls] = dict(cov=d["cov"], L=d["L"], n_per100=100 * len(gp) / d["L"] if d["L"] else np.nan,
                      big_per100=100 * np.sum(gp >= 1.5) / d["L"] if d["L"] else np.nan,
                      p90=float(np.percentile(gp, 90)) if len(gp) else 0.0)
    return s
