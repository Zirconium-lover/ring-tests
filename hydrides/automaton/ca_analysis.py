"""Анализ результата автомата: пластинки, пакеты (кластеры), RHF как на снимке."""
import sys
import os
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hydride_spec as hs  # noqa: E402


def simon_w(dev):
    dev = np.asarray(dev, float)
    return np.where(dev <= 40, 0.0, np.where(dev < 65, 0.5, 1.0))


def plate_table(res):
    P = res["plates"]
    c = np.array([q["c"] for q in P])                      # (y, x) мкм
    psi = np.array([q["psi"] for q in P])                  # рад, от TD вверх
    half = np.array([q["half"] for q in P])
    t = np.stack([-np.sin(psi), np.cos(psi)], 1)           # (dy, dx)
    a = c - t * half[:, None]; b = c + t * half[:, None]
    dev = np.degrees(np.abs(np.arctan(np.tan(psi))))       # 0..90 от TD
    return dict(c=c, psi=psi, half=half, a=a, b=b, dev=dev, L=2 * half)


def _wrap(d, box):
    return d - box * np.round(d / box)


def _pt_seg(p, a, b, box):
    """Расстояние от точек p (N,2) до отрезков ab (M,2) с периодичностью; результат (N, M)."""
    ab = b - a
    ap = _wrap(p[:, None, :] - a[None, :, :], box)
    tt = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9)[None], 0, 1)
    d = ap - tt[..., None] * ab[None]
    return np.sqrt((d ** 2).sum(-1))


def clusters(res, gap=1.0):
    """Пакеты: пластинки, ближе друг к другу чем gap (мкм), с учётом периодичности."""
    T = plate_table(res)
    box = np.array(res["params"].size_um)
    a, b = T["a"], T["b"]
    D = np.minimum.reduce([_pt_seg(a, a, b, box), _pt_seg(b, a, b, box),
                           _pt_seg(a, a, b, box).T, _pt_seg(b, a, b, box).T])
    np.fill_diagonal(D, np.inf)
    i, j = np.where(D <= gap + res["params"].h_um)
    n = len(a)
    ncomp, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    out = []
    for k in range(ncomp):
        idx = np.where(lab == k)[0]
        # развернуть координаты кластера относительно первой пластинки (периодичность)
        ref = T["c"][idx[0]]
        pts = []
        for m in idx:
            for q in (a[m], b[m]):
                pts.append(ref + _wrap(q - ref, box))
        pts = np.array(pts)
        if len(idx) >= 2:
            cov = np.cov((pts - pts.mean(0)).T)
            w, v = np.linalg.eigh(cov)
            d = v[:, -1]                                   # (dy, dx)
            ang = np.degrees(np.arctan2(-d[0], d[1]))      # от TD вверх
            ang = (ang + 90) % 180 - 90
            proj = (pts - pts.mean(0)) @ d
            ext = proj.max() - proj.min()
            width = np.sqrt(max(w[0], 0)) * 4
        else:
            ang = np.degrees(T["psi"][idx[0]]); ext = T["L"][idx[0]]; width = res["params"].h_um
        out.append(dict(idx=idx, n=len(idx), ang=float(ang), dev=float(abs(ang)), ext=float(ext),
                        width=float(width), Lsum=float(T["L"][idx].sum())))
    return out, T


def packet_metrics(res, min_ext=15.0):
    C, T = clusters(res)
    big = [c for c in C if c["ext"] >= min_ext]
    tot = sum(c["Lsum"] for c in C)
    m = dict(n_plates=len(T["L"]), n_packets=len(big),
             frac_in_packets=sum(c["Lsum"] for c in big) / tot if tot else np.nan,
             plate_dev_median=float(np.median(T["dev"])),
             plate_RHF=float((T["L"] * simon_w(T["dev"])).sum() / T["L"].sum()))
    if big:
        ext = np.array([c["ext"] for c in big]); dev = np.array([c["dev"] for c in big])
        m["packet_RHF"] = float((ext * simon_w(dev)).sum() / ext.sum())
        m["packet_ext_median"] = float(np.median(ext))
        rad = [c for c in big if c["dev"] > 65]
        cir = [c for c in big if c["dev"] < 25]
        m["dev_in_radial"] = float(np.median(np.concatenate([T["dev"][c["idx"]] for c in rad]))) if rad else np.nan
        m["dev_in_circ"] = float(np.median(np.concatenate([T["dev"][c["idx"]] for c in cir]))) if cir else np.nan
        m["n_radial_packets"] = len(rad)
    else:
        m.update(packet_RHF=np.nan, packet_ext_median=np.nan, dev_in_radial=np.nan, dev_in_circ=np.nan,
                 n_radial_packets=0)
    return m, C, T


def render(res, um_out=0.2, blur_um=0.0):
    """Изображение «как на шлифе»: гидрид тёмный. Возвращает (картинка 0..1, мкм/пикс)."""
    hyd = res["hyd"]; dx = res["params"].dx
    z = dx / um_out
    img = ndi.zoom(hyd, z, order=1)
    if blur_um:
        img = ndi.gaussian_filter(img, blur_um / um_out)
    img = np.clip(img / max(img.max(), 1e-9) * 1.2, 0, 1)
    return 1 - 0.85 * img, um_out


def image_rhf(res):
    """RHF так же, как сверяли с MATLAB на СЭМ ×300: 0.65 мкм/пикс, размытие ~1 мкм, σ = 3 мкм."""
    g, um = render(res, um_out=0.65, blur_um=1.0)
    rng = np.random.default_rng(0)
    g = np.clip(g + rng.normal(0, 0.03, g.shape), 0, 1)      # лёгкий шум съёмки
    r, D = hs.analyse("ca", um=um, g=g, field=True, scales_um=(3.0,), line_um=5.0, n_layers=3)
    return r["scales"][0]["RHF_simon"] if r["scales"] else np.nan


def interdistance(res_or_hyd, dx, axis, blur_um=2.0, level=0.25, min_gap_um=6.0):
    """Медианное расстояние между соседними пакетами, как в табл. 3.1 Lepine:
    вдоль ND (axis=0) — между окружными пакетами, вдоль TD (axis=1) — между радиальными.
    Пакеты — участки размытой плотности гидрида выше доли level от её 99-го процентиля."""
    hyd = res_or_hyd["hyd"] if isinstance(res_or_hyd, dict) else res_or_hyd
    b = ndi.gaussian_filter(hyd.astype(float), blur_um / dx, mode="wrap")
    m = b > level * np.percentile(b, 99.5)
    gaps = []
    lines = m.T if axis == 0 else m          # перебираем столбцы (для ND) или строки (для TD)
    for ln in lines[:: max(1, int(2.0 / dx))]:
        idx = np.where(ln)[0]
        if idx.size == 0:
            continue
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        cen = np.array([r.mean() for r in runs]) * dx
        if len(cen) >= 2:
            g = np.diff(cen)
            gaps.extend(g[g >= min_gap_um])
    return float(np.median(gaps)) if gaps else np.nan, len(gaps)
