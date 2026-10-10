"""Морфология гидридов одним кодом на снимке опыта и на модели (calib_plan.md, п. 8): маска гидридов из конвейера
снимков (hydride_spec) при одном пикселе (3.5 мкм — как рис. 3–4 Плясова), затем
  skel_density — плотность длины скелета, мм/мм²;
  L_obj_w, L_obj_max — средняя по длине и наибольшая длина связного макрогидрида (по главной оси), мкм;
  L_seg_w — средняя по длине длина отрезка скелета между ветвлениями, мкм;
  spacing_r — среднее расстояние между гидридами вдоль радиуса (столбцы маски), мкм;
  small_density, small_frac — мелкие гидриды (короче 15 мкм): число на мм² и доля в длине;
  iso_density, iso_frac — «рой»: мелкие гидриды дальше 10 мкм от длинных (≥ 30 мкм) — число на мм² и доля мелких;
  ang_w, F20, Fn_seg — угол отрезков скелета к окружному направлению: средний по длине (°), доля длины ближе 20° и
  дальше 45° (Fn по маске 3.5 мкм/пикс — грубее Fn_lab, но тем же конвейером на снимке и на модели).
Опыт, кольца Э635 до и после (рис. 3 Плясова): L_obj_w 55–78, L_obj_max 175–311, L_seg_w 47–68, skel_density 10–18,
spacing_r 66–150. При κ = 1 модель: 23–31 / 50–66 — россыпь коротких гидридов."""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))


def morph(m, um, L_small=15.0, d_iso=10.0):
    from skimage.morphology import skeletonize
    m = np.asarray(m, bool)
    sk = skeletonize(m)
    A = m.size * um * um
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    seg = sk & ~ndi.binary_dilation(sk & (nb >= 3), structure=np.ones((3, 3)))

    def lens(mask):
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        L, ids, ang = [], [], []
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None:
                continue
            ys, xs = np.nonzero(lab[sl] == k)
            if len(ys) < 2:
                continue
            P = np.stack([ys, xs], 1).astype(float); P -= P.mean(0)
            _, v = np.linalg.eigh(P.T @ P); pr = P @ v[:, -1]
            L.append((pr.max() - pr.min() + 1) * um); ids.append(k)
            ang.append(np.degrees(np.arctan2(abs(v[0, -1]), abs(v[1, -1]))))   # к окружному направлению (столбцы)
        return np.array(L), lab, np.array(ids, int), np.array(ang)
    Lo, lab, ids, _ = lens(m); Ls, _, _, As = lens(seg)
    cross = (np.diff(m.astype(int), axis=0) == 1).sum()
    # «рой»: мелкие гидриды (короче L_small) — их число на мм² и доля в суммарной длине объектов (средняя по длине их
    # почти не видит); обособленные — дальше d_iso от длинных (≥ 2 L_small): в опыте мелкие куски лежат на линиях
    # (рваная линия), в россыпи — в матрице между линиями
    small = Lo < L_small if len(Lo) else np.zeros(0, bool)
    big = np.isin(lab, ids[Lo >= 2 * L_small]) if len(Lo) else np.zeros_like(m)
    dist = ndi.distance_transform_edt(~big) * um if big.any() else np.full(m.shape, np.inf)
    dmin = ndi.minimum(dist, lab, ids[small]) if small.any() else np.zeros(0)
    n_iso = int((np.atleast_1d(dmin) > d_iso).sum())
    return dict(area=float(m.mean()), skel_density=float(sk.sum() * um / A * 1000),
                small_density=float(small.sum() / A * 1e6), small_frac=float(Lo[small].sum() / Lo.sum()) if len(Lo) else np.nan,
                iso_density=float(n_iso / A * 1e6), iso_frac=float(n_iso / small.sum()) if small.any() else np.nan,
                L_obj_w=float((Lo ** 2).sum() / Lo.sum()) if len(Lo) else np.nan,
                L_obj_max=float(Lo.max()) if len(Lo) else np.nan,
                L_seg_w=float((Ls ** 2).sum() / Ls.sum()) if len(Ls) else np.nan,
                ang_w=float((Ls * As).sum() / Ls.sum()) if len(Ls) else np.nan,
                F20=float(Ls[As <= 20].sum() / Ls.sum()) if len(Ls) else np.nan,
                Fn_seg=float(Ls[As > 45].sum() / Ls.sum()) if len(Ls) else np.nan,
                spacing_r=float(m.shape[0] * m.shape[1] * um / max(cross, 1)))


def morph_model(hyd, dx, etch_um=2.2, um_px=3.5):
    """Морфология поля модели, снятого как снимок с пикселем um_px через конвейер снимков."""
    import hydride_spec as hs
    import rhc_model as RM
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, D = hs.analyse("модель", um=um_px, field=True, g=RM.as_micrograph(np.asarray(hyd, float), dx, etch_um, um_px),
                          scales_um=(2 * um_px, 4 * um_px))
    return morph(D["hm"], um_px)


def morph_image(path, um_px=3.5):
    """Морфология снимка опыта (стенка выпрямляется в полярные оси)."""
    import hydride_spec as hs
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, D = hs.analyse(path, um=um_px)
    return morph(D["hm"], um_px)


def morph_windows(path, win_um=240.0, um_px=3.5):
    """Морфология снимка опыта в окнах win_um (как поле модели): одна маска на весь снимок, затем окна без перекрытия.
    → {мера: [среднее по окнам, σ по окнам, число окон]} — цель того же оператора и окна, что у модели, и разброс
    одного окна (сравнивать с затравочным разбросом модели)."""
    import hydride_spec as hs
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, D = hs.analyse(path, um=um_px)
    hm = np.asarray(D["hm"], bool); w = int(round(win_um / um_px))
    rows = [morph(hm[y:y + w, x:x + w], um_px) for y in range(0, hm.shape[0] - w + 1, w) for x in range(0, hm.shape[1] - w + 1, w)]
    return {k: [float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows], ddof=1)), len(rows)]
            for k in rows[0]}


if __name__ == "__main__":       # python morph.py выход.json снимок.png ... — морфология снимков в окнах 240 мкм
    import json
    res = {os.path.basename(p).rsplit(".", 1)[0]: morph_windows(p) for p in sys.argv[2:]}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
