"""Морфология гидридов одним кодом на снимке опыта и на модели (calib_plan.md, п. 8): маска гидридов из конвейера
снимков (hydride_spec) при одном пикселе (3.5 мкм — как рис. 3–4 Плясова), затем
  skel_density — плотность длины скелета, мм/мм²;
  L_obj_w, L_obj_max — средняя по длине и наибольшая длина связного макрогидрида (по главной оси), мкм;
  L_seg_w — средняя по длине длина отрезка скелета между ветвлениями, мкм;
  spacing_r — среднее расстояние между гидридами вдоль радиуса (столбцы маски), мкм.
Опыт, кольца Э635 до и после (рис. 3 Плясова): L_obj_w 55–78, L_obj_max 175–311, L_seg_w 47–68, skel_density 10–18,
spacing_r 66–150. При κ = 1 модель: 23–31 / 50–66 — россыпь коротких гидридов."""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))


def morph(m, um):
    from skimage.morphology import skeletonize
    m = np.asarray(m, bool)
    sk = skeletonize(m)
    A = m.size * um * um
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    seg = sk & ~ndi.binary_dilation(sk & (nb >= 3), structure=np.ones((3, 3)))

    def lens(mask):
        lab, n = ndi.label(mask, structure=np.ones((3, 3)))
        L = []
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None:
                continue
            ys, xs = np.nonzero(lab[sl] == k)
            if len(ys) < 2:
                continue
            P = np.stack([ys, xs], 1).astype(float); P -= P.mean(0)
            _, v = np.linalg.eigh(P.T @ P); pr = P @ v[:, -1]
            L.append((pr.max() - pr.min() + 1) * um)
        return np.array(L)
    Lo = lens(m); Ls = lens(seg)
    cross = (np.diff(m.astype(int), axis=0) == 1).sum()
    return dict(area=float(m.mean()), skel_density=float(sk.sum() * um / A * 1000),
                L_obj_w=float((Lo ** 2).sum() / Lo.sum()) if len(Lo) else np.nan,
                L_obj_max=float(Lo.max()) if len(Lo) else np.nan,
                L_seg_w=float((Ls ** 2).sum() / Ls.sum()) if len(Ls) else np.nan,
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
