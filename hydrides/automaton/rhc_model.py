"""RHC модели «как на оптике»: поле гидрида → оптическое изображение (размытие 1 мкм, порог) → утолщение травлением
etch_um → маска → RHCP, RHCF (connectivity.rhc_mask) — той же функцией, что и снимки (../rhc_image.py).

Утолщение травлением — свойство съёмки, а не модели: подбирается один раз, чтобы доля площади гидридов модели без
нагрузки совпала с долей на снимке до опыта (Э635, снимки 1–2: 11–12 %), и дальше одно для всех состояний.
«Только радиальные» — поле из пластинок со следом больше 45° от дуги.

python rhc_model.py папка [--etch-um 1.5] [--fit-area 0.115]   — сводка по расчётам папки
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from connectivity import field_from_plates, rhc_mask  # noqa: E402

UM = 0.5                                 # пиксель «снимка», мкм


def optical(hyd, dx, etch_um, blur_um=1.0, level=0.25):
    """Маска гидридов «как на шлифе»: размытие, порог, затем утолщение травлением (диск радиуса etch_um)."""
    z = dx / UM
    img = ndi.zoom(hyd, z, order=1)
    img = ndi.gaussian_filter(img, blur_um / UM, mode="wrap")
    img = np.clip(img / max(img.max(), 1e-9) * 1.2, 0, 1)
    m = img > level
    r = int(round(etch_um / UM))
    if r > 0:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m = ndi.binary_dilation(m, structure=(yy ** 2 + xx ** 2) <= r * r)
    return m


def load(fn, size_default=(240.0, 240.0), dx=0.4):
    m = json.load(open(fn))
    z = np.load(fn[:-5] + ".npz")
    size = tuple(m.get("size_um", size_default))
    if "hyd" in z.files:
        hyd = np.asarray(z["hyd"], float); dx = float(z["dx"])
    else:
        hyd = None
    P = z["plates"]
    shape = (int(round(size[0] / dx)), int(round(size[1] / dx)))
    if hyd is None:
        hyd = field_from_plates(P, shape, dx)
    rad = P[np.abs(np.sin(P[:, 2])) >= np.sin(np.radians(45))] if len(P) else P
    hyd_r = field_from_plates(rad, shape, dx)
    return m, hyd, hyd_r, dx


def as_micrograph(hyd, dx, etch_um, um_px, seed=0):
    """Модель как серый снимок с пикселем um_px: оптическая маска с утолщением травлением, усреднение по пикселю
    снимка, светлый металл и тёмный гидрид, лёгкий шум съёмки — дальше та же обработка, что у снимков."""
    m = optical(hyd, dx, etch_um).astype(float)
    k = um_px / UM
    img = ndi.zoom(m, 1.0 / k, order=1) if k > 1 else m
    rng = np.random.default_rng(seed)
    return np.clip(0.88 - 0.7 * img + rng.normal(0, 0.03, img.shape), 0, 1)


def rhc_spec(hyd, dx, etch_um, um_px):
    """RHC модели через конвейер снимков (hydride_spec, поле без поверхностей) — для сравнения со снимком того же
    разрешения."""
    sys.path.insert(0, os.path.dirname(HERE))
    import hydride_spec as hs
    g = as_micrograph(hyd, dx, etch_um, um_px)
    s1 = max(1.5, 2 * um_px)
    res, D = hs.analyse("модель", um=um_px, field=True, g=g, scales_um=(s1, 2 * s1))
    a = rhc_mask(D["hm"], um_px, periodic=False); b = rhc_mask(D["hm_rad"], um_px, periodic=False)
    return dict(area=float(D["hm"].mean()), RHCP=a["RHCP"], RHCF=a["RHCF"], RHCP_rad=b["RHCP"], RHCF_rad=b["RHCF"])


def rhc_of(hyd, hyd_r, dx, etch_um):
    m = optical(hyd, dx, etch_um); mr = optical(hyd_r, dx, etch_um)
    a = rhc_mask(m, UM); b = rhc_mask(mr, UM)
    return dict(area=float(m.mean()), RHCP=a["RHCP"], RHCF=a["RHCF"], RHCP_rad=b["RHCP"], RHCF_rad=b["RHCF"])


def fit_etch(files, target, grid=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """Утолщение, при котором средняя доля площади по расчётам без нагрузки равна target."""
    areas = []
    for e in grid:
        areas.append(np.mean([optical(load(f)[1], load(f)[3], e).mean() for f in files]))
    return float(np.interp(target, areas, grid)), list(zip(grid, areas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--etch-um", type=float)
    ap.add_argument("--fit-area", type=float, default=0.115)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.folder, "*.json")))
    files = [f for f in files if os.path.exists(f[:-5] + ".npz")]
    etch = a.etch_um
    if etch is None:
        zero = [f for f in files if float(json.load(open(f)).get("sigma_app", -1)) == 0.0]
        etch, tab = fit_etch(zero, a.fit_area)
        print("утолщение травлением по доле площади без нагрузки:", " ".join(f"{e:g} мкм→{v:.3f}" for e, v in tab),
              f"→ {etch:.2f} мкм")
    rows = []
    for f in files:
        m, hyd, hyd_r, dx = load(f)
        r = rhc_of(hyd, hyd_r, dx, etch)
        r.update({k: m.get(k) for k in ("bias_dT", "sigma_app", "H_ppm", "ring", "F_l_5")})
        rows.append(r)
        print(f"{os.path.basename(f)[:70]:70s} площадь {r['area']:.3f}  RHCP {r['RHCP']:.2f}  RHCF {r['RHCF']:.2f}  "
              f"| радиальные: RHCP {r['RHCP_rad']:.2f}  RHCF {r['RHCF_rad']:.2f}  | F_l {r['F_l_5']}")
    json.dump(dict(etch_um=etch, rows=rows), open(os.path.join(a.folder, "rhc_model.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
