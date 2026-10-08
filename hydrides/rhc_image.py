"""RHC (RHCP, RHCF) на снимке сечения стенки — той же функцией, что и на модели (automaton/connectivity.rhc_mask).

Маска гидридов — как в hydride_spec.analyse (развёртка стенки, «чёрный цилиндр», порог по шуму-суррогату);
RHCP — путь трещины наименьшей цены от наружной поверхности к внутренней (гидрид в 50 раз дешевле металла),
RHCF — наибольшая протяжённость связного кластера по толщине. Считается по всей полосе и по окнам заданной
ширины вдоль дуги (среднее, разброс); также «только радиальные» (участки, отклонённые от дуги больше 45°).

python rhc_image.py снимок.jpg --um-per-px 0.862 [--win-um 240] [другие ключи hydride_spec]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "automaton"))
import hydride_spec as hs  # noqa: E402
from connectivity import rhc_mask  # noqa: E402


def rhc_of(hm, um, win_um):
    """Вся полоса; окна шириной win_um на всю толщину; квадратные окна win_um × win_um (как поле модели)."""
    full = rhc_mask(hm, um, periodic=False)
    nw = max(1, int(round(win_um / um)))
    vals = [rhc_mask(hm[:, c0:c0 + nw], um, periodic=False) for c0 in range(0, hm.shape[1] - nw + 1, max(1, nw // 2))]
    sq = [rhc_mask(hm[r0:r0 + nw, c0:c0 + nw], um, periodic=False)
          for r0 in range(0, hm.shape[0] - nw + 1, max(1, nw // 2)) for c0 in range(0, hm.shape[1] - nw + 1, max(1, nw // 2))]
    return dict(full=full, win=dict(RHCP=[v["RHCP"] for v in vals], RHCF=[v["RHCF"] for v in vals]),
                sq=dict(RHCP=[v["RHCP"] for v in sq], RHCF=[v["RHCF"] for v in sq]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img"); ap.add_argument("--um-per-px", type=float); ap.add_argument("--bar-um", type=float)
    ap.add_argument("--win-um", type=float, default=240.0); ap.add_argument("--crop")
    ap.add_argument("--out"); ap.add_argument("--scales", default="1.5,3")
    a = ap.parse_args()
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    res, D = hs.analyse(a.img, um=a.um_per_px, bar_um=a.bar_um, crop=crop, scales_um=tuple(float(v) for v in a.scales.split(",")))
    um = D["um"]; hm = D["hm"]; hr = D["hm_rad"]
    out = dict(file=os.path.basename(a.img), um_per_px=um, strip_um=res["strip_um"], area_fraction=res["area_fraction"],
               win_um=a.win_um, any=rhc_of(hm, um, a.win_um), radial_only=rhc_of(hr, um, a.win_um))
    for k in ("any", "radial_only"):
        f, w = out[k]["full"], out[k]["win"]
        print(f"{out['file']} ({k}): вся полоса RHCP {f['RHCP']:.2f}, RHCF {f['RHCF']:.2f} | окна {a.win_um:g} мкм: "
              f"RHCP {np.mean(w['RHCP']):.2f} ± {np.std(w['RHCP']):.2f}, RHCF {np.mean(w['RHCF']):.2f} ± {np.std(w['RHCF']):.2f} | "
              f"квадраты {a.win_um:g}²: RHCP {np.mean(out[k]['sq']['RHCP']):.2f} ± {np.std(out[k]['sq']['RHCP']):.2f}, "
              f"RHCF {np.mean(out[k]['sq']['RHCF']):.2f} ± {np.std(out[k]['sq']['RHCF']):.2f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
