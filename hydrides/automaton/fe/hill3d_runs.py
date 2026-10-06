"""3D-проверка: работа выделения дисковой пластинки (радиус 2.5 мкм, толщина 0.6) в матрице Мизеса и Хилла.
Ячейка 96³ с шагом 0.15 мкм. python hill3d_runs.py папка случай[,случай...]  (случай = hill_orient_load)"""
import os
import sys
import json
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hill_fft as hf  # noqa: E402
from hill_runs import hill_of, LOADS  # noqa: E402

N, H, A_HALF, TH = (96, 96, 96), 0.15, 2.5, 0.6

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for case in sys.argv[2].split(","):
        hill, orient, lk = case.split("_")
        fn = os.path.join(out, case + ".json")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        c = hf.Cell(N, H, P=hill_of(hill))
        normal = 1 if orient == "circ" else 0
        pl = hf.Plate(c, hf.rect_mask(N, H, A_HALF, TH, normal), hf.misfit(normal))
        sth, sz = LOADS[lk]
        hist = pl.run([sth, 0, sz, 0, 0, 0], nt=10, verbose=True)
        W = hist[-1]["W"]
        res = dict(hill=hill, alpha=1.0, orient=orient, load=lk, s_theta=sth, s_z=sz, W=W, g=W / hf.EPS_N,
                   pl_vol=float((pl.p > 1e-4).sum() * H ** 3), n_plate=int(pl.mask.sum()), hist=hist, time_s=time.time() - t0)
        json.dump(res, open(fn, "w"))
        print(case, "g", round(res["g"], 1), "время", round(res["time_s"]), flush=True)
