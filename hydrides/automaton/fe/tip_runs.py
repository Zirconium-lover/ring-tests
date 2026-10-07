"""Поля у кромки пластинки гидрида в упругопластичной матрице (Мизес) при одноосной и двухосной нагрузке:
есть ли механизм, чувствующий гидростатическое напряжение σ_h = (σθ + σz)/3 (пороги Cinbiz ложатся на
σ_h ≈ 52 МПа), а не нормальное напряжение на одиночной пластинке.

Диск радиусом 2.5 мкм и толщиной 0.6 мкм в ячейке 96³ с шагом 0.15 мкм (как hill3d_runs.py); радиальная
пластинка — нормаль по θ (x), окружная — по r (y). Сначала нагрузка, затем несоответствие 0 → 1.
Сохраняются поля σ (Мандел) и пластической деформации p — для σ_h, обогащения водородом
c/c0 = exp(V_H σ_h / RT) и работы следующей пластинки σ:ε*/ε_n в местах, где садится следующая в колоде.
python tip_runs.py папка случай[,случай...]   (случай = rad_U155, circ_P110, ...; нагрузки — hill_runs.LOADS)"""
import os
import sys
import json
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hill_fft as hf  # noqa: E402
from hill_runs import LOADS  # noqa: E402

N, H, A_HALF, TH = (96, 96, 96), 0.15, 2.5, 0.6

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for case in sys.argv[2].split(","):
        orient, lk = case.split("_")
        fn = os.path.join(out, case + ".npz")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        c = hf.Cell(N, H)                                    # Мизес, σ_y 350 МПа, упрочнение 200 МПа
        normal = 0 if orient == "rad" else 1
        pl = hf.Plate(c, hf.rect_mask(N, H, A_HALF, TH, normal), hf.misfit(normal))
        sth, sz = LOADS[lk]
        hist = pl.run([sth, 0, sz, 0, 0, 0], nt=10, verbose=False)
        np.savez_compressed(fn, sig=pl.sig.reshape((6,) + N).astype(np.float32),
                            p=pl.p.reshape(N).astype(np.float32), mask=pl.mask.reshape(N))
        json.dump(dict(orient=orient, load=lk, s_theta=sth, s_z=sz, W=hist[-1]["W"], hist=hist,
                       time_s=time.time() - t0), open(os.path.join(out, case + ".json"), "w"))
        print(case, "W", round(hist[-1]["W"], 2), "время", round(time.time() - t0), flush=True)
