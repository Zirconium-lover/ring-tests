"""Работа выделения радиальной и окружной пластинки в матрице Мизеса и Хилла под нагрузкой (σθ, σz).
Обобщённая плоская деформация (сечение r–θ), ячейка 20 × 20 мкм, шаг 0.1 мкм, пластинка 5 × 0.6 мкм.
python hill_runs.py папка [процессов]"""
import os
import sys
import json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hill_fft as hf  # noqa: E402

N, H, A_HALF, TH = (200, 200, 1), 0.1, 2.5, 0.6
# Хилл (1 = r, 2 = θ, 3 = z): изотропия и коэффициенты Massih для текстурованной оболочки
# (F = 0.956, G = 0.304, H = 0.240; F при (σθ−σz)²: радиальное направление самое прочное); α — доля анизотропии
HILL = {"iso": (0.5, 0.5, 0.5), "massih": (0.956, 0.304, 0.240)}
# (σθ, σz): одноосные и пороги Cinbiz (155 / 0, 110 / 0.57, 75 / 0.83)
LOADS = {"0": (0, 0), "U75": (75, 0), "U110": (110, 0), "U155": (155, 0), "U200": (200, 0), "U250": (250, 0),
         "P110": (110, 0.57 * 110), "B75": (75, 0.83 * 75)}


def hill_of(name, alpha=1.0):
    F, G, Hh = HILL[name]
    F, G, Hh = (0.5 + alpha * (v - 0.5) for v in (F, G, Hh))
    return hf.hill_P(F, G, Hh)


def job(a):
    out, hill, alpha, orient, lk = a
    name = f"{hill}{alpha:g}_{orient}_{lk}"
    fn = os.path.join(out, name + ".json")
    if os.path.exists(fn):
        return name
    hf.WORKERS = 1
    c = hf.Cell(N, H, P=hill_of(hill, alpha))
    normal = 1 if orient == "circ" else 0            # окружная: нормаль по r (y); радиальная: нормаль по θ (x)
    pl = hf.Plate(c, hf.rect_mask(N, H, A_HALF, TH, normal), hf.misfit(normal))
    sth, sz = LOADS[lk]
    hist = pl.run([sth, 0, sz, 0, 0, 0], nt=10)
    W = hist[-1]["W"]
    res = dict(hill=hill, alpha=alpha, orient=orient, load=lk, s_theta=sth, s_z=sz, W=W, g=W / hf.EPS_N,
               pl_area=float((pl.p > 1e-4).sum() * H * H), dissip=None, hist=hist)
    np.savez_compressed(os.path.join(out, name + ".npz"), p=pl.p.reshape(N[:2]).astype(np.float32),
                        sig=pl.sig.reshape((6,) + N[:2]).astype(np.float32))
    json.dump(res, open(fn, "w"))
    return name


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    jobs = [(out, h, 1.0, o, lk) for h in ("iso", "massih") for lk in LOADS for o in ("circ", "rad")
            if not (h == "iso" and o == "rad" and lk == "0")]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for nm in pool.imap_unordered(job, jobs):
            print(nm, flush=True)
    print("готово", len(jobs))
