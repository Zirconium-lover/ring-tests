"""Коллективный эффект: работа выделения новой пластинки рядом с уже выпавшими (колода или цепочка) в матрице
Мизеса под одноосной и двухосной нагрузкой. Обобщённая плоская деформация (сечение r–θ, σz входит через ε_zz),
ячейка 30 × 30 мкм, шаг 0.1 мкм, пластинки 5 × 0.6 мкм.
Сначала нагрузка и выпадение старых пластинок (несоответствие 0 → 1), затем — новой; W новой сравнивается с W
одиночной пластинки при той же нагрузке.
Расположения (t — вдоль пластинки, n — по нормали; номер k = 0, 1 — старые, k = 2 — новая):
  single — только новая;  deck — колода: сдвиг (t, n) = (2.5, 1.2) мкм на шаг;  chain — цепочка: (5.8, 0.6) мкм.
Сверка с МКЭ (stack_fe.py): --fut-el — место новой пластинки во время выделения старых упругое (в CalculiX
материал между шагами не сменить), --h шаг сетки (ячейка та же 30 мкм), --cases deck_rad_U110,… — только эти
случаи, --fields — сохранить поля σ и p (npz).
python stack_runs.py папка [процессов] [--fut-el] [--h 0.05] [--cases …] [--fields]"""
import os
import sys
import json
import time
import itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hill_fft as hf  # noqa: E402
from hill_runs import LOADS  # noqa: E402

L_CELL, A_HALF, TH = 30.0, 2.5, 0.6
N, H = (300, 300, 1), 0.1
FUT_EL, FIELDS = False, False
STEP = {"deck": (2.5, 1.2), "chain": (5.8, 0.6)}


def mask_at(normal, ct, cn):
    x = (np.arange(N[0]) + 0.5) * H - N[0] * H / 2
    y = (np.arange(N[1]) + 0.5) * H - N[1] * H / 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    nrm, tan = (X, Y) if normal == 0 else (Y, X)
    return ((np.abs(nrm - cn) < TH / 2) & (np.abs(tan - ct) < A_HALF))[:, :, None]


def job(a):
    out, orient, conf, lk = a
    fn = os.path.join(out, f"{orient}_{conf}_{lk}.json")
    if FUT_EL or H != 0.1:
        fn = os.path.join(out, f"{conf}_{orient}_{lk}_h{H:g}" + ("_fe" if FUT_EL else "") + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    hf.WORKERS = 1
    c = hf.Cell(N, H)
    normal = 0 if orient == "rad" else 1
    e = hf.misfit(normal)
    sth, sz = LOADS[lk]
    Sig = [sth, 0, sz, 0, 0, 0]
    if conf == "single":
        new = mask_at(normal, 0.0, 0.0)
        pl = hf.Plate(c, new, e)
        hist = pl.run(Sig, nt=10)
        W_old = None
    else:
        st, sn = STEP[conf]
        old = mask_at(normal, -2 * st, -2 * sn) | mask_at(normal, -st, -sn)
        new = mask_at(normal, 0.0, 0.0)
        p1 = hf.Plate(c, old, e)
        if FUT_EL:
            p1.matrix = ~(old.reshape(-1) | new.reshape(-1))
        h1 = p1.run(Sig, nt=10)
        W_old = h1[-1]["W"]
        pl = hf.Plate(c, new, e)
        pl.u, pl.E, pl.ep, pl.p = p1.u, p1.E, p1.ep, p1.p
        pl.matrix = ~(old.reshape(-1) | new.reshape(-1))
        pl.eig0 = p1.eig
        hist = pl.run(Sig, nt=10)
    W = hist[-1]["W"]
    if FIELDS:
        np.savez_compressed(fn[:-5] + ".npz", sig=pl.sig.reshape((6,) + N[:2]).astype(np.float32),
                            p=pl.p.reshape(N[:2]).astype(np.float32), new=new[:, :, 0],
                            old=(old[:, :, 0] if conf != "single" else np.zeros(N[:2], bool)))
    json.dump(dict(orient=orient, conf=conf, load=lk, s_theta=sth, s_z=sz, W=W, g=W / hf.EPS_N, W_old=W_old,
                   s_plate=hist[-1]["s_plate"], h=H, fut_el=FUT_EL,
                   pl_area=float((pl.p > 1e-4).sum() * H * H), time_s=time.time() - t0), open(fn, "w"))
    print(orient, conf, lk, "g", round(W / hf.EPS_N, 1), "время", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("nproc", nargs="?", type=int, default=4)
    ap.add_argument("--fut-el", action="store_true"); ap.add_argument("--fields", action="store_true")
    ap.add_argument("--h", type=float, default=0.1); ap.add_argument("--cases", default="")
    a = ap.parse_args()
    out = a.out
    FUT_EL, FIELDS, H = a.fut_el, a.fields, a.h
    n_ = int(round(L_CELL / H)); N = (n_, n_, 1)
    os.makedirs(out, exist_ok=True)
    loads = ["0", "U75", "B75", "U110", "P110", "U155"]
    jobs = [(out, o, cf, lk) for cf, lk, o in itertools.product(["single", "deck", "chain"], loads, ["rad", "circ"])]
    if a.cases:
        jobs = [(out, o, cf, lk) for cf, o, lk in (c.split("_") for c in a.cases.split(","))]
    with Pool(a.nproc) as pool:
        list(pool.imap_unordered(job, jobs))
    print("готово", len(jobs))
