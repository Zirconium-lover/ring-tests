"""Экранирование приложенного напряжения у гидридов (по МКЭ) в 2D-автомате: кривые RHF(σ) с ним и без,
те же параметры (β 0.12, потолок 90 МПа на ближнее поле, ℓ 35 мкм). Сверка с Cinbiz et al. 2016:
одноосный порог 155 ± 10 МПа (RHF₄₅ > 0.05), RHF₄₅ = 0 ниже 145 МПа и ≈ 1 выше 177 МПа.
python screen2d.py папка [процессов]"""
import os
import sys
import json
import itertools
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_hydride import Params, run  # noqa: E402
from ca_analysis import packet_metrics, image_rhf, plate_table  # noqa: E402

OUT = sys.argv[1]
SIG = [0, 50, 100, 125, 150, 175, 200, 250]
VAR = {"off": dict(), "on": dict(screen=True), "half": dict(screen=True, screen_scale=0.5)}


def job(a):
    var, s, seed = a
    fn = os.path.join(OUT, f"{var}_s{s}_seed{seed}.json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    r = run(Params(sigma_app=s, seed=seed, beta=0.12, sigma_cap=90.0, capture_um=35.0, cap_local_only=True, **VAR[var]))
    m, C, T = packet_metrics(r)
    m["RHF_image"] = image_rhf(r)
    m["RHF45"] = float((T["L"] * (T["dev"] > 45)).sum() / T["L"].sum())          # как у Cinbiz: доля длины под 45–135°
    m.update(var=var, sigma=s, seed=seed, time_s=time.time() - t0)
    json.dump(m, open(fn, "w"), default=float)
    print(var, s, seed, f"след {m['plate_RHF']:.2f} RHF45 {m['RHF45']:.2f} изобр {m['RHF_image']:.2f} {m['time_s']:.0f} с", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = list(itertools.product(VAR, SIG, [1, 2]))
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        list(pool.imap_unordered(job, jobs))
    print("готово", len(jobs))
