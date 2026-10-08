"""«ДСК в модели»: охлаждение без нагрузки с разной скоростью — насколько выпадение отстаёт от TSSP.

Крутизна зарождения задаётся B и Δ0 (r = ν_c·exp(−B[(Δ0/Δ)² − 1]), крутизна 2B/Δ0 на МПа):
  острое — B = 80, Δ0 = 15 МПа (калибровка, ≈58 на °C);
  мягкое — B = 80, Δ0 = 380 МПа (CNT при γ = 0.18 Дж/м², ≈2.3 на °C).
Меры: температура первого зародыша против TSSP(H), отставание раствора от TSSP по ходу охлаждения
(c̄ − TSSP при T на 10, 30, 60 °C ниже начала), их зависимость от скорости. Сравнить потом с
Lacroix 2021 (TSSP — кинетическая линия) и Blackmur 2015.

python dsc_model.py папка [процессов] [варианты через запятую]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_kinetic import KParams, run_kinetic  # noqa: E402
from thermo import TSS, c_line, T_line  # noqa: E402

BASE = dict(H_ppm=178.0, T_max=415.0, app_dT=0.08, bias_dT=15.0, gb=True, gb_dT=1.5, grow_kin=True,
            free_z=True, halo=True, cross_tol=15.0, sigma_cap=1e9, sigma_app=0.0)
NUC = {"sharp": dict(B=80.0, Delta0=15.0), "soft": dict(B=80.0, Delta0=380.0),
       "sharp_ref": dict(B=80.0, Delta0=15.0, tssp_ref=True), "soft_ref": dict(B=80.0, Delta0=380.0, tssp_ref=True)}


def job(a):
    out, nuc, rate, seed = a
    fn = os.path.join(out, f"{nuc}_rate{rate}_seed{seed}.json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    p = KParams(**dict(BASE, **NUC[nuc], rate=rate, seed=seed))
    r = run_kinetic(p)
    h = r["hist"]
    L = TSS[p.lines]
    tssp = np.array([c_line(T, L["TSSP"]) for T in h["T"]])
    new = [q for q in r["plates"] if not q.get("init")]
    T_first = float(new[0]["T"]) if new else np.nan
    T_tssp = float(T_line(min(p.H_ppm, c_line(p.T_max, L["TSSD"])), L["TSSP"]))
    lag = {}
    for d in (10, 30, 60):
        i = np.searchsorted(-h["T"], -(T_tssp - d))
        if i < len(h["T"]):
            lag[str(d)] = float(h["c_mean"][i] - tssp[i])
    res = dict(nuc=nuc, rate=rate, seed=seed, T_first=T_first, T_tssp=T_tssp, undercool=T_tssp - T_first,
               lag=lag, n_plates=len(new), time_s=time.time() - t0,
               T=h["T"][::4].tolist(), c=h["c_mean"][::4].tolist(), tssp=tssp[::4].tolist())
    json.dump(res, open(fn, "w"))


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else list(NUC)
    jobs = [(out, n, r, s) for n in which for r in (0.3, 3.0, 30.0) for s in (1,)]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 1) as pool:
        list(pool.imap_unordered(job, jobs))
    print("готово", len(jobs))
