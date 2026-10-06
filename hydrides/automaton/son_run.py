"""Расчёт под условия опыта Son и др. 2026 (Materials Science & Engineering A 977, 151019): трубы Zr-Nb,
187 ppm, нагрев до 400 °C, охлаждение 0.3 °C/мин под окружным напряжением 0 / 83 / 96 / 146 МПа.
Мишени: RHF по оптическому снимку (CWSR 0 / 3.8 / 10.7 / 45.4 %, PRXA 0 / 11.5 / 26.3 / 54.0 %) и доля
длины межзёренных гидридов (CWSR 92 %, PRXA 68 %, у радиальных PRXA 62 %).
Варианты: фора зёрен из опыта Vizcaíno 2014 на Zr-2.5Nb (5 °C) — без подгонки; фора по калибровке на Zry-4
(10.9 °C); первый вариант с ростом во времени и переходом через границы.
python son_run.py папка [процессов]"""
import os
import sys
import json
import time
import itertools
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool import MATERIALS, Texture, History, Model, simulate, _plates_arr  # noqa: E402
from calib_beta import metrics  # noqa: E402

SIG = [0, 83, 96, 120, 146, 175]
SEEDS = [1, 2]
KIN = dict(engine="kin", B=80.0, Delta0=15.0, sigma_cap=180.0, app_dT=0.08, cap_local_only=False, gb=True, gb_dT=1.5)
VARIANTS = {
    "viz5": dict(bias_dT=5.0),
    "zry10.9": dict(bias_dT=10.9),
    "viz5_grow": dict(bias_dT=5.0, grow_kin=True, cross_tol=15.0),
}
SON_CWSR = {0: 0.0, 83: 0.038, 96: 0.107, 146: 0.454}
SON_PRXA = {0: 0.0, 83: 0.115, 96: 0.263, 146: 0.540}


def job(a):
    out, var, sig, seed = a
    fn = os.path.join(out, f"{var}_s{sig}_seed{seed}")
    if os.path.exists(fn + ".json"):
        return
    t0 = time.time()
    model = Model(**dict(KIN, **VARIANTS[var]))
    r = simulate(MATERIALS["ZrNb_CWSR"], Texture(chi0=30.0), History(H_ppm=187.0, T_max=400.0, sigma=float(sig), rate=0.3),
                 model=model, seed=seed, metrics=False, cache=out)
    m = metrics(r)
    new = [q for q in r["plates"] if not q.get("init")]
    L = np.array([2 * q["half"] for q in new]); gbk = np.array([q.get("kind") == "gb" for q in new], bool)
    rad = np.array([abs(np.sin(q["psi"])) >= np.sin(np.radians(45)) for q in new], bool)
    m["GB_frac"] = float(L[gbk].sum() / L.sum()) if len(L) else np.nan
    m["GB_frac_rad"] = float(L[gbk & rad].sum() / L[rad].sum()) if rad.any() else np.nan
    m["T_first"] = new[0]["T"] if new else np.nan
    m["n_init"] = sum(1 for q in r["plates"] if q.get("init"))
    m.update(variant=var, sigma_app=sig, seed=seed, time_s=time.time() - t0, **VARIANTS[var])
    json.dump(m, open(fn + ".json", "w"), default=float)
    np.savez_compressed(fn + ".npz", plates=_plates_arr(r["plates"]),
                        kind=np.array([{"intra": 0, "gb": 1}.get(q.get("kind"), -1) for q in r["plates"]]))


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for s in SEEDS:                     # нерастворившиеся гидриды считаются один раз на затравку (кэш)
        job((out, "viz5", 0, s))
    jobs = [(out, v, s, seed) for v, s, seed in itertools.product(VARIANTS, SIG, SEEDS)]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs)):
            print(i + 1, "из", len(jobs), flush=True)
    print("готово")
