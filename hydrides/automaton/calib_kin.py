"""Калибровка кинетического движка с измеренным сдвигом температуры выпадения под нагрузкой (app_dT,
Vizcaíno и др. 2014) и форой зёрен с осью c по радиусу (bias_dT) — по тем же мишеням, что calib_beta.py:
RHF по изображению (Lepine) и доля длины под 45–135° (Cinbiz). Метрики — calib_beta.metrics.
python calib_kin.py папка '{"bias_dT": [...], "B": [...], ...}' [процессов]
"init_auto": [true] — нерастворившиеся при T_max гидриды строятся, как в tool.simulate."""
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
from ca_kinetic import KParams, run_kinetic  # noqa: E402
from calib_beta import metrics  # noqa: E402

SIG = [0, 100, 125, 150, 175, 200, 250]
BASE = dict(H_ppm=178.0, T_max=415.0, rate=3.0, B=80.0, Delta0=15.0, sigma_cap=180.0, app_dT=0.08)


def init_plates_for(p):
    """Нерастворившиеся гидриды, если водорода больше TSSD(T_max) — как в tool.simulate: исходная окружная
    структура при полном водороде (последовательный автомат без нагрузки), остаются самые длинные пластинки
    общей площадью, равной нерастворившейся доле."""
    from ca_hydride import Params, run
    from thermo import Cooling, area_fraction
    th = Cooling(p.H_ppm, p.T_max, p.T_end, p.lines)
    if th.frac_left <= 1e-5:
        return None
    base = dict(size_um=tuple(p.size_um), dx=p.dx, grain_um=tuple(p.grain_um), chi0=p.chi0, chi_s=p.chi_s,
                E=p.E, nu=p.nu, seed=p.seed)
    r0 = run(Params(**base, frac=float(area_fraction(p.H_ppm)), sigma_app=0.0))
    P0 = np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r0["plates"]])
    P0 = P0[np.lexsort((np.random.default_rng([p.seed, 3]).random(len(P0)), -P0[:, 3]))]
    area = np.cumsum(2 * P0[:, 3] * p.h_um) / (p.size_um[0] * p.size_um[1])
    return P0[: max(1, int(np.searchsorted(area, th.frac_left)))]


def job(a):
    out, kw = a
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(out, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    kp = dict(BASE, **{k: v for k, v in kw.items() if k != "init_auto"})
    if kw.get("init_auto"):                     # водород сверх TSSD(T_max) — в нерастворившихся гидридах
        kp["init_plates"] = init_plates_for(KParams(**kp))
    r = run_kinetic(KParams(**kp))
    m = metrics(r)
    new = [q for q in r["plates"] if not q.get("init")]
    m["T_first"] = new[0]["T"] if new else np.nan
    m["T_median"] = float(np.median([q["T"] for q in new])) if new else np.nan
    # доля длины межзёренных пластинок (как у Son и др. 2026), всего и среди радиальных (след под 45–135°)
    L = np.array([2 * q["half"] for q in new]); gbk = np.array([q.get("kind") == "gb" for q in new], bool)
    rad = np.array([abs(np.sin(q["psi"])) >= np.sin(np.radians(45)) for q in new], bool)
    m["GB_frac"] = float(L[gbk].sum() / L.sum()) if len(L) else np.nan
    m["GB_frac_rad"] = float(L[gbk & rad].sum() / L[rad].sum()) if rad.any() else np.nan
    m["GB_frac_circ"] = float(L[gbk & ~rad].sum() / L[~rad].sum()) if (~rad).any() else np.nan
    m.update(dict(BASE, **kw), time_s=time.time() - t0)
    np.savez_compressed(os.path.join(out, name + ".npz"),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]),
                        kind=np.array([{"intra": 0, "gb": 1}.get(q.get("kind"), -1) for q in r["plates"]]))
    json.dump(m, open(fn, "w"), default=float)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    grid = json.loads(sys.argv[2])
    grid.setdefault("sigma_app", SIG); grid.setdefault("seed", [1, 2])
    keys = list(grid)
    jobs = [(out, dict(zip(keys, v))) for v in itertools.product(*[grid[k] for k in keys])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs)):
            if (i + 1) % 10 == 0:
                print(i + 1, "из", len(jobs), flush=True)
    print("готово", len(jobs))
