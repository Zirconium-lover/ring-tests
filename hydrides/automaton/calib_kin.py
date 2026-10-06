"""Калибровка кинетического движка с измеренным сдвигом температуры выпадения под нагрузкой (app_dT,
Vizcaíno и др. 2014) и форой зёрен с осью c по радиусу (bias_dT) — по тем же мишеням, что calib_beta.py:
RHF по изображению (Lepine) и доля длины под 45–135° (Cinbiz). Метрики — calib_beta.metrics.
python calib_kin.py папка '{"bias_dT": [...], "B": [...], ...}' [процессов]"""
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


def job(a):
    out, kw = a
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(out, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    r = run_kinetic(KParams(**dict(BASE, **kw)))
    m = metrics(r)
    new = [q for q in r["plates"] if not q.get("init")]
    m["T_first"] = new[0]["T"] if new else np.nan
    m["T_median"] = float(np.median([q["T"] for q in new])) if new else np.nan
    m.update(dict(BASE, **kw), time_s=time.time() - t0)
    np.savez_compressed(os.path.join(out, name + ".npz"),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]))
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
