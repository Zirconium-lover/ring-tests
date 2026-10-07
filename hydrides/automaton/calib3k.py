"""Перебор для кинетического 3D-движка (ca3d_kinetic.py): сечения как шлифы, доли нормалей.
python calib3k.py папка '[{"sigma_app": [..], "sigma_axial": [..], ...}, ...]' [процессов]
Каждый словарь — своя сетка (декартово произведение), к нему добавляются затравки seed (по умолчанию 1, 2)."""
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
from ca3d_kinetic import K3, run3d_kinetic, section_measures  # noqa: E402


def job(a):
    out, kw = a
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(out, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    r = run3d_kinetic(K3(**kw))
    m = section_measures(r)
    P = r["plates"]
    m.update(kw, n_plates=len(P), R_mean=float(np.mean([q["R"] for q in P])) if P else np.nan,
             T_first=P[0]["T"] if P else np.nan, T_median=float(np.median([q["T"] for q in P])) if P else np.nan,
             hyd_frac=float(r["hyd"].mean()), time_s=time.time() - t0)
    json.dump(m, open(fn, "w"), default=float)
    np.savez_compressed(os.path.join(out, name + ".npz"),
                        c=np.array([q["c"] for q in P]), n=np.array([q["n"] for q in P]),
                        R=np.array([q["R"] for q in P]), T=np.array([q["T"] for q in P]))


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    grids = json.loads(sys.argv[2])
    if isinstance(grids, dict):
        grids = [grids]
    jobs = []
    for grid in grids:
        grid = dict(grid); grid.setdefault("seed", [1, 2])
        keys = list(grid)
        jobs += [(out, dict(zip(keys, v))) for v in itertools.product(*[grid[k] for k in keys])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs)):
            print(i + 1, "из", len(jobs), flush=True)
    print("готово")
