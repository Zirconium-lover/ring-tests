"""Прогоны автомата с поправкой на напряжения несовместности (gextra.py).
python run_gextra.py папка dT '{"sigma_app": [...], "seed": [...], ...}' [процессов]"""
import os
import sys
import json
import time
import itertools
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from ca_hydride import Params, run  # noqa: E402
from ca_analysis import packet_metrics, image_rhf  # noqa: E402
from gextra import g_extra  # noqa: E402

OUT, DT = sys.argv[1], float(sys.argv[2])
WORK = os.environ.get("FE_WORK", os.path.join(OUT, "fe"))
BASE = dict(beta=0.12, sigma_cap=90.0, capture_um=35.0)


def job(kw):
    kw = dict(BASE, **kw)
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items())) + f"_dT{DT:g}"
    fn = os.path.join(OUT, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    p = Params(**kw)
    p.g_extra = g_extra(p, WORK, DT) if DT else None
    r = run(p)
    m, C, T = packet_metrics(r)
    m["RHF_image"] = image_rhf(r)
    m.update(kw); m["dT"] = DT; m["time_s"] = time.time() - t0
    np.savez_compressed(os.path.join(OUT, name + ".npz"),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]))
    json.dump(m, open(fn, "w"), default=float)
    print(name, f"{m['time_s']:.0f} с", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    grid = json.loads(sys.argv[3])
    keys = list(grid)
    jobs = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    # поля МКЭ сначала (по одному на раскладку зёрен), потом прогоны параллельно
    seen = set()
    for kw in jobs:
        p = Params(**dict(BASE, **kw))
        key = (p.chi0, p.chi0_profile, p.seed)
        if DT and key not in seen:
            seen.add(key); g_extra(p, WORK, DT)
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 2) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
