"""Перебор параметров кинетической модели (ca_kinetic): python kin_sweep.py папка '{"B": [...], ...}' [процессов]"""
import sys, json, os, warnings, itertools, time
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_kinetic import KParams, run_kinetic
from ca_analysis import packet_metrics, image_rhf, interdistance

OUT = sys.argv[1]
BASE = dict(H_ppm=178.0, T_max=415.0, rate=3.0)


def job(kw):
    kw = dict(BASE, **kw)
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(OUT, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    r = run_kinetic(KParams(**kw))
    m, C, T = packet_metrics(r)
    m["RHF_image"] = image_rhf(r)
    new = [q for q in r["plates"] if not q.get("init")]
    m["T_first"] = new[0]["T"] if new else np.nan
    m["T_median"] = float(np.median([q["T"] for q in new])) if new else np.nan
    m["dist_TD"], m["n_TD"] = interdistance(r, r["params"].dx, 1)
    m["frac"] = float(r["hyd"].mean()); m["c_end"] = float(r["hist"]["c_mean"][-1])
    m.update(kw); m["time_s"] = time.time() - t0
    np.savez_compressed(os.path.join(OUT, name + ".npz"),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]),
                        T=np.array([np.nan if q["T"] is None else q["T"] for q in r["plates"]]))
    json.dump(m, open(fn, "w"), default=float)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    grid = json.loads(sys.argv[2])
    keys = list(grid)
    jobs = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print("готово", len(jobs))
