"""Перебор параметров автомата; результаты — в JSON (по одному файлу на прогон)."""
import sys, json, os, warnings, itertools, time
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_hydride import Params, run
from ca_analysis import packet_metrics, image_rhf

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def job(kw):
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(OUT, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    r = run(Params(**kw))
    m, C, T = packet_metrics(r)
    m["RHF_image"] = image_rhf(r)
    m.update(kw); m["time_s"] = time.time() - t0
    np.savez_compressed(os.path.join(OUT, name + ".npz"), hyd=r["hyd"].astype(np.float32),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]))
    json.dump(m, open(fn, "w"), default=float)


if __name__ == "__main__":
    grid = json.loads(sys.argv[2])
    keys = list(grid)
    jobs = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print("готово", len(jobs))
