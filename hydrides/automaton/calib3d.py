"""Калибровка 3D-автомата (быстрый движок) по RHF Lepine при 0 / 200 / 250 МПа.
python calib3d.py папка '{"beta": [...], "sigma_cap": [...], "sigma_app": [0, 200, 250], "seed": [1]}' [процессов] [L]
RHF — по изображениям шести сечений ⊥ оси трубы (как на шлифе) и по следам дисков."""
import os
import sys
import json
import itertools
import time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca3d import P3, run3d_fast, section_rhf  # noqa: E402
from sec3d import rhf_sections  # noqa: E402

LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}    # MATLAB / HAPPy, табл. 3.1


def job(kw):
    out = kw.pop("_out")
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(out, name)
    if os.path.exists(fn + ".json"):
        return
    t0 = time.time()
    r = run3d_fast(P3(**kw))
    c = np.array([q["c"] for q in r["plates"]]); nv = np.array([q["n"] for q in r["plates"]])
    R = np.array([q["R"] for q in r["plates"]])
    np.savez_compressed(fn + ".npz", c=c, n=nv, R=R)
    m = dict(kw, RHF_trace=section_rhf(r), RHF_image=rhf_sections(c, nv, R, kw["size_um"]), n_plates=len(R),
             R_mean=float(R.mean()), time_s=time.time() - t0)
    json.dump(m, open(fn + ".json", "w"), default=float)
    print(name, f"след {m['RHF_trace']:.2f} изобр. {m['RHF_image']:.2f} {m['time_s']:.0f} с", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    grid = json.loads(sys.argv[2])
    L = float(sys.argv[4]) if len(sys.argv) > 4 else 96.0
    keys = list(grid)
    jobs = [dict(zip(keys, v), size_um=L, _out=out) for v in itertools.product(*[grid[k] for k in keys])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print("готово", len(jobs))
