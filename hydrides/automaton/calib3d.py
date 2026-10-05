"""Калибровка 3D-автомата (быстрый движок) по RHF Lepine при 0 / 200 / 250 МПа.
python calib3d.py папка '{"beta": [...], "sigma_cap": [...], "sigma_app": [0, 200, 250], "seed": [1]}' [процессов] [L]
RHF — по изображениям 32 сечений ⊥ оси трубы (как на шлифе) и по следам дисков.
python calib3d.py папка --image — пересчитать RHF по изображению для готовых прогонов (32 сечения)."""
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
    ell = {k: np.array([q[k] for q in r["plates"]]) for k in ("u", "A", "B")} if r["plates"] and "u" in r["plates"][0] else {}
    np.savez_compressed(fn + ".npz", c=c, n=nv, R=R, **ell)
    img, se = rhf_sections(c, nv, R, kw["size_um"], se=True, **ell)
    m = dict(kw, RHF_trace=section_rhf(r), RHF_image=img, RHF_image_se=se, img_nsec=32, n_plates=len(R),
             R_mean=float(R.mean()), time_s=time.time() - t0)
    json.dump(m, open(fn + ".json", "w"), default=float)
    print(name, f"след {m['RHF_trace']:.2f} изобр. {m['RHF_image']:.2f} {m['time_s']:.0f} с", flush=True)


def reimage(fn):
    m = json.load(open(fn))
    if m.get("img_nsec") == 32:
        return
    z = np.load(fn[:-5] + ".npz")
    ell = {k: z[k] for k in ("u", "A", "B") if k in z}
    m["RHF_image"], m["RHF_image_se"] = rhf_sections(z["c"], z["n"], z["R"], m["size_um"], se=True, **ell)
    m["img_nsec"] = 32
    json.dump(m, open(fn, "w"), default=float)


if __name__ == "__main__":
    out = sys.argv[1]
    if sys.argv[2] == "--image":
        import glob
        with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool:
            list(pool.imap_unordered(reimage, glob.glob(os.path.join(out, "*.json"))))
        sys.exit()
    os.makedirs(out, exist_ok=True)
    grid = json.loads(sys.argv[2])
    L = float(sys.argv[4]) if len(sys.argv) > 4 else 96.0
    keys = list(grid)
    jobs = [dict(zip(keys, v), size_um=L, _out=out) for v in itertools.product(*[grid[k] for k in keys])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print("готово", len(jobs))
