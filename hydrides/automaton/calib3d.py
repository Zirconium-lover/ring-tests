"""Калибровка 3D-автомата (быстрый движок) по RHF Lepine при 0 / 200 / 250 МПа.
python calib3d.py папка '{"beta": [...], "sigma_cap": [...], "sigma_app": [0, 200, 250], "seed": [1]}' [процессов] [L]
RHF — по изображениям 32 сечений поля гидрида ⊥ оси трубы (как image_rhf в 2D), по следам пластинок
и по пакетам на сечениях (как в 2D).
python calib3d.py папка --image — пересчитать RHF по изображению для готовых прогонов (32 сечения);
python calib3d.py папка --stats — досчитать меру пакетов на сечениях (как в 2D)."""
import os
import sys
import json
import itertools
import time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca3d import P3, run3d_fast, section_rhf  # noqa: E402
from sec3d import rhf_sections, rhf_voxel, hyd_from_cells, packet_stats  # noqa: E402

LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}    # MATLAB / HAPPy, табл. 3.1


def job(kw):
    out = kw.pop("_out")
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(out, name)
    if os.path.exists(fn + ".json"):
        return
    t0 = time.time()
    r = run3d_fast(P3(**kw))
    pl = r["plates"]
    c = np.array([q["c"] for q in pl]); nv = np.array([q["n"] for q in pl])
    R = np.array([q["R"] for q in pl])
    ext = {k: np.array([q[k] for q in pl]) for k in ("u", "A", "B", "area", "cax") if pl and k in pl[0]}
    if pl and "cells" in pl[0]:                             # отрезки {10-17}: клетки для 3D-геометрии
        ext["cells"] = np.concatenate([q["cells"] for q in pl]); ext["cfr"] = np.concatenate([q["cfr"] for q in pl])
        ext["cptr"] = np.cumsum([0] + [len(q["cells"]) for q in pl])
    np.savez_compressed(fn + ".npz", c=c, n=nv, R=R, **ext)
    img, se = rhf_voxel(r["hyd"], r["params"].dx, se=True)       # срез поля гидрида, как image_rhf в 2D
    ps = packet_stats(dict(c=c, n=nv, R=R, **ext), kw["size_um"], r["params"].dx)
    m = dict(kw, **ps, RHF_trace=section_rhf(r), RHF_image=img, RHF_image_se=se, img_nsec=32, img_method="voxel", n_plates=len(R),
             R_mean=float(R.mean()), time_s=time.time() - t0)
    json.dump(m, open(fn + ".json", "w"), default=float)
    print(name, f"след {m['RHF_trace']:.2f} пакеты {m['packet_RHF']:.2f} ({m['n_packets']}) изобр. {m['RHF_image']:.2f} {m['time_s']:.0f} с", flush=True)


def restat(fn):
    """Досчитать меру пакетов на сечениях для готового прогона."""
    m = json.load(open(fn))
    if "packet_RHF" in m:
        return
    z = dict(np.load(fn[:-5] + ".npz"))
    m.update(packet_stats(z, m["size_um"], m.get("dx", 0.5)))
    json.dump(m, open(fn, "w"), default=float)


def reimage(fn):
    m = json.load(open(fn))
    if m.get("img_nsec") == 32:
        return
    z = np.load(fn[:-5] + ".npz")
    if "cells" in z:
        dx = m.get("dx", 0.5); n = int(round(m["size_um"] / dx))
        m["RHF_image"], m["RHF_image_se"] = rhf_voxel(hyd_from_cells(n, z["cells"], z["cfr"]), dx, se=True)
    else:
        ell = {k: z[k] for k in ("u", "A", "B") if k in z}
        m["RHF_image"], m["RHF_image_se"] = rhf_sections(z["c"], z["n"], z["R"], m["size_um"], se=True, **ell)
    m["img_nsec"] = 32
    json.dump(m, open(fn, "w"), default=float)


if __name__ == "__main__":
    out = sys.argv[1]
    if sys.argv[2] in ("--image", "--stats"):
        import glob
        with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool:
            list(pool.imap_unordered(reimage if sys.argv[2] == "--image" else restat, glob.glob(os.path.join(out, "*.json"))))
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
