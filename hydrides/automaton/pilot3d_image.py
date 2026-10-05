"""RHF «как на шлифе» для пилота 3D: сечения ⊥ оси трубы → следы пластинок → картинка → RHF по
изображению (как image_rhf в калибровке). Периодическое сечение 48 мкм размножается 3×3.
python pilot3d_image.py папка_пилота"""
import os
import sys
import json
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "fe")); sys.path.insert(0, os.path.join(HERE, ".."))
from sec3d import section_plates, rhf_image as _rhf  # noqa: E402
from ca_hydride import Params, run  # noqa: E402

D = sys.argv[1]
L = 48.0
TILE = 3


def rhf_image(P):
    return _rhf(P, L, TILE)


def job3(fn):
    z = np.load(fn)
    vals = [rhf_image(section_plates(z["c"], z["n"], z["R"], zc, L)) for zc in (np.arange(6) + 0.5) * L / 6]
    return fn, float(np.nanmean(vals))


def job2(args):
    s, seed = args
    r = run(Params(size_um=(L, L), dx=0.5, sigma_app=s, seed=seed, beta=0.12, sigma_cap=90.0, capture_um=35.0,
                   cap_local_only=True))
    P = np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]])
    return (s, seed), rhf_image(P)


if __name__ == "__main__":
    out = {}
    with Pool(4) as pool:
        for fn, v in pool.imap_unordered(job3, sorted(glob.glob(os.path.join(D, "d3_*.npz")))):
            m = json.load(open(fn[:-4] + ".json")); m["RHF_image"] = v; json.dump(m, open(fn[:-4] + ".json", "w"), default=float)
        for (s, seed), v in pool.imap_unordered(job2, [(s, k) for s in (0, 100, 150, 200, 250) for k in range(1, 7)]):
            fn = os.path.join(D, f"d2_s{s}_seed{seed}.json")
            m = json.load(open(fn)); m["RHF_image"] = v; json.dump(m, open(fn, "w"), default=float)
    print("ok")
