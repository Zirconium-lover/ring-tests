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
import hydride_spec as hs  # noqa: E402
from gaps import render_plates  # noqa: E402
from ca_hydride import Params, run  # noqa: E402

D = sys.argv[1]
L = 48.0
TILE = 3


def section_plates(c, nv, R, zc):
    """Следы дисков в сечении z = zc: (ND, TD, ψ, полудлина) в координатах автомата (строки вниз)."""
    out = []
    for ci, ni, ri in zip(c, nv, R):
        sin_t = np.sqrt(max(1e-12, 1 - ni[2] ** 2))
        dz = (zc - ci[2] + L / 2) % L - L / 2
        rr = abs(dz) / sin_t
        if rr >= ri:
            continue
        half = np.sqrt(ri * ri - rr * rr)
        d = np.cross(ni, [0, 0, 1.0]); d /= np.linalg.norm(d)
        w = np.array([0, 0, 1.0]) - ni[2] * ni; w /= max(np.linalg.norm(w), 1e-9)
        p0 = ci + w * (dz / max(w[2], 1e-9))
        psi = np.arctan2(-d[1], d[0])                       # строки вниз: угол следа от TD «вверх»
        out.append((p0[1] % L, p0[0] % L, psi, half))
    return np.array(out) if out else np.zeros((0, 4))


def tile(P):
    return np.concatenate([P + np.array([a * L, b * L, 0, 0]) for a in range(TILE) for b in range(TILE)]) if len(P) else P


def rhf_image(P):
    g = render_plates(tile(P), (L * TILE, L * TILE), um=0.65, h_um=0.6, blur_um=1.0, noise=0.03) / 255.0
    r, _ = hs.analyse("ca", um=0.65, g=g, field=True, scales_um=(3.0,), line_um=5.0, n_layers=3)
    return r["scales"][0]["RHF_simon"] if r["scales"] else np.nan


def job3(fn):
    z = np.load(fn)
    vals = [rhf_image(section_plates(z["c"], z["n"], z["R"], zc)) for zc in (np.arange(6) + 0.5) * L / 6]
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
