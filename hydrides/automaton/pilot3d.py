"""Пилот 3D: тот же автомат и параметры в 2D и 3D; RHF по следам пластинок в сечении r–θ.
python pilot3d.py папка [процессов]"""
import os
import sys
import json
import itertools
import time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca3d import P3, run3d, section_rhf  # noqa: E402
from ca_hydride import Params, run  # noqa: E402
from ca_analysis import packet_metrics  # noqa: E402

OUT = sys.argv[1]
L = 48.0
SIG = [0, 100, 150, 200, 250]


def job(c):
    fn = os.path.join(OUT, c["name"])
    if os.path.exists(fn + ".json"):
        return
    t0 = time.time()
    if c["dim"] == 3:
        r = run3d(P3(size_um=L, sigma_app=c["s"], seed=c["seed"], chi_sL=c["chi_sL"]))
        nv = np.array([q["n"] for q in r["plates"]]); R = np.array([q["R"] for q in r["plates"]])
        vol = R ** 2
        m = dict(RHF_trace=section_rhf(r), n_plates=len(R), R_mean=float(R.mean()),
                 # доля объёма гидрида с нормалью ближе 40° к TD (радиальные) и к ND (окружные)
                 vol_radial=float(vol[np.abs(nv[:, 0]) > np.cos(np.radians(40))].sum() / vol.sum()),
                 vol_circ=float(vol[np.abs(nv[:, 1]) > np.cos(np.radians(40))].sum() / vol.sum()))
        np.savez_compressed(fn + ".npz", c=np.array([q["c"] for q in r["plates"]]), n=nv, R=R)
    else:
        r = run(Params(size_um=(L, L), dx=0.5, sigma_app=c["s"], seed=c["seed"], beta=0.12, sigma_cap=90.0,
                       capture_um=35.0, cap_local_only=True))
        pm, _, _ = packet_metrics(r)
        m = dict(RHF_trace=pm["plate_RHF"], n_plates=len(r["plates"]))
    m.update(c); m["time_s"] = time.time() - t0
    json.dump(m, open(fn + ".json", "w"), default=float)
    print(c["name"], f"RHF {m['RHF_trace']:.2f}", f"{m['time_s']:.0f} с", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cases = [dict(name=f"d3_sL{sL}_s{s}_seed{k}", dim=3, chi_sL=sL, s=s, seed=k)
             for sL, s, k in itertools.product([0.0, 21.0], SIG, [1, 2])]
    cases += [dict(name=f"d2_s{s}_seed{k}", dim=2, chi_sL=0.0, s=s, seed=k) for s, k in itertools.product(SIG, [1, 2, 3, 4, 5, 6])]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for _ in pool.imap_unordered(job, cases):
            pass
    print("готово", len(cases))
