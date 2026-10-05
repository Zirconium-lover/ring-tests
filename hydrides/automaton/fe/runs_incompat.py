"""Несовместность: поле 72 × 72 мкм, клетка 0.25 мкм, текстура χ0 = 30° ± 26° (как в калибровке).
Сохраняет поля и выгоду g собственной пластинки каждой клетки."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ca_hydride import Params, make_grains  # noqa: E402
from incompat import solve  # noqa: E402
from gmaps import gmap, EPS_N, EPS_T  # noqa: E402

OUT = sys.argv[1]
chi0 = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
p = Params(size_um=(72.0, 72.0), dx=0.25, chi0=chi0, chi_s=26.0, seed=seed)
gr, gpsi = make_grains(p, np.random.default_rng(seed))
res = dict(grains=gr, gpsi=gpsi, dx=p.dx)
for case in ("thermal", "mech"):
    t = time.time()
    S = solve(os.path.join(OUT, "ccx"), f"inc_chi{chi0:g}_s{seed}_{case}", gr, gpsi, p.dx, case=case)
    psi_deg = np.degrees(gpsi[gr])
    # выгода для пластинки, которая может вырасти в этой клетке (её ψ — у зерна)
    g = np.zeros_like(S["sxx"])
    for ang in np.unique(np.round(psi_deg, 3)):
        m = np.isclose(np.round(psi_deg, 3), ang)
        g[m] = gmap(S["sxx"][m], S["syy"][m], S["sxy"][m], S["szz"][m], ang)
    res.update({f"{case}_{k}": v.astype(np.float32) for k, v in S.items()})
    res[f"{case}_g"] = g.astype(np.float32)
    print(case, f"{time.time() - t:.0f} с", flush=True)
# формула автомата для приложенного напряжения (σ = 100 МПа): g_app = σ (ε11 − ε̄)/ε_n
psi = gpsi[gr]
e11 = EPS_N * np.sin(psi) ** 2 + EPS_T * np.cos(psi) ** 2
res["gapp_formula"] = (100.0 * (e11 - 0.5 * (EPS_N + EPS_T)) / EPS_N).astype(np.float32)
np.savez_compressed(os.path.join(OUT, f"incompat_chi{chi0:g}_s{seed}.npz"), **res)
