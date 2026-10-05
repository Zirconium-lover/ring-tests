"""Чувствительность: форма торца, толщина, несоответствие вдоль пластинки (σ_y = 250 МПа)."""
import os
import sys
import time
import numpy as np
from runs import on_grid, HARD, E, NU
from plate_fe import mesh_quarter, solve

OUT = sys.argv[1]
WORK = os.path.join(OUT, "ccx")
g = np.arange(-12 + 0.05, 12, 0.1)
X, Y = np.meshgrid(g, g)
res = dict(x=g, y=g)
variants = [("lens", dict(shape="lens"), dict()),
            ("thin", dict(h=0.3), dict()),
            ("et0", dict(), dict(eps_t=0.0))]
for vname, mkw, skw in variants:
    mesh = mesh_quarter(a=2.5, **({"h": 0.6} | mkw))
    for tag, kw in (("el", dict()), ("pl250", dict(sy=250, hard=HARD()))):
        t0 = time.time()
        r = solve(WORK, f"{vname}_{tag}", mesh, E=E, nu=NU, **kw, **skw)
        for k, v in on_grid(r, X, Y).items():
            res[f"{vname}_{tag}_{k}"] = v
        print(vname, tag, len(mesh[1]), f"{time.time() - t0:.0f} с", flush=True)
np.savez_compressed(os.path.join(OUT, "fields_sens.npz"), **res)
