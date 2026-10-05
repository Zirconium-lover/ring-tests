"""Проверки: сгущение сетки и идеальная пластичность (σ_y = 250 МПа, прямоугольная пластинка)."""
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
for name, mkw, kw in (("fine_pl250", dict(s_min=0.025, s_win=0.1), dict(sy=250, hard=HARD())),
                      ("perf_pl250", dict(), dict(sy=250, hard=((0.0, 0.0), (1.0, 1.0))))):
    t0 = time.time()
    mesh = mesh_quarter(a=2.5, h=0.6, **mkw)
    r = solve(WORK, name, mesh, E=E, nu=NU, **kw)
    for k, v in on_grid(r, X, Y).items():
        res[f"{name}_{k}"] = v
    print(name, len(mesh[1]), f"{time.time() - t0:.0f} с", flush=True)
np.savez_compressed(os.path.join(OUT, "fields_check.npz"), **res)
