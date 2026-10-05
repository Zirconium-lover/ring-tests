"""Расчёты одной пластинки: Фурье (как в автомате) и МКЭ — упругий и упругопластические.
Результат: поля на общей сетке (окно ±12 мкм, шаг 0.1 мкм) в OUT/fields.npz."""
import os
import sys
import time
import numpy as np
from scipy.interpolate import LinearNDInterpolator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ca_hydride import Elastic, plate_eigen  # noqa: E402
from plate_fe import mesh_quarter, solve, mirror  # noqa: E402

OUT = sys.argv[1]
WORK = os.path.join(OUT, "ccx")
E, NU, HALF, H = 90e3, 0.34, 2.5, 0.6
SY = [150, 250, 400]
HARD = lambda: ((0.0, 0.0), (100.0, 0.1), (150.0, 1.0))          # Δσ над σ_y при ε_p


def fft_field(dx, L=60.0):
    n = int(round(L / dx))
    el = Elastic((n, n), dx, E, NU)
    c = np.array([L / 2, L / 2])
    ys, xs, fr, comps = plate_eigen((n, n), dx, c, 0.0, HALF, H)
    Ein = [np.zeros((n, n)) for _ in range(4)]
    for a, v in zip(Ein, comps):
        a[np.ix_(ys, xs)] = fr * v
    S11, S22, S12, S33 = el.stress(*Ein)
    hyd = np.zeros((n, n)); hyd[np.ix_(ys, xs)] = fr
    # строки вниз → ось y вверх: переворот, σ_xy меняет знак
    x = (np.arange(n) + 0.5) * dx - L / 2
    y = L / 2 - (np.arange(n) + 0.5) * dx
    return dict(x=x, y=y[::-1], sxx=S11[::-1], syy=S22[::-1], sxy=-S12[::-1], szz=S33[::-1], hyd=hyd[::-1])


def on_grid(r, X, Y):
    x, y, S, PE, F = mirror(r["xy"], r["S"], r["PE"], r["plate"])
    pts = np.stack([x, y], 1)
    out = {}
    for k, j in (("sxx", 0), ("syy", 1), ("szz", 2), ("sxy", 3)):
        out[k] = LinearNDInterpolator(pts, S[:, j])(X, Y)
    out["peeq"] = LinearNDInterpolator(pts, PE)(X, Y)
    out["plate"] = LinearNDInterpolator(pts, F.astype(float))(X, Y) > 0.5
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    g = np.arange(-12 + 0.05, 12, 0.1)
    X, Y = np.meshgrid(g, g)
    res = dict(x=g, y=g)
    for dx in (0.1, 0.4):
        f = fft_field(dx)
        sel_x = np.abs(f["x"]) < 12; sel_y = np.abs(f["y"]) < 12
        for k in ("sxx", "syy", "sxy", "szz", "hyd"):
            res[f"fft{dx}_{k}"] = f[k][np.ix_(sel_y, sel_x)]
        res[f"fft{dx}_x"] = f["x"][sel_x]; res[f"fft{dx}_y"] = f["y"][sel_y]
    t = time.time()
    mesh = mesh_quarter(a=HALF, h=H)
    print("сетка:", len(mesh[1]), "элементов", flush=True)
    cases = [("el", dict())] + [(f"pl{s}", dict(sy=s, hard=HARD())) for s in SY]
    for name, kw in cases:
        t0 = time.time()
        r = solve(WORK, name, mesh, E=E, nu=NU, **kw)
        fld = on_grid(r, X, Y)
        for k, v in fld.items():
            res[f"{name}_{k}"] = v
        np.savez_compressed(os.path.join(OUT, f"raw_{name}.npz"), **r)
        print(name, f"{time.time() - t0:.0f} с, PEEQ max {r['PE'].max():.3f}", flush=True)
    np.savez_compressed(os.path.join(OUT, "fields.npz"), **res)
