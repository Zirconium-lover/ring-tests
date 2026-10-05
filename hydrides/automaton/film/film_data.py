"""Данные для ролика: всё, что сцены Manim потом только рисуют.
Запуск (в окружении с numpy/scipy/scikit-image): python film_data.py папка
"""
import os
import sys
import json
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from ca_hydride import Params, run, Elastic, plate_eigen, eigen_components_for, EPS_N, EPS_T  # noqa: E402

OUT = sys.argv[1]
RUNS = sys.argv[2] if len(sys.argv) > 2 else None        # папка прогонов v2 (стенка, серия по σ)
os.makedirs(OUT, exist_ok=True)
CAL = dict(beta=0.12, sigma_cap=90.0, capture_um=35.0)


def plates_arr(plates):
    return np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in plates]) if plates else np.zeros((0, 4))


# 1. поле одной пластинки: выгода зарождения новой пластинки 0°, 45°, 90°
p = Params(); dx = 0.2; shape = (300, 300)
el = Elastic(shape, dx, p.E, p.nu)
c = np.array([30.0, 30.0]); half = 2.5
ys, xs, fr, comps = plate_eigen(shape, dx, c, 0.0, half, p.h_um)
E = [np.zeros(shape) for _ in range(4)]
for arr, v in zip(E, comps):
    arr[np.ix_(ys, xs)] = fr * v
S = el.stress(*E)
win = slice(int(18 / dx), int(42 / dx))
field = {}
for psi in (0, 45, 90):
    e11, e22, e12 = eigen_components_for(np.radians(psi))
    g = (e11 * S[0] + e22 * S[1] + 2 * e12 * S[2] + EPS_T * S[3]) / EPS_N
    g = np.where(E[0] > 0, np.nan, g)[win, win]
    field[f"g{psi}"] = g.astype(np.float32)
# напряжение σ_yy (по нормали к пластинке) — для «поле вокруг пластинки»
field["syy"] = np.where(E[0] > 0, np.nan, S[1])[win, win].astype(np.float32)
np.savez_compressed(os.path.join(OUT, "field_one_plate.npz"), extent=np.array([-12, 12, -12, 12]), **field)
print("1: поле одной пластинки")


# 2. пошаговый прогон на маленьком поле (как работает алгоритм)
def stepwise(sigma, n_steps, size=(60.0, 60.0), seed=3):
    rec = []

    def cb(plates, expo, hyd, cH):
        if len(plates) >= n_steps:
            raise StopIteration
        ok = np.isfinite(expo)
        lw = np.full(expo.shape, -8.0)
        lw[ok] = (expo[ok] - expo[ok].max()) / np.log(10)
        rec.append(dict(plates=plates_arr(plates), fav=np.clip(lw, -5, 0).astype(np.float32), cH=cH.astype(np.float32)))

    prm = Params(size_um=size, sigma_app=sigma, seed=seed, **CAL)
    try:
        r = run(prm, callback=cb)
        P = plates_arr(r["plates"])
    except StopIteration:
        P = rec[-1]["plates"]
    # зёрна и углы — тем же генератором
    from ca_hydride import make_grains
    grains, gpsi = make_grains(prm, np.random.default_rng(seed))
    return rec, grains, gpsi


for sigma in (0.0, 250.0):
    rec, grains, gpsi = stepwise(sigma, 40)
    np.savez_compressed(os.path.join(OUT, f"steps_s{int(sigma)}.npz"),
                        fav=np.stack([r["fav"] for r in rec]), cH=np.stack([r["cH"] for r in rec]),
                        plates=np.array([r["plates"] for r in rec], dtype=object),
                        grains=grains, gpsi=gpsi, size=np.array([60.0, 60.0]), dx=0.4)
    print("2: пошаговый прогон", sigma, len(rec), "шагов")


# 3. рост на поле 200 × 200 мкм: порядок пластинок и карта выгоды каждые 3 пластинки
def growth(sigma, size=(200.0, 200.0), seed=7, every=3, chi0=30.0):
    frames = []

    def cb(plates, expo, hyd, cH):
        n = len(plates)
        if n % every == 0 and (not frames or frames[-1]["n"] != n):
            ok = np.isfinite(expo)
            lw = np.full(expo.shape, -8.0)
            lw[ok] = (expo[ok] - expo[ok].max()) / np.log(10)
            frames.append(dict(n=n, fav=ndi.zoom(np.clip(lw, -5, 0), 0.5, order=1).astype(np.float32), frac=float(hyd.mean())))

    r = run(Params(size_um=size, sigma_app=sigma, seed=seed, chi0=chi0, **CAL), callback=cb)
    return plates_arr(r["plates"]), frames


for sigma in (0.0, 100.0, 250.0):
    P, frames = growth(sigma)
    np.savez_compressed(os.path.join(OUT, f"growth_s{int(sigma)}.npz"), plates=P,
                        fav=np.stack([f["fav"] for f in frames]), n=np.array([f["n"] for f in frames]),
                        frac=np.array([f["frac"] for f in frames]), size=np.array([200.0, 200.0]))
    print("3: рост", sigma, len(P), "пластинок")

# 4. стенка Э635 при изгибе и серия по напряжениям — из готовых прогонов
if RUNS:
    out = {}
    for n in ("wall_free_seed1", "wall_bend200_seed1", "wall_unif200_seed1"):
        out[n] = np.load(os.path.join(RUNS, n + ".npz"))["plates"]
    ser = {}
    for s in (0, 100, 150, 200, 250):
        z = np.load(os.path.join(RUNS, f"tex_chi30_s{s}_seed1.npz"))["plates"]
        rhf = [json.load(open(f))["RHF_image"] for f in glob.glob(os.path.join(RUNS, f"tex_chi30_s{s}_seed*.json"))]
        ser[f"s{s}"] = z
        ser[f"rhf{s}"] = np.array(rhf)
    np.savez_compressed(os.path.join(OUT, "wall.npz"), **out)
    np.savez_compressed(os.path.join(OUT, "series.npz"), **ser)
    print("4: стенка и серия")
