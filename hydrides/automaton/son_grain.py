"""Мелкое зерно в условиях опыта Son и др. 2026 (Zr-Nb CWSR, зерно 0.89 мкм; 187 ppm, 400 °C, 0.3 °C/мин).
Фора зёрен 5 °C (Vizcaíno), фора границы 1.5 °C — как вариант viz5 в son_run.py. Варианты геометрии:
  ref02     — зерно Zry-4 2.5 × 4.5 мкм на сетке 0.2 мкм, поле 120 мкм: проверка шага сетки и размера поля;
  fine_sim  — всё в масштабе 0.27 (зерно 0.66 × 1.2, пластинка 0.16 мкм, L_min 0.32, L_max 1.6, сетка 0.1,
              поле 60 мкм): та же упругая задача в единицах зерна, иначе только диффузия и масштаб снимка;
  fine_h06  — зерно 0.66 × 1.2 при прежней толщине пластинки 0.6 мкм (сетка 0.2, поле 120 мкм, L_min 0.6):
              радиальная пластинка в зерно не помещается — только на границах;
  fine_h06x — то же с ростом во времени и продолжением через границы (сквозьзёренные перемычки, как у Son).
python son_grain.py папка [процессов] [варианты через запятую]"""
import os
import sys
import json
import time
import itertools
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool import MATERIALS, Texture, History, Model, simulate, _plates_arr  # noqa: E402
from calib_beta import metrics  # noqa: E402
from son_run import KIN, SIG, SEEDS  # noqa: E402

FINE = (0.66, 1.2)                      # эквивалентный диаметр 0.89 мкм при вытянутости 1.8, как у Zry-4
GEOM = {
    "ref02": dict(grain=(2.5, 4.5), dx=0.2, size=120.0, plate={}),
    "fine_sim": dict(grain=FINE, dx=0.1, size=60.0, plate=dict(h_um=0.16, L_min=0.32, L_max=1.6)),
    "fine_h06": dict(grain=FINE, dx=0.2, size=120.0, plate=dict(L_min=0.6)),
    "fine_h06x": dict(grain=FINE, dx=0.2, size=120.0, plate=dict(L_min=0.6), model=dict(grow_kin=True, cross_tol=15.0)),
}


def job(a):
    out, var, sig, seed = a
    fn = os.path.join(out, f"{var}_s{sig}_seed{seed}")
    if os.path.exists(fn + ".json"):
        return
    g = GEOM[var]
    t0 = time.time()
    model = Model(**dict(KIN, bias_dT=5.0, **g.get("model", {})))
    r = simulate(MATERIALS["ZrNb_CWSR"], Texture(chi0=30.0, grain_um=g["grain"]),
                 History(H_ppm=187.0, T_max=400.0, sigma=float(sig), rate=0.3), model=model, seed=seed,
                 metrics=False, cache=out, size_um=(g["size"], g["size"]), dx=g["dx"], plate=g["plate"])
    m = metrics(r)
    new = [q for q in r["plates"] if not q.get("init")]
    L = np.array([2 * q["half"] for q in new]); gbk = np.array([q.get("kind") == "gb" for q in new], bool)
    rad = np.array([abs(np.sin(q["psi"])) >= np.sin(np.radians(45)) for q in new], bool)
    m["GB_frac"] = float(L[gbk].sum() / L.sum()) if len(L) else np.nan
    m["GB_frac_rad"] = float(L[gbk & rad].sum() / L[rad].sum()) if rad.any() else np.nan
    m["L_mean"] = float(L.mean()) if len(L) else np.nan
    m["n_new"] = len(new)
    m.update(variant=var, sigma_app=sig, seed=seed, time_s=time.time() - t0, grain=list(g["grain"]), dx=g["dx"],
             size=g["size"], **g["plate"])
    json.dump(m, open(fn + ".json", "w"), default=float)
    np.savez_compressed(fn + ".npz", plates=_plates_arr(r["plates"]),
                        kind=np.array([{"intra": 0, "gb": 1}.get(q.get("kind"), -1) for q in r["plates"]]))


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(GEOM)          # какие варианты считать
    pre = [(out, v, 0, s) for v in only for s in SEEDS if v != "fine_h06x"]     # кэш нерастворившихся по геометрии
    rest = [(out, v, s, seed) for v, s, seed in itertools.product(only, SIG, SEEDS)]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, pre)):
            print("кэш", i + 1, "из", len(pre), flush=True)
        for i, _ in enumerate(pool.imap_unordered(job, rest)):
            print(i + 1, "из", len(rest), flush=True)
    print("готово")
