"""Данные для рисунков учебника: поля одной пластинки, ореол, сложение ореолов, структуры автомата,
зёрна, сверка Фурье — МКЭ. Источники — расчёты сессии (папка RUNS: halo, stackval, stackfe, calib_fz,
calib_halo1) и сам код модели. Результат — npz/json в папке DATA.
python prep.py RUNS DATA"""
import os
import sys
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUT = os.path.join(HERE, "..", "..", "automaton")
sys.path.insert(0, AUT); sys.path.insert(0, os.path.join(AUT, "fe"))

from ca_hydride import Elastic, Params, make_grains, EPS_N, EPS_T  # noqa: E402
import thermo  # noqa: E402

RUNS, DATA = sys.argv[1], sys.argv[2]
os.makedirs(DATA, exist_ok=True)


def rect_frac(T, N, ct, cn, a, th, dx, sub=4):
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * dx
    fr = np.zeros(T.shape)
    for u in off:
        for v in off:
            fr += (np.abs(T + u - ct) < a) & (np.abs(N + v - cn) < th / 2)
    return fr / sub ** 2


# 1. упругое поле одной пластинки 5 × 0.6 мкм: g соосной и перпендикулярной, σ_h
L, dx = 24.0, 0.05
n = int(L / dx)
x = (np.arange(n) + 0.5) * dx - L / 2
T, N = np.meshgrid(x, x, indexing="xy")             # ось 0 — n (y), ось 1 — t (x)
fr = rect_frac(T, N, 0.0, 0.0, 2.5, 0.6, dx)
el = Elastic((n, n), dx, 90e3, 0.34, free_z=True)
S11, S22, S12, S33 = el.stress(EPS_T * fr, EPS_N * fr, 0 * fr, EPS_T * fr)
g_par = (EPS_T * S11 + EPS_N * S22 + EPS_T * S33) / EPS_N
g_perp = (EPS_N * S11 + EPS_T * S22 + EPS_T * S33) / EPS_N
sh = (S11 + S22 + S33) / 3
np.savez_compressed(os.path.join(DATA, "plate_elastic.npz"), g_par=g_par.astype(np.float32),
                    g_perp=g_perp.astype(np.float32), sh=sh.astype(np.float32), fr=fr.astype(np.float32), L=L, dx=dx)

# 2. ореол одиночной пластинки (Фурье, Мизес σ_y 350): p, g соосной, без нагрузки и под 110 МПа
for case in ("a2.5_al0_0", "a2.5_al0_U110", "a2.5_al90_U110"):
    z = np.load(os.path.join(RUNS, "halo", case + ".npz"))
    s = z["sig"].astype(np.float64)
    g = (s[0] * EPS_T + s[1] * EPS_N + s[2] * EPS_T) / EPS_N            # оси ячейки = оси пластинки (x — t, y — n)
    np.savez_compressed(os.path.join(DATA, f"halo_{case}.npz"), p=z["p"], g=g.astype(np.float32), mask=z["mask"],
                        ep_tt=z["ep"][0], ep_nn=z["ep"][1])

# 3. сложение ореолов: колода и цепочка, радиальные, 110 МПа (как fe/halo_check.py)
import hill_fft as hf  # noqa: E402
from halo_check import elastic, g_of, E_STAR, STEP  # noqa: E402
from stack_compare import fft_tn  # noqa: E402
hf.WORKERS = 4
out = {}
for conf in ("deck", "chain"):
    H1 = np.load(os.path.join(RUNS, "halo", "a2.5_al0_U110.npz"))
    meta = json.load(open(os.path.join(RUNS, "halo", "a2.5_al0_U110.json")))
    Sig = np.array(meta["Sig"])
    m = H1["p"].shape[0]; h = 30.0 / m
    st, sn = STEP[conf]
    shifts = [(0, 0), (-int(round(st / h)), -int(round(sn / h))), (-int(round(2 * st / h)), -int(round(2 * sn / h)))]
    m1 = H1["mask"].astype(bool)
    hyd = np.zeros((m, m), bool); es = np.zeros((6, m, m)); halo = np.zeros((6, m, m))
    for a, b in shifts:
        mk = np.roll(m1, (a, b), (0, 1)); hyd |= mk; es += E_STAR[:, None, None] * mk
        halo += np.roll(H1["ep"].astype(np.float64), (a, b), (1, 2))
    halo *= ~hyd
    f = fft_tn(np.load(os.path.join(RUNS, "stackval", f"{conf}_rad_U110_h0.1_fe.npz")), "rad")
    g_ref = (f["tt"] * EPS_T + f["nn"] * EPS_N + f["zz"] * EPS_T) / EPS_N
    g_app = float(Sig @ E_STAR) / EPS_N
    g_el = g_of(elastic(es, Sig, m, h)); g_h = g_of(elastic(es + halo, Sig, m, h))
    g_cap = np.clip(g_el - g_app, -180, 180) + g_app
    out[conf] = dict(ref=g_ref - g_app, halo=g_h - g_app, cap=g_cap - g_app, el=g_el - g_app, hyd=hyd)
np.savez_compressed(os.path.join(DATA, "superpose.npz"),
                    **{f"{c}_{k}": np.asarray(v, np.float32 if k != "hyd" else bool) for c, d in out.items() for k, v in d.items()})

# 4. структуры автомата: потолок 180 против ореолов, seed 1
st = {}
for tag, pat in (("cap", "calib_fz/*free_zTrue*seed1_sigma_app{s}.npz"), ("halo", "calib_halo1/*seed1_sigma_app{s}_*.npz")):
    for s in (0, 125, 175, 250):
        f = glob.glob(os.path.join(RUNS, pat.format(s=s)))
        if f:
            z = np.load(f[0]); st[f"{tag}_{s}"] = z["plates"]
            if "kind" in z:
                st[f"{tag}_{s}_kind"] = z["kind"]
np.savez_compressed(os.path.join(DATA, "structures.npz"), **st)

# 5. зёрна и границы для картинки (поле 30 × 30 мкм)
p = Params(size_um=(30.0, 30.0), dx=0.1, grain_um=(2.5, 4.5), chi0=30.0, chi_s=26.0, seed=4)
grains, gpsi, gbd = make_grains(p, np.random.default_rng(p.seed), gb=True)
np.savez_compressed(os.path.join(DATA, "grains.npz"), grains=grains, gpsi=gpsi, gcells=gbd["cells"], gpsi_f=gbd["psi"],
                    gpair=gbd["pair"])

# 6. сверка Фурье — МКЭ (числа) и растворимость
cmp = json.load(open(os.path.join(RUNS, "stackfe", "compare.json")))
json.dump(cmp, open(os.path.join(DATA, "fft_fem.json"), "w"), indent=1, default=float)
print("готово:", sorted(os.listdir(DATA)))
print("TSSD(400) Э635 =", float(thermo.c_line(400, thermo.TSS["E635"]["TSSD"])))
