"""Обработка стендов bench.py.

python bench_report.py bias  папка                 — порог σ* от форы, фора под 45 МПа
python bench_report.py noise папка                 — шум мер (среднее ± разброс по затравкам), поле 160 и 240 мкм
python bench_report.py knock папка папка_шума      — вклад механизма при всех прочих: выключенный − номинал (те же затравки)
python bench_report.py lhs   папка                 — суррогат (гауссов процесс) и индексы Соболя S1, ST по каждой мере
python bench_report.py etch  папка                 — этап 1: утолщение травлением по площади до опыта, F_l до опыта
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LEVEL = 0.25          # порог: F_l посередине между «до опыта» (~0.1) и зоной (~0.4), как граница зон колец
OBS = ("F_l", "F_l0", "RHCP", "RHCF", "RHCP_05", "RHCF_05", "GB_frac", "n_plates", "L_plate_mean", "area", "T_first")


def load(folder, prefix=""):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, prefix + "*.json"))):
        rows.append(json.load(open(f)))
    return rows


def sigmas(r):
    return sorted(float(k[1:]) for k in r if k.startswith("s") and k[1:].replace(".", "").isdigit())


def curve(r, key="F_l"):
    s = sigmas(r)
    return np.array(s), np.array([r[f"s{v:g}"][key] for v in s], float)


def sigma_star(r, level=LEVEL, key="F_l"):
    s, y = curve(r, key)
    ok = np.isfinite(y)
    s, y = s[ok], y[ok]
    if not len(y):
        return np.nan
    if y[0] >= level:
        return 0.0
    i = np.where((y[:-1] < level) & (y[1:] >= level))[0]
    if not len(i):
        return np.inf
    i = i[0]
    return float(s[i] + (level - y[i]) * (s[i + 1] - s[i]) / (y[i + 1] - y[i]))


def fmt(m, s=None):
    return f"{m:.2f}" if s is None else f"{m:.2f}±{s:.2f}"


def rep_bias(folder):
    by = defaultdict(list)
    for r in load(folder):
        by[r["bias_dT"]].append(r)
    xs, ys = [], []
    for b in sorted(by):
        rs = by[b]
        s = sigmas(rs[0])
        Fl = np.array([curve(r)[1] for r in rs])
        st = [sigma_star(r) for r in rs]
        before = np.mean([r["before"]["F_l"] for r in rs])
        print(f"фора {b:4.1f}: до опыта F_l {before:.2f} | F_l(σ) " + " ".join(f"{v:g}:{m:.2f}" for v, m in zip(s, Fl.mean(0)))
              + f" | σ* {', '.join(f'{v:.0f}' for v in st)} | RHCF(σmax) {np.mean([r[f's{s[-1]:g}']['RHCF'] for r in rs]):.2f}"
              + f" | межзёренных {np.mean([r[f's{s[-1]:g}']['GB_frac'] for r in rs]):.2f}")
        fin = [v for v in st if np.isfinite(v)]
        if fin:
            xs.append(b); ys.append(np.mean(fin))
    if len(xs) > 1:
        o = np.argsort(ys)
        print("фора под σ* = 45 МПа:", round(float(np.interp(45.0, np.array(ys)[o], np.array(xs)[o])), 2))


def stats(rows, key, stage):
    v = np.array([r[stage][key] for r in rows if stage in r], float)
    v = v[np.isfinite(v)]
    return (v.mean(), v.std(ddof=1)) if len(v) > 1 else (np.nan, np.nan)


def rep_noise(folder):
    for size in ("nom160", "nom240"):
        rows = load(folder, size)
        if not rows:
            continue
        print(f"== {size}: {len(rows)} затравок")
        for stage in ["before"] + [f"s{v:g}" for v in sigmas(rows[0])]:
            print(f"  {stage:7s}" + "  ".join(f"{k} {fmt(*stats(rows, k, stage))}" for k in ("F_l", "RHCP", "RHCF", "GB_frac", "n_plates")))
        st = np.array([sigma_star(r) for r in rows])
        print(f"  σ* = {np.nanmean(st):.1f} ± {np.nanstd(st, ddof=1):.1f} МПа  ({', '.join(f'{v:.0f}' for v in st)})")


def rep_knock(folder, nom_folder):
    nom = {r["seed"]: r for r in load(nom_folder, "nom160")}
    by = defaultdict(list)
    for r in load(folder):
        by[r["tag"].rsplit("_seed", 1)[0]].append(r)
    s0 = sigmas(next(iter(nom.values())))
    print("вклад механизма при всех прочих: (номинал − без него), среднее ± ст. ошибка по затравкам")
    for name, rs in sorted(by.items()):
        line = []
        for stage in ["before"] + [f"s{v:g}" for v in s0]:
            d = [nom[r["seed"]][stage]["F_l"] - r[stage]["F_l"] for r in rs if r["seed"] in nom and stage in r]
            d = np.array(d, float); d = d[np.isfinite(d)]
            line.append(f"{stage}:{d.mean():+.2f}±{d.std(ddof=1) / np.sqrt(len(d)):.2f}" if len(d) > 1 else f"{stage}:—")
        dst = [sigma_star(nom[r["seed"]]) - sigma_star(r) for r in rs if r["seed"] in nom]
        dst = np.array(dst, float); dst = dst[np.isfinite(dst)]
        print(f"  {name:14s} F_l " + " ".join(line) + (f" | σ* {dst.mean():+.0f}±{dst.std(ddof=1) / np.sqrt(len(dst)):.0f} МПа" if len(dst) > 1 else ""))
        for key in ("RHCF", "GB_frac", "n_plates"):
            d = [nom[r["seed"]][f"s{s0[-2]:g}"][key] - r[f"s{s0[-2]:g}"][key] for r in rs if r["seed"] in nom]
            d = np.array(d, float); d = d[np.isfinite(d)]
            if len(d) > 1:
                print(f"  {'':14s} {key} при {s0[-2]:g} МПа: {d.mean():+.2f}±{d.std(ddof=1) / np.sqrt(len(d)):.2f}")


def rep_lhs(folder):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.model_selection import cross_val_predict
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from bench import FACTORS
    rows = load(folder, "lhs")
    names = [f[0] for f in FACTORS]

    def unit(r):
        u = []
        for name, lo, hi, sc in FACTORS:
            x = r[name]
            u.append(np.log(x / lo) / np.log(hi / lo) if sc == "log" else (x - lo) / (hi - lo))
        return u
    X = np.array([unit(r) for r in rows])
    s0 = sigmas(rows[0])
    targets = {"before_F_l": lambda r: r["before"]["F_l"]}
    for v in s0:
        targets[f"F_l@{v:g}"] = (lambda v_: lambda r: r[f"s{v_:g}"]["F_l"])(v)
    targets["sigma*"] = lambda r: min(sigma_star(r), 200.0)
    targets[f"RHCF@{s0[-2]:g}"] = lambda r: r[f"s{s0[-2]:g}"]["RHCF"]
    targets[f"RHCP@{s0[-2]:g}"] = lambda r: r[f"s{s0[-2]:g}"]["RHCP"]
    targets[f"GB_frac@{s0[-2]:g}"] = lambda r: r[f"s{s0[-2]:g}"]["GB_frac"]
    targets[f"L_plate@{s0[-2]:g}"] = lambda r: r[f"s{s0[-2]:g}"]["L_plate_mean"]
    problem = dict(num_vars=len(names), names=names, bounds=[[0, 1]] * len(names))
    Xs = saltelli.sample(problem, 2048, calc_second_order=False)
    out = {}
    print(f"точек плана: {len(rows)}; факторы: {', '.join(names)}")
    for tname, fn in targets.items():
        y = np.array([fn(r) for r in rows], float)
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        mu, sd = y[ok].mean(), y[ok].std() + 1e-12
        kern = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(names)), length_scale_bounds=(0.05, 50.0)) + WhiteKernel(0.1)
        gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=2, random_state=0)
        yy = (y[ok] - mu) / sd
        pred = cross_val_predict(gp, X[ok], yy, cv=5)
        q2 = 1 - np.mean((pred - yy) ** 2) / np.var(yy)
        gp.fit(X[ok], yy)
        noise = float(gp.kernel_.k2.noise_level)
        Si = sobol.analyze(problem, gp.predict(Xs), calc_second_order=False, print_to_console=False)
        out[tname] = dict(Q2=float(q2), noise_frac=noise, S1=dict(zip(names, map(float, Si["S1"]))),
                          ST=dict(zip(names, map(float, Si["ST"]))), mean=float(mu), sd=float(sd))
        top = sorted(names, key=lambda n: -Si["ST"][names.index(n)])[:6]
        print(f"{tname:14s} Q²={q2:.2f} шум={noise:.2f} | " + ", ".join(
            f"{n} ST {Si['ST'][names.index(n)]:.2f} (S1 {Si['S1'][names.index(n)]:.2f})" for n in top))
    json.dump(out, open(os.path.join(folder, "sobol.json"), "w"), indent=1, ensure_ascii=False)


def rep_etch(folder, target=0.115, grid=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)):
    """Этап 1: утолщение травлением по доле площади до опыта (снимки колец до опыта: 11–12 %), затем F_l до опыта
    тем же оператором (опыт: 0.097–0.125) — проверка без новой подгонки."""
    import rhc_model as RM
    from observe import fl_objects
    fields = []
    for f in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        z = np.load(f)
        fields.append((np.asarray(z["before"], float), float(z["dx"])))
    areas = [np.mean([RM.optical(h, dx, e).mean() for h, dx in fields]) for e in grid]
    e_fit = float(np.interp(target, areas, grid))
    print("доля площади до опыта от утолщения: " + ", ".join(f"{e:g} мкм → {a:.3f}" for e, a in zip(grid, areas)))
    print(f"утолщение под {target:.3f}: {e_fit:.2f} мкм")
    fl = [fl_objects(RM.optical(h, dx, e_fit), RM.UM)["F_l"] for h, dx in fields]
    print(f"F_l до опыта при этом утолщении: {np.nanmean(fl):.3f} ± {np.nanstd(fl, ddof=1):.3f} (опыт 0.097–0.125); "
          f"по полям: {', '.join(f'{v:.2f}' for v in fl)}")
    return e_fit


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "bias":
        rep_bias(sys.argv[2])
    elif mode == "noise":
        rep_noise(sys.argv[2])
    elif mode == "knock":
        rep_knock(sys.argv[2], sys.argv[3])
    elif mode == "lhs":
        rep_lhs(sys.argv[2])
    elif mode == "etch":
        rep_etch(sys.argv[2])
