"""Обработка стендов bench.py.

python bench_report.py bias  папка                 — порог σ* от форы, фора под 45 МПа
python bench_report.py noise папка                 — шум мер (среднее ± разброс по затравкам), поле 160 и 240 мкм
python bench_report.py knock папка папка_шума      — вклад механизма при всех прочих: выключенный − номинал (те же затравки)
python bench_report.py lhs   папка [папка_шума]    — суррогат (гауссов процесс, шум — по повторам номинала) и индексы Соболя ST/S1
python bench_report.py etch  папка                 — этап 1: утолщение травлением по площади до опыта, F_l до опыта
python bench_report.py remeasure папка [...]       — пересъёмка мер по снимку из сохранённых полей (observe.ETCH_UM)
python bench_report.py match папка папка_шума      — согласование с данными калибровки (history matching) и прогнозы на нём
python bench_report.py lhs2  папка                 — второй круг: Соболь по Fn лаборатории и морфологии
python bench_report.py calib папка [папка_шума]    — фора как функция остальных факторов, откалиброванные прогнозы и их полоса
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
    """Строки расчётов; меры по снимку — из пересъёмки (remeasured.json), если она есть."""
    rm_fn = os.path.join(folder, "remeasured.json")
    rm = json.load(open(rm_fn)) if os.path.exists(rm_fn) else {}
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, prefix + "*.json"))):
        if os.path.basename(f) in ("remeasured.json", "sobol.json"):
            continue
        r = json.load(open(f))
        for stage, v in rm.get(r.get("tag"), {}).items():
            if stage in r:
                r[stage].update(v)
        rows.append(r)
    return rows


def remeasure(folder, etch_um=None, procs=4, force=False):
    """Пересъёмка мер по снимку всех сохранённых полей папки одним оператором (observe.ETCH_UM); force — заново все."""
    from multiprocessing import Pool
    import observe as OB
    e = OB.ETCH_UM if etch_um is None else etch_um
    fn = os.path.join(folder, "remeasured.json")
    done = json.load(open(fn)) if (os.path.exists(fn) and not force) else {}
    todo = [f for f in sorted(glob.glob(os.path.join(folder, "*.npz")))
            if os.path.basename(f)[:-4] not in done and os.path.exists(f[:-4] + ".json")]
    with Pool(procs) as pool:
        for tag, v in pool.imap_unordered(_remeasure_one, [(f, e) for f in todo]):
            done[tag] = v
            json.dump(done, open(fn, "w"), default=float)
    print(f"пересъёмка {folder}: {len(todo)} полей, утолщение {e} мкм")


def _remeasure_one(a):
    f, e = a
    import observe as OB
    meta = json.load(open(f[:-4] + ".json"))
    z = np.load(f)
    nw = int(round(float(meta.get("walls_um", 0.0)) / float(z["dx"])))
    v = {k: OB.observe_image(z[k], float(z["dx"]), nw, e) for k in z.files if k != "dx"}
    v["etch_um"] = e
    return meta["tag"], v


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


def rep_lhs(folder, noise_folder=None):
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from bench import FACTORS
    rows = load(folder, "lhs")
    names = [f[0] for f in FACTORS]

    def unit(r):
        return [np.log(r[n] / lo) / np.log(hi / lo) if sc == "log" else (r[n] - lo) / (hi - lo) for n, lo, hi, sc in FACTORS]
    X = np.array([unit(r) for r in rows])
    s0 = sigmas(rows[0])
    sm = f"s{s0[-2]:g}"
    targets = ([("before", "F_l")] + [(f"s{v:g}", "F_l") for v in s0]
               + [("before", "F_plates")] + [(f"s{v:g}", "F_plates") for v in s0]       # доля длины радиальных пластинок
               + [(sm, k) for k in ("RHCF", "RHCP", "GB_frac", "L_plate_mean", "n_plates")])
    problem = dict(num_vars=len(names), names=names, bounds=[[0, 1]] * len(names))
    Xs = saltelli.sample(problem, 2048, calc_second_order=False)
    Xs[:, names.index("wang")] = np.round(Xs[:, names.index("wang")])      # правило — да/нет, не промежуточное
    out = {}
    print(f"точек плана: {len(rows)}; факторы: {', '.join(names)}")
    for stage, key in targets:
        y = np.array([r[stage][key] for r in rows], float)
        if np.isfinite(y).sum() < 20:
            continue
        nsd = noise_sd(noise_folder, stage, key)
        f, q2, gp, nz = _gp_fit(X, y, nsd, cv=True)
        Si = sobol.analyze(problem, f(Xs), calc_second_order=False, print_to_console=False)
        tname = f"{key}@{stage}"
        # доля шума в дисперсии меры по плану и потолок Q² при таком шуме
        out[tname] = dict(Q2=q2, noise_frac=nz, Q2_max=1 - nz, S1=dict(zip(names, map(float, Si["S1"]))),
                          ST=dict(zip(names, map(float, Si["ST"]))), ST_conf=dict(zip(names, map(float, Si["ST_conf"]))),
                          mean=float(np.nanmean(y)), sd=float(np.nanstd(y)))
        top = sorted(names, key=lambda n: -Si["ST"][names.index(n)])[:6]
        print(f"{tname:18s} Q²={q2:.2f} (потолок {1 - nz:.2f}) | " + ", ".join(
            f"{n} {Si['ST'][names.index(n)]:.2f}/{Si['S1'][names.index(n)]:.2f}" for n in top), flush=True)
    json.dump(out, open(os.path.join(folder, "sobol.json"), "w"), indent=1, ensure_ascii=False)


def _gp_fit(X, y, noise_sd=None, cv=False):
    """Гауссов процесс с ARD. Шум — затравочный разброс меры, измеренный на повторах номинала (noise_sd): без него
    суррогат подгоняет шум (Q² < 0). → (предсказатель, Q² по перекрёстной проверке или None)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    ok = np.isfinite(y)
    mu, sd = y[ok].mean(), y[ok].std() + 1e-12
    nz = (noise_sd / sd) ** 2 if noise_sd else 0.1
    white = WhiteKernel(nz, "fixed") if noise_sd else WhiteKernel(0.1, (0.02, 1.0))
    kern = ConstantKernel(1.0, (0.01, 10.0)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(0.2, 100.0)) + white
    mk = lambda: GaussianProcessRegressor(kern, n_restarts_optimizer=3, random_state=0)
    yy = (y[ok] - mu) / sd
    q2 = None
    if cv:
        from sklearn.model_selection import cross_val_predict
        pred = cross_val_predict(mk(), X[ok], yy, cv=5)
        q2 = float(1 - np.mean((pred - yy) ** 2) / np.var(yy))
    gp = mk().fit(X[ok], yy)
    return (lambda Z: gp.predict(Z) * sd + mu), q2, gp, float(nz)


def factors_of(rows):
    """Набор факторов плана (bench.FACTORS или FACTORS2 — по полю factor_set строк)."""
    import bench
    return getattr(bench, rows[0].get("factor_set", "FACTORS"))


def noise_sd(noise_folder, stage, key):
    """Затравочный разброс меры по повторам номинала (bench noise, поле 160 мкм); нет меры — None (шум оценивается)."""
    if not noise_folder:
        return None
    v = np.array([r[stage][key] for r in load(noise_folder, "nom160") if stage in r and key in r[stage]], float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1)) if len(v) > 2 else None


def rep_calib(folder, level=LEVEL, s_target=45.0, n_mc=3000, noise_folder=None):
    """Фора как функция остальных неопределённых факторов: по суррогатам F_l(σ) для каждого набора факторов ищется
    фора, при которой кривая проходит уровень level при s_target (порог колец). Затем откалиброванные прогнозы
    (до опыта, F_l(σ), RHCF) — их полоса от неопределённости остальных факторов и доли (первого порядка, по
    корреляционному отношению) от каждого фактора."""
    from bench import FACTORS
    rows = load(folder, "lhs")
    names = [f[0] for f in FACTORS]
    ib = names.index("bias_dT")

    def unit(r):
        return [np.log(r[n] / lo) / np.log(hi / lo) if sc == "log" else (r[n] - lo) / (hi - lo) for n, lo, hi, sc in FACTORS]
    X = np.array([unit(r) for r in rows])
    s0 = sigmas(rows[0])
    gps = {v: _gp_fit(X, np.array([r[f"s{v:g}"]["F_l"] for r in rows], float), noise_sd(noise_folder, f"s{v:g}", "F_l"))[0]
           for v in s0}
    g_before = _gp_fit(X, np.array([r["before"]["F_l"] for r in rows], float), noise_sd(noise_folder, "before", "F_l"))[0]
    g_rhcf = _gp_fit(X, np.array([r[f"s{s0[-2]:g}"]["RHCF"] for r in rows], float),
                     noise_sd(noise_folder, f"s{s0[-2]:g}", "RHCF"))[0]
    rng = np.random.default_rng(1)
    Z = rng.random((n_mc, len(names)))
    iw = names.index("wang"); Z[:, iw] = np.round(Z[:, iw])
    ub = np.linspace(0, 1, 26)
    # F_l(σ = s_target) для каждой точки и каждой форы: линейно между соседними σ сетки
    i1 = np.searchsorted(s0, s_target); sa, sb = s0[i1 - 1], s0[i1]
    w = (s_target - sa) / (sb - sa)
    Zb = np.repeat(Z, len(ub), 0); Zb[:, ib] = np.tile(ub, n_mc)
    F_t = ((1 - w) * gps[sa](Zb) + w * gps[sb](Zb)).reshape(n_mc, len(ub))
    bias = np.full(n_mc, np.nan)
    for k in range(n_mc):                     # фора, при которой F_l(45 МПа) = level: первый переход сверху вниз
        d = F_t[k] - level
        j = np.where((d[:-1] >= 0) & (d[1:] < 0))[0]
        if len(j):
            j = j[0]; bias[k] = ub[j] + d[j] / (d[j] - d[j + 1]) * (ub[1] - ub[0])
    ok = np.isfinite(bias)
    lo, hi = FACTORS[ib][1], FACTORS[ib][2]
    print(f"F_l({s_target:g} МПа) выше уровня при любой форе: {(F_t.min(1) >= level).mean() * 100:.0f} % наборов, "
          f"ниже при любой: {(F_t.max(1) < level).mean() * 100:.0f} %")
    if ok.sum() < 20:
        print("форы под порог почти не находится — суррогату мало точек или уровень вне досягаемости")
        return
    print(f"фора под порог {s_target:g} МПа (F_l = {level}): найдена в {ok.mean() * 100:.0f} % наборов; "
          f"{lo + (hi - lo) * np.nanmedian(bias):.1f} °C (5–95 %: {lo + (hi - lo) * np.nanpercentile(bias, 5):.1f}–"
          f"{lo + (hi - lo) * np.nanpercentile(bias, 95):.1f})")
    Zc = Z[ok].copy(); Zc[:, ib] = bias[ok]
    preds = {"до опыта": g_before(Zc)}
    for v in s0:
        preds[f"F_l {v:g} МПа"] = gps[v](Zc)
    preds[f"RHCF {s0[-2]:g} МПа"] = g_rhcf(Zc)
    preds["фора"] = lo + (hi - lo) * bias[ok]
    others = [n for n in names if n != "bias_dT"]
    out = {}
    for key, y in preds.items():
        # доля дисперсии от каждого фактора (корреляционное отношение по 10 бинам)
        eta = {}
        for n in others:
            xb = np.minimum((Zc[:, names.index(n)] * 10).astype(int), 9)
            m = np.array([y[xb == b].mean() if (xb == b).any() else y.mean() for b in range(10)])
            cnt = np.bincount(xb, minlength=10)
            eta[n] = float((cnt * (m - y.mean()) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        top = sorted(eta, key=lambda n: -eta[n])[:4]
        out[key] = dict(median=float(np.median(y)), p5=float(np.percentile(y, 5)), p95=float(np.percentile(y, 95)), share=eta)
        print(f"  {key:12s} {np.median(y):.2f} (5–95 %: {np.percentile(y, 5):.2f}–{np.percentile(y, 95):.2f}) | "
              + ", ".join(f"{n} {eta[n]:.2f}" for n in top))
    json.dump(out, open(os.path.join(folder, "calib.json"), "w"), indent=1, ensure_ascii=False)


def rep_match(folder, noise_folder, n=200000, cut=3.0, disc=0.03):
    """Согласование с данными калибровки (history matching): все неопределённые факторы — в своих физических
    пределах; область, где суррогат не противоречит данным калибровки (неправдоподобие < cut), и прогнозы
    проверочных мер на ней. Данные калибровки: F_l до опыта 0.10–0.125 (табл. 1 Плясова) и порог колец 45 ± 9 МПа
    (F_l = 0.25 по средней кривой). Неправдоподобие = |прогноз − опыт| / √(ошибка опыта² + ошибка суррогата² +
    расхождение модели² (disc))."""
    from bench import FACTORS
    rows = load(folder, "lhs")
    names = [f[0] for f in FACTORS]

    def unit(r):
        return [np.log(r[k] / lo) / np.log(hi / lo) if sc == "log" else (r[k] - lo) / (hi - lo) for k, lo, hi, sc in FACTORS]
    X = np.array([unit(r) for r in rows])
    s0 = sigmas(rows[0])
    fits = {}
    for stage in ["before"] + [f"s{v:g}" for v in s0]:
        y = np.array([r[stage]["F_l"] for r in rows], float)
        _, _, gp, nz = _gp_fit(X, y, noise_sd(noise_folder, stage, "F_l"))
        ok = np.isfinite(y); mu, sd = y[ok].mean(), y[ok].std()
        fits[stage] = (gp, nz, mu, sd)
    for stage, key in ((f"s{s0[-2]:g}", "RHCF"), (f"s{s0[-2]:g}", "GB_frac")):
        y = np.array([r[stage][key] for r in rows], float)
        _, _, gp, nz = _gp_fit(X, y, noise_sd(noise_folder, stage, key))
        ok = np.isfinite(y); fits[key] = (gp, nz, y[ok].mean(), y[ok].std())

    def pred(key, Z):
        gp, nz, mu, sd = fits[key]
        m, s = gp.predict(Z, return_std=True)
        return m * sd + mu, np.sqrt(np.clip(s ** 2 - nz, 0, None)) * sd     # среднее и ошибка суррогата (без шума затравок)
    rng = np.random.default_rng(7)
    Z = rng.random((n, len(names)))
    Z[:, names.index("wang")] = np.round(Z[:, names.index("wang")])
    I = np.zeros(n)
    mb, sb = pred("before", Z)
    I = np.maximum(I, np.abs(mb - 0.1125) / np.sqrt(0.0125 ** 2 + sb ** 2 + disc ** 2))
    curve = {v: pred(f"s{v:g}", Z) for v in s0}
    # порог по средней кривой: σ, где F_l проходит 0.25 (линейно по сетке σ), ошибка — из ошибок суррогата у перехода
    F = np.stack([curve[v][0] for v in s0], 1); Fs = np.stack([curve[v][1] for v in s0], 1)
    sg = np.array(s0)
    st = np.full(n, np.nan); st_err = np.full(n, np.nan)
    above0 = F[:, 0] >= LEVEL
    st[above0] = 0.0
    for j in range(len(sg) - 1):
        m = np.isnan(st) & (F[:, j] < LEVEL) & (F[:, j + 1] >= LEVEL)
        slope = (F[m, j + 1] - F[m, j]) / (sg[j + 1] - sg[j])
        st[m] = sg[j] + (LEVEL - F[m, j]) / slope
        st_err[m] = np.sqrt(Fs[m, j] ** 2 + disc ** 2) / slope
    st_err[above0] = np.sqrt(Fs[above0, 0] ** 2 + disc ** 2) / np.maximum((F[above0, 1] - F[above0, 0]) / (sg[1] - sg[0]), 1e-3)
    st[np.isnan(st)] = 300.0; st_err[np.isnan(st_err)] = 50.0
    I = np.maximum(I, np.abs(st - 45.0) / np.sqrt(9.0 ** 2 + st_err ** 2))
    keep = I < cut
    print(f"согласовано с данными калибровки (неправдоподобие < {cut:g}): {keep.mean() * 100:.1f} % пространства факторов "
          f"({keep.sum()} из {n})")
    if keep.sum() < 50:
        return
    print("где лежат согласованные наборы (медиана и 5–95 % в долях диапазона → в единицах):")
    for i, (k, lo, hi, sc) in enumerate(FACTORS):
        q = np.percentile(Z[keep, i], [5, 50, 95])
        conv = (lambda u: lo * (hi / lo) ** u) if sc == "log" else (lambda u: lo + (hi - lo) * u)
        narrowed = (q[2] - q[0]) < 0.75
        print(f"  {k:10s} {conv(q[1]):8.3g} ({conv(q[0]):.3g}–{conv(q[2]):.3g}){'  ← сужен' if narrowed else ''}")
    print("прогнозы на согласованной области (медиана, 5–95 %):")
    out = {}
    for key in ["before"] + [f"s{v:g}" for v in s0] + ["RHCF", "GB_frac"]:
        m, _ = pred(key, Z[keep])
        out[key] = [float(np.median(m)), float(np.percentile(m, 5)), float(np.percentile(m, 95))]
        print(f"  {key:8s} {out[key][0]:.2f} ({out[key][1]:.2f}–{out[key][2]:.2f})")
    print(f"  порог σ* {np.median(st[keep]):.0f} ({np.percentile(st[keep], 5):.0f}–{np.percentile(st[keep], 95):.0f}) МПа")
    json.dump(dict(frac=float(keep.mean()), pred=out), open(os.path.join(folder, "match.json"), "w"), indent=1)


def rep_lhs2(folder):
    """Второй круг: индексы Соболя по ориентации (Fn лаборатории) и морфологии (M_*) до опыта и под нагрузкой."""
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    rows = load(folder, "lhs2")
    FACTORS = factors_of(rows)
    names = [f[0] for f in FACTORS]

    def unit(r):
        return [np.log(r[k] / lo) / np.log(hi / lo) if sc == "log" else (r[k] - lo) / (hi - lo) for k, lo, hi, sc in FACTORS]
    X = np.array([unit(r) for r in rows])
    stages = ["before"] + [k for k in rows[0] if k.startswith("s") and k[1:].replace(".", "").isdigit()]
    keys = ("Fn_lab", "M_L_obj_w", "M_L_seg_w", "M_spacing_r", "M_skel_density")
    problem = dict(num_vars=len(names), names=names, bounds=[[0, 1]] * len(names))
    Xs = saltelli.sample(problem, 2048, calc_second_order=False)
    out = {}
    print(f"точек: {len(rows)}; факторы: {', '.join(names)}")
    for stage in stages:
        for key in keys:
            y = np.array([r[stage].get(key, np.nan) for r in rows], float)
            if np.isfinite(y).sum() < 20:
                continue
            f, q2, gp, nz = _gp_fit(X, y, None, cv=True)
            Si = sobol.analyze(problem, f(Xs), calc_second_order=False, print_to_console=False)
            out[f"{key}@{stage}"] = dict(Q2=q2, noise_frac=nz, mean=float(np.nanmean(y)), p10=float(np.nanpercentile(y, 10)),
                                         p90=float(np.nanpercentile(y, 90)), ST=dict(zip(names, map(float, Si["ST"]))),
                                         S1=dict(zip(names, map(float, Si["S1"]))))
            top = sorted(names, key=lambda n: -Si["ST"][names.index(n)])[:5]
            print(f"{key + '@' + stage:22s} Q²={q2:.2f} шум={nz:.2f} | {np.nanpercentile(y, 10):.3g}–{np.nanpercentile(y, 90):.3g} | "
                  + ", ".join(f"{n} {Si['ST'][names.index(n)]:.2f}/{Si['S1'][names.index(n)]:.2f}" for n in top), flush=True)
    json.dump(out, open(os.path.join(folder, "sobol2.json"), "w"), indent=1, ensure_ascii=False)


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
        rep_lhs(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif mode == "etch":
        rep_etch(sys.argv[2])
    elif mode == "lhs2":
        rep_lhs2(sys.argv[2])
    elif mode == "calib":
        rep_calib(sys.argv[2], noise_folder=sys.argv[3] if len(sys.argv) > 3 else None)
    elif mode == "match":
        rep_match(sys.argv[2], sys.argv[3])
    elif mode == "remeasure":
        for d in sys.argv[2:]:
            remeasure(d)
