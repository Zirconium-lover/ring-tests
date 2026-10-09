"""Быстрые стенды плана калибровки (calib_plan.md, п. 5–6): полная модель — все механизмы, которые физически есть,
включены всегда; меняются только неопределённые параметры и спорные описания.

Точка плана = набор параметров + затравка: исходная структура (этап 1, без нагрузки) → для каждой нагрузки
охлаждение под ней с памятью (history.py). Меры — единым оператором съёмки (observe.py).

python bench.py папка режим [процессов]
  bias   — фора полной модели по порогу (σ*, где F_l = 0.25 — середина между «до опыта» и зоной; кольца: 45 ± 9 МПа)
  noise  — шум: номинальная точка, 8 затравок, поле 160 и 240 мкм
  knock  — выключение по одному из полной модели (диагностика вклада при всех прочих), 6 затравок
  lhs    — глобальный план (латинский гиперкуб) по неопределённым параметрам: полные индексы Соболя по суррогату
  grad   — С2: полоса 430 × 120 мкм с поверхностями, профиль +80 → −20 МПа (участки 0° колец)
  hyd    — С4: водород 160 / 300 / 450 ppm при 60 и 120 МПа
  tube   — трубы под давлением: двухосно (σz = σθ/2), σ ∝ T, 0.5 °C/мин
Номинальная фора — E635_BIAS (после режима bias).
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ca_kinetic import KParams, run_kinetic  # noqa: E402
from history import run_history, stage1_params  # noqa: E402
from observe import observe  # noqa: E402

# Полная модель: механизмы М1–М16 плана, параметры — измеренные, рассчитанные или номинал допущения
NOM = dict(lines="E635", T_max=400.0, T_end=100.0, H_ppm=160.0, rate=0.75, size_um=(160.0, 160.0),
           chi0=39.0, chi_s=26.0, grain_um=(5.0, 5.0),
           B=80.0, Delta0=380.0, tssp_ref=True,                 # зарождение: теория при γ = 0.18 Дж/м² (Jo 2025)
           app_dT=0.08, bias_dT=float(os.environ.get("E635_BIAS", "4.0")), dT_s=3.0,
           gb=True, gb_dT=1.5, gb_rule="trace", grow_kin=True, k_tip=0.05, cross_tol=15.0, L_max=6.0,
           free_z=True, sigma_cap=1e9, kappa=1.0, plast=True, plast_sy=200.0, plast_h=200.0,
           stress_diff=True, sh_cap=1e4, gorsky="cons")
SIG = (0.0, 30.0, 60.0, 100.0, 150.0)

# неопределённые параметры: (имя, низ, верх, шкала); mem_keep — доля ε_p, пережившая выдержку (возврат)
FACTORS = [("bias_dT", 0.0, 10.0, "lin"), ("app_dT", 0.06, 0.10, "lin"), ("Delta0", 250.0, 800.0, "log"),
           ("k_tip", 0.01, 0.25, "log"), ("gb_dT", 0.5, 3.0, "lin"), ("dT_s", 2.0, 5.0, "lin"),
           ("kappa", 0.5, 2.0, "log"), ("plast_sy", 150.0, 260.0, "lin"), ("plast_h", 50.0, 500.0, "log"),
           ("chi0", 33.0, 45.0, "lin"), ("chi_s", 18.0, 34.0, "lin"), ("grain", 3.0, 8.0, "lin"),
           ("L_max", 4.0, 12.0, "lin"), ("cross_tol", 0.0, 30.0, "lin"), ("mem_keep", 0.0, 1.0, "lin"),
           ("wang", 0.0, 1.0, "bin")]


def params_of(kw):
    kw = dict(kw)
    for k in ("mem_keep", "wang", "grain", "sigmas", "tag", "memory", "loads"):
        kw.pop(k, None)
    return kw


def point(a):
    out, tag, kw = a
    fn = os.path.join(out, tag + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    base = dict(NOM)
    base.update({k: v for k, v in kw.items() if k in NOM or k in ("seed", "sz_ratio", "walls_um", "v_h_over_rt", "sigma_T0")})
    if "grain" in kw:
        base["grain_um"] = (kw["grain"], kw["grain"])
    if kw.get("wang", 0) >= 0.5:
        base["gb_rule"] = "wang"
    p = KParams(**params_of(base))
    r1 = run_kinetic(stage1_params(p))
    res = dict(kw, tag=tag, before=observe(r1))
    keep = float(kw.get("mem_keep", 1.0))
    if keep < 1.0 and "Ep_plast" in r1:              # возврат при выдержке: часть ε_p и упрочнения снимается
        r1["Ep_plast"] = [a * keep for a in r1["Ep_plast"]]; r1["p_plast"] = r1["p_plast"] * keep
    store = {"before": np.asarray(r1["hyd"], np.float16)}
    loads = kw.get("loads") or [(f"s{s:g}", dict(sigma_app=float(s))) for s in kw.get("sigmas", SIG)]
    for label, ld in loads:
        ps = KParams(**dict(params_of(base), **ld))
        _, r2 = run_history(ps, stage1=r1, memory=kw.get("memory", True))
        res[label] = observe(r2)
        store[label] = np.asarray(r2["hyd"], np.float16)
    res["time_s"] = time.time() - t0
    np.savez_compressed(os.path.join(out, tag + ".npz"), dx=p.dx, **store)
    json.dump(res, open(fn, "w"), default=float)


def lhs(n, k, rng):
    u = (np.argsort(rng.random((k, n)), axis=1).T + rng.random((n, k))) / n
    return u


def cases(mode):
    if mode == "bias":
        return [(f"bias{b:g}_seed{s}", dict(bias_dT=float(b), seed=s, sigmas=(0.0, 30.0, 45.0, 60.0, 90.0, 120.0)))
                for b in (0, 2, 4, 6, 8, 10) for s in (1, 2)]
    if mode == "noise":
        return ([(f"nom160_seed{s}", dict(seed=s)) for s in range(1, 9)]
                + [(f"nom240_seed{s}", dict(seed=s, size_um=(240.0, 240.0))) for s in range(1, 9)])
    if mode == "knock":
        ko = {"no_memory": dict(memory=False), "no_stressdiff": dict(stress_diff=False), "gorsky_off": dict(gorsky="old", v_h_over_rt=0.0),
              "no_scatter": dict(dT_s=0.0), "no_gb": dict(gb=False), "no_cross": dict(cross_tol=0.0),
              "no_plast": dict(plast=False), "no_interact": dict(kappa=0.0), "wang": dict(wang=1.0),
              "sharp_nuc": dict(Delta0=15.0, tssp_ref=False)}
        return [(f"{name}_seed{s}", dict(kw, seed=s)) for name, kw in ko.items() for s in range(1, 7)]
    if mode == "grad":     # С2: полоса в половину стенки с поверхностями, профиль как у участков 0° (+80 → −20 МПа)
        return [(f"grad_seed{s}", dict(seed=s, size_um=(430.0, 120.0), walls_um=5.0,
                                       loads=[("prof", dict(sigma_prof=((0.0, 80.0), (1.0, -20.0))))])) for s in range(1, 5)]
    if mode == "hyd":      # С4: водород сверх TSSD(400 °C) — нерастворившиеся гидриды и полка F_l
        return [(f"H{h:g}_seed{s}", dict(seed=s, H_ppm=float(h), sigmas=(60.0, 120.0)))
                for h in (160, 300, 450) for s in range(1, 4)]
    if mode == "tube":     # трубы под давлением (рис. 4, 6): двухосно, напряжение спадает ∝ T, 0.5 °C/мин
        return [(f"tube_H{h:g}_seed{s}", dict(seed=s, H_ppm=float(h), rate=0.5, sz_ratio=0.5, sigma_T0=400.0,
                                              sigmas=(0.0, 50.0, 70.0, 90.0, 110.0, 140.0)))
                for h in (160, 300, 450) for s in range(1, 3)]
    if mode == "lhs":
        n = int(os.environ.get("LHS_N", "200"))
        U = lhs(n, len(FACTORS), np.random.default_rng(2026))
        out = []
        for i, u in enumerate(U):
            kw = dict(seed=1000 + i)
            for (name, lo, hi, sc), x in zip(FACTORS, u):
                kw[name] = float(lo * (hi / lo) ** x) if sc == "log" else (float(x >= 0.5) if sc == "bin" else float(lo + (hi - lo) * x))
            out.append((f"lhs{i:03d}", kw))
        return out
    raise ValueError(mode)


if __name__ == "__main__":
    out, mode = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    jobs = [(out, tag, kw) for tag, kw in cases(mode)]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for i, _ in enumerate(pool.imap_unordered(point, jobs)):
            print(f"{i + 1}/{len(jobs)}", time.strftime("%H:%M:%S"), flush=True)
    print("готово", mode)
