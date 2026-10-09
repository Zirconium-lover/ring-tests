"""История образца в два этапа (calib_plan.md, М12–М13): исходная структура — выпадение без нагрузки после
наводороживания; нагрев до T_max — гидриды растворяются до TSSD(T_max), крупнейшие остаются (водород сверх TSSD),
пластическая деформация матрицы и упрочнение остаются в металле (память о растворённых гидридах); затем
охлаждение под нагрузкой тем же движком.

Наводороживание: условия у Плясова не приведены — исходная структура выпадает при охлаждении от температуры полного
растворения (TSSD(H) + 5 °C, не ниже T_max) с той же скоростью, что в опыте (data_request_e635.md, п. 5).
"""
import dataclasses

import numpy as np

from ca_kinetic import KParams, run_kinetic
from thermo import TSS, T_line, Cooling


def stage1_params(p: KParams, rate1=None):
    L = TSS[p.lines] if isinstance(p.lines, str) else p.lines
    T1 = max(p.T_max, float(T_line(p.H_ppm, L["TSSD"])) + 5.0)
    return dataclasses.replace(p, sigma_app=0.0, sigma_app_grad=0.0, sigma_prof=(), sigma_T0=0.0, T_max=T1,
                               rate=rate1 or p.rate, rate2=0.0, init_plates=None, init_plast=None)


def survivors(plates, p: KParams):
    """Нерастворившиеся при T_max: крупнейшие пластинки исходной структуры общей площадью, равной доле водорода
    сверх TSSD(T_max) (как calib_kin.init_plates_for, но из того же движка)."""
    th = Cooling(p.H_ppm, p.T_max, p.T_end, p.lines)
    if th.frac_left <= 1e-5 or not len(plates):
        return None
    P0 = np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in plates])
    P0 = P0[np.lexsort((np.random.default_rng([p.seed, 3]).random(len(P0)), -P0[:, 3]))]
    area = np.cumsum(2 * P0[:, 3] * p.h_um) / (p.size_um[0] * p.size_um[1])
    return P0[: max(1, int(np.searchsorted(area, th.frac_left)))]


def run_history(p: KParams, rate1=None, memory=True, stage1=None):
    """→ (r1, r2): исходная структура и структура после охлаждения под нагрузкой. memory=False — без памяти:
    ε_p исходной структуры не переносится (нерастворившиеся пластинки — те же). stage1 — готовый r1 (общий для
    нескольких нагрузок)."""
    r1 = stage1 if stage1 is not None else run_kinetic(stage1_params(p, rate1))
    surv = survivors(r1["plates"], p)
    kw = dict(init_plates=surv)
    if memory and p.plast:
        kw["init_plast"] = ([a.copy() for a in r1["Ep_plast"]], r1["p_plast"].copy())
    r2 = run_kinetic(dataclasses.replace(p, **kw))
    return r1, r2
