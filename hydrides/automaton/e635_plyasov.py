"""Э635, трубы направляющих каналов: опыты Плясова и др. (Physics of Atomic Nuclei 86 (2023) 2604).

Материал: Э635 частично рекристаллизованный, зерно 3–10 мкм (берём 5 мкм), текстура канала Э635
(средний слой: f_R 0.504, f_T 0.377, f_L 0.119 → χ0 ≈ 39° при разбросе 26°), линии растворимости Э635,
таблица ореолов для σ_y = 226 МПа (data_halo/halo_tab_e635.npz).

Метод 1 (кольца, постоянная сила, 0.5–1 °C/мин после выдержки при 400 °C; ~160 ppm): порог 45 ± 9 МПа —
им подбирается одна величина, фора зёрен bias_dT.
Метод 2 (трубы под давлением газа: σmax 50–140 МПа при 400 °C, выдержка 2 ч, 0.5 °C/мин, напряжение
спадает ∝ абсолютной температуре): доля радиальных F_l(σmax) — предсказание.
Мера Плясова: F_l — доля длины гидридов, отклонённых от радиуса не больше чем на 45°, по оптическому
снимку (SIAMS), учитываются гидриды длиннее 5–10 мкм. До опыта F_l ≈ 0.10.

python e635_plyasov.py папка m1|m2|m2H|ring|zry [процессов]
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ca_kinetic import KParams, run_kinetic  # noqa: E402
from ca_analysis import render  # noqa: E402
from calib_beta import metrics  # noqa: E402
from calib_kin import init_plates_for  # noqa: E402

BASE = dict(lines="E635", T_max=400.0, T_end=100.0, chi0=39.0, chi_s=26.0, grain_um=(5.0, 5.0),
            B=80.0, Delta0=15.0, app_dT=0.08, gb=True, gb_dT=1.5, grow_kin=True, free_z=True, halo=True,
            halo_tab=os.path.join(HERE, "data_halo", "halo_tab_e635.npz"), cross_tol=15.0, sigma_cap=1e9)


def f_l(res, min_len_um=(5.0, 10.0), um=0.5, blur_um=1.0, level=0.25):
    """F_l по Плясову: связные гидриды на «оптическом» снимке, длина — размах вдоль главной оси,
    радиальный — если главная ось ближе 45° к радиусу (строки — ND)."""
    g, um = render(res, um_out=um, blur_um=blur_um)
    img = (1 - g) / 0.85
    lab, n = ndi.label(img > level, structure=np.ones((3, 3)))
    if n == 0:
        return {f"F_l_{m:g}": np.nan for m in min_len_um}
    idx = ndi.find_objects(lab)
    L, rad, yc = [], [], []
    for k, sl in enumerate(idx, start=1):
        ys, xs = np.nonzero(lab[sl] == k)
        if len(ys) < 3:
            continue
        yc.append((ys.mean() + sl[0].start) / lab.shape[0])
        P = np.stack([ys * um, xs * um], 1); P -= P.mean(0)
        w, v = np.linalg.eigh(P.T @ P)
        ax = v[:, -1]                                   # главная ось (y, x)
        proj = P @ ax
        L.append(proj.max() - proj.min() + um)
        rad.append(abs(ax[0]) >= np.cos(np.radians(45)))    # к радиусу (ND) ближе 45°
    L, rad, yc = np.array(L), np.array(rad), np.array(yc)
    out = {}
    for m in min_len_um:
        sel = L >= m
        out[f"F_l_{m:g}"] = float((L[sel] * rad[sel]).sum() / L[sel].sum()) if sel.any() else np.nan
        out[f"n_obj_{m:g}"] = int(sel.sum())
        # по третям стенки (строка 0 — наружная поверхность), как табл. 1 Плясова: по центру объекта
        for i, name in enumerate(("out", "mid", "in")):
            b = sel & (yc >= i / 3) & (yc < (i + 1) / 3)
            out[f"F_l_{m:g}_{name}"] = float((L[b] * rad[b]).sum() / L[b].sum()) if b.any() else np.nan
    return out


def job(a):
    out, kw = a
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k not in ("sigma_prof", "chi0_profile", "size_um"))
    fn = os.path.join(out, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    kp = dict(BASE, **{k: v for k, v in kw.items() if k != "ring"})
    p = KParams(**kp)
    if p.H_ppm > 0:
        init = init_plates_for(p)
        if init is not None:
            p = KParams(**dict(kp, init_plates=init))
    r = run_kinetic(p)
    m = metrics(r)
    m.update(f_l(r))
    new = [q for q in r["plates"] if not q.get("init")]
    L = np.array([2 * q["half"] for q in new]); gbk = np.array([q.get("kind") == "gb" for q in new], bool)
    m["GB_frac"] = float(L[gbk].sum() / L.sum()) if len(L) else np.nan
    m["T_first"] = new[0]["T"] if new else np.nan
    m.update({k: v for k, v in kw.items()}, time_s=time.time() - t0)
    np.savez_compressed(os.path.join(out, name + ".npz"),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]),
                        kind=np.array([{"intra": 0, "gb": 1, "init": 2}.get(q.get("kind"), -1) for q in r["plates"]]))
    json.dump(m, open(fn, "w"), default=float)


def cases(mode):
    if mode == "m1":       # кольца: постоянная нагрузка, ~160 ppm, 0.75 °C/мин; подбор форы по порогу 45 МПа
        return [dict(H_ppm=160.0, rate=0.75, bias_dT=b, sigma_app=s, seed=sd)
                for b in (1.0, 3.0, 5.0, 8.0) for s in (0, 30, 45, 60, 90) for sd in (1,)]
    if mode == "m2":       # трубы под давлением: напряжение задано при 400 °C и спадает; фора — из m1
        b = float(os.environ.get("E635_BIAS", "3.0"))
        return [dict(H_ppm=210.0, rate=0.5, bias_dT=b, sigma_app=s, sigma_T0=400.0, seed=sd)
                for s in (0, 50, 70, 90, 110, 140) for sd in (1, 2)]
    if mode == "m2H":      # полка F_l от водорода: у Плясова трубы 150–450 ppm, на графике водород не указан
        b = float(os.environ.get("E635_BIAS", "8.0"))
        return [dict(H_ppm=h, rate=0.5, bias_dT=b, sigma_app=s, sigma_T0=400.0, seed=1)
                for h in (150.0, 300.0, 400.0) for s in (50, 90, 140)]
    if mode == "ring":     # кольцо целиком: профиль σ по толщине (рис. 8, 9 при 400 °C), F_l по третям (табл. 1)
        prof = {"S1_0": ((0.0, 63.0), (1.0, -31.0)),                       # 200 Н, участок 0°: наружная растянута
                "S1_90": ((0.0, -130.0), (0.8, 195.0), (1.0, 215.0)),      # 200 Н, 90°: внутренняя, пласт. полка
                "S2_0": ((0.0, 81.0), (1.0, -15.0)),                        # 350 Н, 0° (рис. 9б, наклон 115 МПа/мм)
                "S2_90": ((0.0, -220.0), (0.63, 220.0), (1.0, 220.0))}      # 350 Н, 90°: оценка ×1.75 с пределом
        Hs = {"S1": 152.0, "S2": 168.0}
        return [dict(H_ppm=Hs[k[:2]], rate=0.75, bias_dT=b, sigma_prof=v, ring=k, size_um=(850.0, 240.0),
                     chi0_profile=(38.3, 38.7, 32.8), seed=1)
                for b in (5.0, 8.0) for k, v in prof.items()]
    if mode == "zry":      # для сравнения: фора холоднодеформированного Zircaloy-4 (17 °C) на Э635
        return [dict(H_ppm=210.0, rate=0.5, bias_dT=17.0, sigma_app=s, sigma_T0=400.0, seed=1)
                for s in (0, 50, 90, 140)]
    raise ValueError(mode)


if __name__ == "__main__":
    out, mode = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        list(pool.imap_unordered(job, [(out, kw) for kw in cases(mode)]))
    print("готово", mode)
