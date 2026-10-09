"""Разброс остаточного напряжения по зёрнам для автомата (dT_s): часть, не объясняемая углом ψ.

Автомат берёт фору зёрен как bias_dT·cos²ψ (ψ — угол оси c к радиусу в сечении), то есть только среднюю зависимость
от ориентации. Самосогласованная модель (esc_grains + vsc_grains: Крёнер, ползучесть по призме, базису,
пирамиде ⟨c+a⟩) даёт напряжение каждого зерна: оно зависит от всей ориентации (наклон оси c к оси трубы,
поворот вокруг c), поэтому при одном ψ есть разброс. Проход изготовления (пилигрим) масштабируется так, чтобы
средняя разница σ_nn «радиальные − окружные зёрна» дала фору bias_dT при 0.08 °C/МПа (Vizcaíno); тогда разброс —
предсказание модели без новой подгонки. Добавляется тепловое остаточное (охлаждение после отжига на ΔT).
Самосогласованная модель не видит соседей — это нижняя оценка разброса (Holden 2002: измеренные межзёренные
деформации после деформации больше, чем даёт EPSC).

python grain_scatter.py [фора °C] [ΔT охлаждения, K]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from esc_grains import ESC, ZR_A, ZR_C, families, t2m, texture  # noqa: E402
from vsc_grains import Creep, solve_eig  # noqa: E402

APP_DT = 0.08          # °C/МПа


def main():
    bias = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    dT_cool = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
    target = -bias / APP_DT                                    # рад − окр, МПа
    out = {}
    for tex_name, (chi0, chis) in {"Э635 НК (χ0 39°, разброс 26°)": (39.0, 26.0)}.items():
        cax = texture(1500, chi0=chi0, chi_s=chis, seed=5)
        e = ESC(cax, ZR_C["RT"], ZR_A["turner"])
        fam, psi = families(cax)
        inpl = ~fam["axial"]
        for name, (rb, rp) in {"Holden 2002": (160 / 90, 240 / 90), "Turner 1994": (5.0, 2.25)}.items():
            cr = Creep(e, dict(prism=1.0, basal=rb, pyr=rp), n=20.0)
            eta, _ = cr.run(t2m(np.diag([-50.0, -100.0, 50.0])), eps_target=5e-3, nrec=5)
            s_pass = e.snn(solve_eig(e, np.zeros(6), eta))
            s_th = e.snn(solve_eig(e, np.zeros(6), np.einsum("g,gi->gi", np.full(len(cax), -dT_cool), e.ag)))
            d = lambda s: s[fam["rad"]].mean() - s[fam["circ"]].mean()
            k = (target - d(s_th)) / d(s_pass)                    # масштаб прохода под заданную фору
            s = k * s_pass + s_th
            c2 = np.cos(np.radians(psi)) ** 2                     # cos²ψ: 1 — ось c по радиусу (окружные пластинки)
            A = np.stack([np.ones(inpl.sum()), c2[inpl]], 1)
            coef, *_ = np.linalg.lstsq(A, s[inpl], rcond=None)
            resid = s[inpl] - A @ coef
            sd = float(resid.std())
            fam_sd = {f: float(s[m].std()) for f, m in fam.items() if m.any()}
            r = dict(scale=float(k), d_rad_circ=float(d(s)), slope_cos2=float(coef[1]), sd_resid_MPa=sd,
                     dT_s=APP_DT * sd, sd_family=fam_sd, d_thermal=float(d(s_th)))
            out[f"{tex_name} | {name}"] = r
            print(f"{tex_name} | τc {name}: рад − окр {d(s):6.1f} МПа (тепловое {d(s_th):5.1f}), наклон по cos²ψ "
                  f"{coef[1]:6.1f} МПа; разброс при том же ψ ±{sd:5.1f} МПа → dT_s = {APP_DT * sd:4.2f} °C; "
                  f"разброс внутри семейств: " + ", ".join(f"{f} ±{v:.0f}" for f, v in fam_sd.items()), flush=True)
    json.dump(out, open(os.path.join(HERE, "grain_scatter.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
