"""Разбор полей tip_runs.py: гидростатическое напряжение у кромки пластинки относительно среднего, обогащение
водородом (Горский, local equilibrium) и работа следующей пластинки колоды в местах укладки.

Оси: x — θ, y — r, z — ось трубы. Радиальная пластинка: нормаль x, диск в плоскости (y, z); окружная: нормаль y,
диск в (x, z). Зона кромки — кольцо вокруг края диска: расстояние от оси диска R…R + w в его плоскости, по нормали
|q| < t. Места укладки «колоды» — над и под краем: в плоскости R − 1…R + 1 мкм, по нормали 0.6…1.6 мкм.
Обогащение у кромки c/c_far = exp(V_H (σ_h − σ_h,far)/RT), в градусах форы — ΔT = V_H Δσ_h T/Q (T = 603 K,
Q = 32.2 кДж/моль — наклон TSSP Э635). python tip_analyze.py папка"""
import os
import sys
import json
import glob
import numpy as np

N, H, R, TH = 96, 0.15, 2.5, 0.6
EPS_N, EPS_T = 0.0720, 0.0458
V_H, RG, T, Q = 1.7e-6, 8.314, 603.0, 32.2e3
K_ENR = V_H * 1e6 / (RG * T)                   # 1/МПа
K_DEG = V_H * 1e6 * T / Q                      # °C/МПа гидростатического


def coords():
    x = (np.arange(N) + 0.5) * H - N * H / 2
    return np.meshgrid(x, x, x, indexing="ij")


def zones(orient):
    X, Y, Z = coords()
    q, a1, a2 = (X, Y, Z) if orient == "rad" else (Y, X, Z)
    rho = np.sqrt(a1 ** 2 + a2 ** 2)
    tip = (rho > R) & (rho < R + 1.0) & (np.abs(q) < 0.45)
    tip_near = (rho > R) & (rho < R + 0.5) & (np.abs(q) < 0.3)
    stack = (rho > R - 1.0) & (rho < R + 1.0) & (np.abs(q) > 0.6) & (np.abs(q) < 1.6)
    return tip, tip_near, stack


def g_next(sig, normal):
    """σ:ε*/ε_n для новой пластинки с нормалью по оси normal (Мандел: сдвиги √2 — скалярное произведение верно)."""
    e = np.full(3, EPS_T); e[normal] = EPS_N
    return (sig[0] * e[0] + sig[1] * e[1] + sig[2] * e[2]) / EPS_N


if __name__ == "__main__":
    d = sys.argv[1]
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        case = os.path.basename(f)[:-4]
        orient, lk = case.split("_")
        m = json.load(open(f[:-4] + ".json"))
        z = np.load(f)
        sig = z["sig"].astype(np.float64); p = z["p"]
        sh = (sig[0] + sig[1] + sig[2]) / 3.0
        sh_far = (m["s_theta"] + m["s_z"]) / 3.0
        tip, tip_near, stack = zones(orient)
        X, Y, Zc = coords()
        q_, a1_, a2_ = (X, Y, Zc) if orient == "rad" else (Y, X, Zc)
        dist = np.maximum(np.sqrt(a1_ ** 2 + a2_ ** 2) - R, 0.0) ** 2 + np.maximum(np.abs(q_) - TH / 2, 0.0) ** 2
        near = (np.sqrt(dist) < 2.0) & ~z["mask"]                        # матрица в 2 мкм от пластинки
        normal = 0 if orient == "rad" else 1
        gn = g_next(sig, normal)
        out[case] = dict(orient=orient, load=lk, s_theta=m["s_theta"], s_z=m["s_z"], W=m["W"],
                         sh_tip=float(sh[tip].mean() - sh_far), sh_tip_near=float(sh[tip_near].mean() - sh_far),
                         sh_tip_max=float(sh[tip].max() - sh_far),
                         g_stack=float(gn[stack].mean()), g_stack_max=float(gn[stack].max()),
                         sh_stack=float(sh[stack].mean() - sh_far), sh_stack_max=float(sh[stack].max() - sh_far),
                         sh_near_max=float(sh[near].max() - sh_far),
                         p_tip=float(p[tip].mean()), pl_vol=float((p > 1e-4).sum() * H ** 3))
    # приращения от нагрузки относительно того же расчёта без нагрузки
    for case, r in out.items():
        base = out.get(r["orient"] + "_0")
        if base:
            for k in ("sh_tip", "sh_tip_near", "g_stack", "sh_stack", "sh_near_max"):
                r["d_" + k] = r[k] - base[k]
            r["dT_H_tip"] = K_DEG * r["d_sh_tip"]                       # фора от обогащения водородом, °C
            r["dT_mech_stack"] = EPS_N * r["d_g_stack"] / 5.4           # фора от работы следующей пластинки, °C
    json.dump(out, open(os.path.join(d, "tip_summary.json"), "w"), indent=1)
    print(f"{'случай':10s} {'σθ':>4s} {'σz':>4s} | σh−far: кромка  укладка  макс(2мкм) | Δ от нагр.: кромка укладка макс  g_укл | ΔT: H  мех | пласт. мкм³")
    for case in sorted(out, key=lambda c: (out[c]["orient"], out[c]["s_theta"], out[c]["s_z"])):
        r = out[case]
        print(f"{case:10s} {r['s_theta']:4.0f} {r['s_z']:4.0f} | {r['sh_tip']:13.0f} {r['sh_stack']:8.0f} {r['sh_near_max']:9.0f} |"
              f" {r.get('d_sh_tip', 0):12.1f} {r.get('d_sh_stack', 0):7.1f} {r.get('d_sh_near_max', 0):5.1f} {r.get('d_g_stack', 0):6.1f} |"
              f" {r.get('dT_H_tip', 0):5.2f} {r.get('dT_mech_stack', 0):5.2f} | {r['pl_vol']:7.1f}")
