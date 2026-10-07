"""Напряжения по семействам зёрен: упругая самосогласованная модель поликристалла (Кренер–Эшелби–Хилл)
с анизотропией монокристалла Zr и анизотропным тепловым расширением.

Вопросы:
  1. Может ли охлаждение (разное сжатие зёрен по a и c) дать фору зёрен: нормальное напряжение на
     базисной плоскости у зёрен «радиального» семейства (ось c к окружности) меньше, чем у «окружного»
     (ось c к радиусу)? Фора 12 °C при 0.08 °C/МПа — это ~150 МПа разницы.
  2. Даёт ли упругая анизотропия эффект осевого напряжения σz на ту же разницу (двухосность)?

Зерно g: ε_g = (C_g + L*)⁻¹ (C_g ε_th,g + Σ + L* E), σ_g = C_g (ε_g − ε_th,g); L* = P⁻¹ − C* — тензор
Хилла для сферического включения в эффективной среде C*, P — интеграл по единичной сфере. C* находится
самосогласованно, E — из ⟨ε_g⟩ = E при заданном Σ. Нотация Манделя (6 компонент, сдвиги ×√2).
Оси образца: x = TD (θ), y = ND (r), z = ось трубы (как в ca3d).

Запуск: python fe/esc_grains.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = np.sqrt(2.0)
PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

# монокристалл Zr: упругие константы (ГПа), Fisher & Renken 1964, комнатная температура и ~600 K (оценка)
ZR_C = {"RT": dict(C11=143.4, C12=72.8, C13=65.3, C33=164.8, C44=32.0),
        "600K": dict(C11=128.0, C12=76.0, C13=66.0, C33=155.0, C44=27.0)}
# тепловое расширение по осям a и c, 1/K (подгонки a(T), c(T) α-Zr; вторая пара — часто цитируемые значения)
ZR_A = {"fit": (5.15e-6, 9.21e-6), "alt": (5.7e-6, 10.3e-6)}


def hex_C(C11, C12, C13, C33, C44):
    C66 = 0.5 * (C11 - C12)
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C11
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    C[2, 2] = C33
    C[3, 3] = C[4, 4] = 2 * C44                       # Мандель: 2·C44
    C[5, 5] = 2 * C66
    return C * 1e3                                    # МПа


def t2m(t):
    return np.array([t[0, 0], t[1, 1], t[2, 2], S2 * t[1, 2], S2 * t[0, 2], S2 * t[0, 1]])


def m2t(v):
    return np.array([[v[0], v[5] / S2, v[4] / S2], [v[5] / S2, v[1], v[3] / S2], [v[4] / S2, v[3] / S2, v[2]]])


def rot6(R):
    """Матрица поворота в нотации Манделя: v' = Q v для тензора t' = R t Rᵀ."""
    Q = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6); e[j] = 1.0
        Q[:, j] = t2m(R @ m2t(e) @ R.T)
    return Q


def frame_from_c(c):
    """Поворот кристалл → образец, переводящий ось кристалла z в c (поворот вокруг c не важен: ГПУ
    упруго и термически трансверсально изотропен)."""
    c = c / np.linalg.norm(c)
    a = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = a - c * (a @ c); a /= np.linalg.norm(a)
    b = np.cross(c, a)
    return np.stack([a, b, c], 1)


def sphere(n=3000):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


XI = sphere()


def hill_P(C):
    """P_ijkl = (1/4π)∮ sym[ξ_j (K⁻¹)_ik ξ_l] dS, K_ik = C_ijkl ξ_j ξ_l (сферическое включение)."""
    Ct = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(PAIRS):
        for b, (k, l) in enumerate(PAIRS):
            v = C[a, b] / ((S2 if i != j else 1) * (S2 if k != l else 1))
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    Ct[p, q, r, s] = v
    K = np.einsum("ijkl,nj,nl->nik", Ct, XI, XI)
    Ki = np.linalg.inv(K)
    T = np.einsum("nj,nik,nl->nijkl", XI, Ki, XI).mean(0)       # (1/4π)∮ = среднее по сфере
    T = 0.25 * (T + T.transpose(1, 0, 2, 3) + T.transpose(0, 1, 3, 2) + T.transpose(1, 0, 3, 2))
    P = np.zeros((6, 6))
    for a, (i, j) in enumerate(PAIRS):
        for b, (k, l) in enumerate(PAIRS):
            P[a, b] = T[i, j, k, l] * (S2 if i != j else 1) * (S2 if k != l else 1)
    return P


def texture(n, chi0=30.0, chi_s=26.0, chi_sL=21.0, seed=1):
    """Оси c как в ca3d.make_grains3: наклон от ND к TD ±chi0 с разбросом chi_s, к оси трубы — chi_sL."""
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1.0, 1.0], n)
    al = np.radians(sign * chi0 + rng.normal(0, chi_s, n))
    ga = np.radians(rng.normal(0, chi_sL, n))
    return np.stack([np.sin(al) * np.cos(ga), np.cos(al) * np.cos(ga), np.sin(ga)], 1)


class ESC:
    def __init__(self, cax, Cc, alpha):
        self.cax = cax
        self.Cg, self.ag = [], []
        Cx = hex_C(**Cc)
        ax = t2m(np.diag([alpha[0], alpha[0], alpha[1]]))
        for c in cax:
            Q = rot6(frame_from_c(c))
            self.Cg.append(Q @ Cx @ Q.T)
            self.ag.append(Q @ ax)
        self.Cg, self.ag = np.array(self.Cg), np.array(self.ag)
        C = self.Cg.mean(0)                                         # старт — Фойгт
        I6 = np.eye(6)
        for _ in range(60):
            L = np.linalg.inv(hill_P(C)) - C
            A = np.linalg.solve(self.Cg + L, np.broadcast_to(C + L, self.Cg.shape))
            Cn = np.einsum("gij,gjk->ik", self.Cg, A) / len(cax)
            Cn = 0.5 * (Cn + Cn.T)
            if np.abs(Cn - C).max() < 1e-3:
                C = Cn
                break
            C = Cn
        self.C, self.L = C, np.linalg.inv(hill_P(C)) - C
        self.Bi = np.linalg.inv(self.Cg + self.L)                   # (C_g + L*)⁻¹
        self.M = self.Bi.mean(0)
        self.I6 = I6

    def solve(self, Sigma, dT):
        """Σ (Мандель, МПа) и ΔT (K) → напряжения в зёрнах (n, 6)."""
        eth = self.ag * dT
        b = np.einsum("gij,gjk,gk->i", self.Bi, self.Cg, eth) / len(self.cax)
        E = np.linalg.solve(self.I6 - self.M @ self.L, self.M @ Sigma + b)
        rhs = np.einsum("gij,gj->gi", self.Cg, eth) + Sigma + self.L @ E
        eps = np.einsum("gij,gj->gi", self.Bi, rhs)
        sig = np.einsum("gij,gj->gi", self.Cg, eps - eth)
        return sig, E

    def snn(self, sig):
        """Нормальное напряжение на базисной плоскости (≈ на пластинке) в каждом зерне."""
        return np.array([c @ m2t(s) @ c for c, s in zip(self.cax, sig)])


def families(cax):
    """Угол следа базисной пластинки к окружности в сечении r–θ: ψ = угол оси c от радиуса к окружности."""
    psi = np.degrees(np.arctan2(np.abs(cax[:, 0]), np.abs(cax[:, 1])))
    inplane = np.abs(cax[:, 2]) < np.sin(np.radians(30))        # ось c не слишком к оси трубы
    return dict(circ=inplane & (psi < 40), mixed=inplane & (psi >= 40) & (psi < 65),
                rad=inplane & (psi >= 65), axial=~inplane), psi


def run(n=1500, Cset="RT", aset="fit", seed=1):
    cax = texture(n, seed=seed)
    m = ESC(cax, ZR_C[Cset], ZR_A[aset])
    fam, psi = families(cax)
    out = dict(n=n, C=Cset, alpha=aset, frac={k: float(v.mean()) for k, v in fam.items()},
               kearns=dict(TD=float((cax[:, 0] ** 2).mean()), ND=float((cax[:, 1] ** 2).mean()),
                           L=float((cax[:, 2] ** 2).mean())))
    loads = {"thermal_-100K": (np.zeros(6), -100.0),
             "hoop_100": (t2m(np.diag([100.0, 0, 0])), 0.0),
             "hoop_100_axial_80": (t2m(np.diag([100.0, 0, 80.0])), 0.0),
             "axial_100": (t2m(np.diag([0, 0, 100.0])), 0.0),
             "radial_-100": (t2m(np.diag([0, -100.0, 0])), 0.0)}
    for name, (S, dT) in loads.items():
        sig, E = m.solve(S, dT)
        s = m.snn(sig)
        mean_sig = sig.mean(0)
        r = {k: dict(mean=float(s[v].mean()), sd=float(s[v].std())) for k, v in fam.items() if v.any()}
        r["rad_minus_circ"] = r["rad"]["mean"] - r["circ"]["mean"]
        r["check_mean_stress"] = [round(float(x), 3) for x in mean_sig]
        # ход по углу ψ: среднее σ_nn в бинах по 10°
        bins = np.arange(0, 91, 10)
        k = np.digitize(psi, bins) - 1
        inpl = ~fam["axial"]
        r["by_psi"] = [float(s[(k == i) & inpl].mean()) if ((k == i) & inpl).any() else None for i in range(9)]
        out[name] = r
        print(f"{Cset}/{aset} {name:20s} σnn: окр {r['circ']['mean']:7.2f} ± {r['circ']['sd']:5.2f}  "
              f"рад {r['rad']['mean']:7.2f} ± {r['rad']['sd']:5.2f}  рад−окр {r['rad_minus_circ']:7.2f}  "
              f"⟨σ⟩ {r['check_mean_stress']}")
    return out


if __name__ == "__main__":
    res = [run(Cset=c, aset=a) for c in ("RT", "600K") for a in ("fit", "alt")]
    json.dump(res, open(os.path.join(HERE, "esc_grains.json"), "w"), indent=1)
