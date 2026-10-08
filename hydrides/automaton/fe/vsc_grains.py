"""Ползучесть по системам скольжения в самосогласованной модели: как нагрузка перераспределяется между
семействами зёрен при одноосном и двухосном растяжении.

Зерно ползёт по степенному закону на системах скольжения ГПУ: γ̇_s = γ̇0·|τ_s/τc_s|ⁿ·sign τ_s,
призма ⟨a⟩ (3), базис ⟨a⟩ (3), пирамида ⟨c+a⟩ {10-11}⟨11-23⟩ (12); двойникование не учтено.
Пластическая деформация зерна — собственная деформация в упругой самосогласованной задаче
(esc_grains.ESC; взаимодействие по Крёнеру — верхняя оценка межзёренных напряжений, знак надёжен).
Критические напряжения неизвестны (ждём Turner & Tomé 1994), поэтому перебор отношений
базис/призма и пирамида/призма. Установившееся распределение от γ̇0 не зависит; время — в единицах
макроскопической деформации ползучести.

Вопросы: (1) растёт ли σ_nn радиального семейства относительно окружного при ползучести под σθ —
то есть усиливает ли ползучесть выбор радиальных; (2) меняет ли σz эту разницу (двухосность Cinbiz);
(3) какие остаточные напряжения по семействам остаются после разгрузки.

Запуск: python fe/vsc_grains.py
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from esc_grains import ESC, ZR_A, ZR_C, families, frame_from_c, m2t, t2m, texture  # noqa: E402

CA = 1.593                                          # c/a циркония


def mb_dir(U, V, T, W):
    u, v, w = U - T, V - T, W
    a1 = np.array([1.0, 0, 0]); a2 = np.array([-0.5, np.sqrt(3) / 2, 0]); c = np.array([0, 0, CA])
    d = u * a1 + v * a2 + w * c
    return d / np.linalg.norm(d)


def mb_plane(h, k, i, l):
    a1s = np.array([1.0, 1 / np.sqrt(3), 0]); a2s = np.array([0, 2 / np.sqrt(3), 0]); cs = np.array([0, 0, 1 / CA])
    n = h * a1s + k * a2s + l * cs
    return n / np.linalg.norm(n)


def family_perms(hkil):
    """Все варианты семейства {hkil}/⟨UVTW⟩: перестановки первых трёх индексов и знаки."""
    out = set()
    for p in itertools.permutations(hkil[:3]):
        for s3 in (1, -1):
            for s4 in (1, -1):
                v = (s3 * p[0], s3 * p[1], s3 * p[2], s4 * hkil[3])
                out.add(v)
    return out


def systems(kind):
    if kind == "basal":
        pl, di = [(0, 0, 0, 1)], family_perms((2, -1, -1, 0))
    elif kind == "prism":
        pl, di = family_perms((1, 0, -1, 0)), family_perms((2, -1, -1, 0))
    else:                                             # pyramidal <c+a>
        pl, di = family_perms((1, 0, -1, 1)), family_perms((1, 1, -2, 3))
    res, seen = [], set()
    for P in pl:
        n = mb_plane(*P)
        for D in di:
            d = mb_dir(*D)
            if abs(n @ d) > 1e-8:
                continue
            m = 0.5 * (np.outer(d, n) + np.outer(n, d))
            v = np.round(t2m(m), 6)
            nz = v[np.abs(v) > 1e-9]
            key = tuple(v * np.sign(nz[0]))                              # система и её противоположная — одна
            if key in seen:
                continue
            seen.add(key)
            res.append(t2m(m))
    return np.array(res)


SYS = {k: systems(k) for k in ("prism", "basal", "pyr")}


class Creep:
    def __init__(self, esc, tau, n=5.0):
        self.e = esc
        self.n = n
        ms, tc = [], []
        for k, t in tau.items():
            ms.append(SYS[k]); tc += [t] * len(SYS[k])
        self.m_cr = np.concatenate(ms)                                   # Шмид в осях кристалла
        self.tc = np.array(tc)
        self.mg = []                                                     # Шмид в осях образца по зёрнам
        from esc_grains import rot6
        for c in esc.cax:
            Q = rot6(frame_from_c(c))
            self.mg.append(self.m_cr @ Q.T)
        self.mg = np.array(self.mg)                                      # (g, s, 6)

    def rate(self, sig):
        tau = np.einsum("gsi,gi->gs", self.mg, sig)
        g = np.sign(tau) * np.abs(tau / self.tc) ** self.n
        return np.einsum("gs,gsi->gi", g, self.mg)                      # γ̇0 = 1

    def run(self, Sigma, eps_target=5e-3, eta0=None, nrec=40):
        e = self.e
        eta = np.zeros((len(e.cax), 6)) if eta0 is None else eta0.copy()
        start = eta.mean(0).copy()                                      # деформация ползучести — от начала прогона
        rec, macro = [], 0.0
        marks = list(np.linspace(0, eps_target, nrec + 1)[1:])
        while marks:
            sig = solve_eig(e, Sigma, eta)
            r = self.rate(sig)
            rm = np.abs(r).max()
            dt = 2e-6 / max(rm, 1e-30)                                   # шаг: Δη ≤ 2·10⁻⁶ в любом зерне
            eta += r * dt
            macro = float(np.linalg.norm(eta.mean(0) - start) * np.sqrt(2 / 3))   # эквивалентная макродеформация
            if macro >= marks[0]:
                marks.pop(0)
                rec.append((macro, sig.copy()))
        return eta, rec


def solve_eig(e, Sigma, eta):
    """ESC.solve с произвольной собственной деформацией зёрен."""
    b = np.einsum("gij,gjk,gk->i", e.Bi, e.Cg, eta) / len(e.cax)
    E = np.linalg.solve(e.I6 - e.M @ e.L, e.M @ Sigma + b)
    rhs = np.einsum("gij,gj->gi", e.Cg, eta) + Sigma + e.L @ E
    eps = np.einsum("gij,gj->gi", e.Bi, rhs)
    return np.einsum("gij,gj->gi", e.Cg, eps - eta)


def fam_stats(e, fam, sig):
    s = e.snn(sig)
    return {k: float(s[v].mean()) for k, v in fam.items() if v.any()}


def main():
    cax = texture(600, seed=3)
    e = ESC(cax, ZR_C["600K"], ZR_A["fit"])
    fam, psi = families(cax)
    loads = {"hoop": np.diag([100.0, 0, 0]), "closed_tube": np.diag([100.0, 0, 50.0]),
             "biax_0.8": np.diag([100.0, 0, 80.0])}
    ratios = [(1.5, 3.0), (3.0, 3.0), (1.5, 5.0), (5.0, 5.0)]
    out = []
    for rb, rp in ratios:
        cr = Creep(e, dict(prism=1.0, basal=rb, pyr=rp), n=5.0)
        for name, S in loads.items():
            Sig = t2m(S)
            el = fam_stats(e, fam, solve_eig(e, Sig, np.zeros((len(cax), 6))))
            eta, rec = cr.run(Sig, eps_target=3e-3)
            cr_end = fam_stats(e, fam, rec[-1][1])
            res_unl = fam_stats(e, fam, solve_eig(e, np.zeros(6), eta))     # после разгрузки
            curve = [(m, fam_stats(e, fam, s)) for m, s in rec[::5]]
            r = dict(basal=rb, pyr=rp, load=name, elastic=el, creep=cr_end, residual=res_unl,
                     d_el=el["rad"] - el["circ"], d_cr=cr_end["rad"] - cr_end["circ"],
                     d_res=res_unl["rad"] - res_unl["circ"],
                     curve=[(m, f["rad"] - f["circ"]) for m, f in curve])
            out.append(r)
            print(f"базис/призма {rb:3.1f} пирамида/призма {rp:3.1f} {name:12s} рад−окр: упруго {r['d_el']:6.1f}"
                  f" → ползучесть {r['d_cr']:6.1f}; остаток после разгрузки {r['d_res']:6.1f}  "
                  f"(окр {cr_end['circ']:6.1f}, рад {cr_end['rad']:6.1f})", flush=True)
    json.dump(out, open(os.path.join(HERE, "vsc_grains.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
