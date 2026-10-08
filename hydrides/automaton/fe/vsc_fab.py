"""Остаточные напряжения по семействам после пластического «прохода» изготовления (вне ползучести под
нагрузкой опыта): деформация под Σ, разгрузка, затем частичная релаксация при Σ = 0 (отжиг).
Те же допущения, что в vsc_grains.py (Крёнер — верхняя оценка; отношения τc перебираются).
Нужный знак для форы: у окружного семейства σ_nn больше, чем у радиального (рад − окр < 0), ~150 МПа.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from esc_grains import ESC, ZR_A, ZR_C, families, t2m, texture  # noqa: E402
from vsc_grains import Creep, fam_stats, solve_eig  # noqa: E402


def relax(cr, eta, frac_keep=0.5):
    """Отжиг при Σ = 0: ползучесть до тех пор, пока разница рад − окр не упадёт до frac_keep от начальной."""
    e = cr.e
    fam, _ = families(e.cax)
    d0 = None
    for _ in range(200000):
        sig = solve_eig(e, np.zeros(6), eta)
        f = fam_stats(e, fam, sig)
        d = f["rad"] - f["circ"]
        d0 = d if d0 is None else d0
        if abs(d) <= frac_keep * abs(d0):
            return eta, f
        r = cr.rate(sig)
        eta = eta + r * 2e-6 / max(np.abs(r).max(), 1e-30)
    return eta, f


def main():
    cax = texture(600, seed=3)
    e = ESC(cax, ZR_C["RT"], ZR_A["fit"])
    fam, _ = families(cax)
    paths = {"r_comp": np.diag([0.0, -100.0, 0.0]),
             "thin_elong": np.diag([0.0, -100.0, 100.0]),
             "pilger": np.diag([-50.0, -100.0, 50.0])}
    out = []
    for rb, rp in [(1.5, 3.0), (3.0, 3.0), (1.5, 5.0), (5.0, 5.0)]:
        cr = Creep(e, dict(prism=1.0, basal=rb, pyr=rp), n=20.0)          # почти пластичность
        for name, S in paths.items():
            eta, _ = cr.run(t2m(S), eps_target=5e-3, nrec=5)
            f_res = fam_stats(e, fam, solve_eig(e, np.zeros(6), eta))
            r = dict(basal=rb, pyr=rp, path=name, residual=f_res, d_res=f_res["rad"] - f_res["circ"])
            out.append(r)
            print(f"базис/призма {rb:3.1f} пирамида/призма {rp:3.1f} {name:11s} остаток: окр {f_res['circ']:7.1f} "
                  f"рад {f_res['rad']:7.1f}  рад−окр {r['d_res']:7.1f}  (на 100 МПа нагрузки прохода)", flush=True)
    hist = history(e, fam)
    lit = literature()
    json.dump(dict(passes=out, history=hist, literature=lit), open(os.path.join(HERE, "vsc_fab.json"), "w"), indent=1)


def literature():
    """Отношения τc по литературе: Holden и др. 2002 (Pang): призма 90, базис 160, пирамида 240 МПа;
    Turner & Tomé 1994: призма 110, пирамида 247.5 (базис не активен — берём 5). α по Turner & Tomé."""
    cax = texture(600, seed=3)
    e = ESC(cax, ZR_C["RT"], ZR_A["turner"])
    fam, _ = families(cax)
    out = []
    for name, (rb, rp) in {"Holden2002": (160 / 90, 240 / 90), "Turner1994": (5.0, 2.25)}.items():
        cr = Creep(e, dict(prism=1.0, basal=rb, pyr=rp), n=20.0)
        for pname, S in {"pilger": np.diag([-50.0, -100.0, 50.0]), "thin_elong": np.diag([0.0, -100.0, 100.0])}.items():
            eta, _ = cr.run(t2m(S), eps_target=5e-3, nrec=5)
            f = fam_stats(e, fam, solve_eig(e, np.zeros(6), eta))
            out.append(dict(set=name, basal=rb, pyr=rp, path=pname, d_res=f["rad"] - f["circ"], residual=f))
            print(f"{name:10s} {pname:10s} остаток рад − окр {f['rad'] - f['circ']:6.1f} МПа на 100 МПа прохода", flush=True)
    th = fam_stats(e, fam, solve_eig(e, np.zeros(6), np.einsum("g,gi->gi", np.full(len(cax), -100.0), e.ag)))
    print(f"тепловое (α Turner), охлаждение на 100 K: рад − окр {th['rad'] - th['circ']:.1f} МПа", flush=True)
    return dict(passes=out, thermal_100K=th)


def history(e, fam, sig_pass=500.0, keep=0.5):
    """Проход «прокатки» (напряжение sig_pass), отжиг до keep остатка форы, затем ползучесть под нагрузкой
    опыта: окружное 150 МПа или двухосное 150/120. Как меняется рад − окр по мере ползучести."""
    out = []
    for rb, rp in [(1.5, 3.0), (3.0, 3.0), (1.5, 5.0), (5.0, 5.0)]:
        tau = dict(prism=1.0, basal=rb, pyr=rp)
        eta, _ = Creep(e, tau, n=20.0).run(t2m(np.diag([-50.0, -100.0, 50.0])), eps_target=5e-3, nrec=5)
        eta = eta * sig_pass / 100.0                                   # остаток ∝ напряжению прохода
        cr = Creep(e, tau, n=5.0)
        eta, f0 = relax(cr, eta, keep)
        for name, S in (("hoop_150", np.diag([150.0, 0, 0])), ("biax_150_120", np.diag([150.0, 0, 120.0]))):
            Sig = t2m(S)
            f_load = fam_stats(e, fam, solve_eig(e, Sig, eta))
            _, rec = cr.run(Sig, eps_target=3e-3, eta0=eta, nrec=6)
            curve = [(round(m, 5), round(fam_stats(e, fam, s_)["rad"] - fam_stats(e, fam, s_)["circ"], 1)) for m, s_ in rec]
            out.append(dict(basal=rb, pyr=rp, load=name, d_after_anneal=f0["rad"] - f0["circ"],
                            d_loaded=f_load["rad"] - f_load["circ"], curve=curve))
            print(f"базис/призма {rb:3.1f} пирамида/призма {rp:3.1f} {name:13s} фора после отжига "
                  f"{f0['rad'] - f0['circ']:7.1f}; под нагрузкой {f_load['rad'] - f_load['circ']:7.1f}; "
                  f"по ходу ползучести: {curve}", flush=True)
    return out


if __name__ == "__main__":
    if "--lit" in sys.argv:
        literature()
    else:
        main()
