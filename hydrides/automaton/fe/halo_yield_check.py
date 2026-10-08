"""Складываются ли ореолы у предела текучести? Э635 (σ_y 226 МПа, упрочнение 200 МПа), радиальные пластинки
1.8 × 0.6 мкм (как в «рое» колец), окружная нагрузка 0 / 110 / 200 МПа.

Эталон — упругопластический расчёт на Фурье (hill_fft): две пластинки выпадают под нагрузкой, затем третья.
На месте следующей (той же ориентации) — выгода g = σ:ε*/ε_n, средняя по месту, за вычетом прямой добавки нагрузки
(вклад соседей). С эталоном сравниваются:
  ореолы сложены  — упругое решение с ε* трёх пластинок + ε_p одиночной пластинки (та же нагрузка), сдвинутая в каждую;
  упругое         — без ореолов;
  автомат 0.4     — то, что делает ca_kinetic: сетка 0.4 мкм, plate_eigen + таблица ореолов Э635, ca_hydride.Elastic.
Расположения (шаг между пластинками: t — вдоль пластинки, n — по нормали):
  бок о бок (0, 1.6) — как 68 % зародышей роя; колода (0.9, 1.2) — эшелон; цепочка (2.6, 0) — у кончика.
Ячейка 20 × 20 мкм, шаг 0.1 мкм. Оси hill_fft: x — окружное (θ), y — радиальное (r); нормаль пластинки по x.

python halo_yield_check.py папка [процессов]       — расчёты (по одному файлу на случай) и сводка
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import hill_fft as hf  # noqa: E402

SY, HARD = 226.0, 200.0
L, H, A, TH = 20.0, 0.1, 0.9, 0.6
NN = int(round(L / H))
LOADS = (0.0, 110.0, 200.0)
CONF = {"side": (0.0, 1.6), "deck": (0.9, 1.2), "chain": (2.6, 0.0)}
NAME = {"side": "бок о бок", "deck": "колода", "chain": "цепочка"}
E_RAD = hf.misfit(0)                    # нормаль по x (окружное) — радиальная пластинка


def coords():
    x = (np.arange(NN) + 0.5) * H - L / 2
    return np.meshgrid(x, x, indexing="ij")          # X — окружное (нормаль), Y — радиальное (вдоль пластинки)


def mask_at(ct, cn):
    X, Y = coords()
    return (np.abs(X - cn) < TH / 2) & (np.abs(Y - ct) < A)


def centres(conf):
    st, sn = CONF[conf]
    return [(k * st, k * sn) for k in (-2, -1, 0)], (st, sn)       # три пластинки и место следующей


def job(a):
    out, kind, conf, s = a
    fn = os.path.join(out, f"{kind}_{conf}_{s:g}.npz")
    if os.path.exists(fn):
        return
    t0 = time.time()
    hf.WORKERS = 1
    c = hf.Cell((NN, NN, 1), H, sy=SY, hard=HARD)
    Sig = [s, 0, 0, 0, 0, 0]
    if kind == "single":
        pl = hf.Plate(c, mask_at(0.0, 0.0)[:, :, None], E_RAD)
        pl.run(Sig, nt=10)
    else:
        cs, _ = centres(conf)
        old = (mask_at(*cs[0]) | mask_at(*cs[1]))[:, :, None]
        new = mask_at(*cs[2])[:, :, None]
        p1 = hf.Plate(c, old, E_RAD)
        p1.run(Sig, nt=10)
        pl = hf.Plate(c, new, E_RAD)
        pl.u, pl.E, pl.ep, pl.p = p1.u, p1.E, p1.ep, p1.p
        pl.matrix = ~(old.reshape(-1) | new.reshape(-1))
        pl.eig0 = p1.eig
        pl.run(Sig, nt=10)
    np.savez_compressed(fn, sig=pl.sig.reshape(6, NN, NN).astype(np.float32),
                        ep=pl.ep.reshape(6, NN, NN).astype(np.float32), p=pl.p.reshape(NN, NN).astype(np.float32))
    print(kind, conf, s, "время", round(time.time() - t0), flush=True)


def g_of(sig):
    """Выгода радиальной пластинки (Мандель, ε* без сдвига)."""
    return (sig[0] * E_RAD[0] + sig[1] * E_RAD[1] + sig[2] * E_RAD[2]) / hf.EPS_N


def elastic(eig, s):
    c = hf.Cell((NN, NN, 1), H, sy=1e9)
    pl = hf.Plate(c, np.zeros((NN, NN, 1), bool), np.zeros(6))
    pl.eig0 = eig.reshape(6, -1)
    sig, _, _, _ = pl.solve(np.array([s, 0, 0, 0, 0, 0], float), 0.0)
    return sig.reshape(6, NN, NN)


def automaton(conf, s):
    """Как в ca_kinetic: сетка 0.4 мкм, собственная деформация plate_eigen + ореолы из таблицы Э635 (на нагрузке s),
    упругое решение ca_hydride.Elastic (free_z); вклад соседей в g радиальной пластинки на месте следующей."""
    from ca_hydride import Elastic, plate_eigen
    from halo import HaloTable
    dx = 0.4; n = int(round(L / dx))
    tab = HaloTable(path=os.path.join(os.path.dirname(HERE), "data_halo", "halo_tab_e635.npz")); tab.set_load(s)
    cs, step = centres(conf)
    E = [np.zeros((n, n)) for _ in range(4)]; hyd = np.zeros((n, n))
    out = {}
    for with_halo in (False, True):
        E = [np.zeros((n, n)) for _ in range(4)]; hyd = np.zeros((n, n))
        for ct, cn in cs:
            # автомат: строки — y (ND, радиальное, вниз), столбцы — x (TD, окружное); ψ = 90° — пластинка вдоль ND
            cc = np.array([L / 2 + ct, L / 2 + cn])
            ys, xs, fr, comps = plate_eigen((n, n), dx, cc, np.pi / 2, A, TH)
            sel = np.ix_(ys, xs)
            hyd[sel] = np.maximum(hyd[sel], fr)
            for a_, v in zip(E, comps):
                a_[sel] += fr * v
            if with_halo:
                ys, xs, hc = tab.patch((n, n), dx, cc, np.pi / 2, A)
                for a_, v in zip(E, hc):
                    a_[np.ix_(ys, xs)] += v
        el = Elastic((n, n), dx, 90e3, 0.34, free_z=True)
        S11, S22, S12, S33 = el.stress(*E)
        g = (hf.EPS_N * S11 + hf.EPS_T * S22 + hf.EPS_T * S33) / hf.EPS_N      # радиальная: нормаль по x
        yy = (np.arange(n) + 0.5) * dx - L / 2
        Yr, Xc = np.meshgrid(yy, yy, indexing="ij")
        site = (np.abs(Xc - step[1]) < TH / 2 + 0.2) & (np.abs(Yr - step[0]) < A) & (hyd < 0.2)
        out[with_halo] = float(g[site].mean())
    return out


def summary(out):
    X, Y = coords()
    rows = []
    for conf in CONF:
        cs, step = centres(conf)
        hyd = mask_at(*cs[0]) | mask_at(*cs[1]) | mask_at(*cs[2])
        site = mask_at(*step) & ~hyd
        for s in LOADS:
            fr = os.path.join(out, f"stack_{conf}_{s:g}.npz"); f1 = os.path.join(out, f"single_{conf}_{s:g}.npz")
            f1 = os.path.join(out, f"single_side_{s:g}.npz")
            if not (os.path.exists(fr) and os.path.exists(f1)):
                continue
            g_app = s * E_RAD[0] / hf.EPS_N
            ref = np.load(fr); one = np.load(f1)
            g_ref = float(g_of(ref["sig"].astype(float))[site].mean()) - g_app
            eps_star = np.zeros((6, NN, NN)); halo = np.zeros((6, NN, NN))
            for ct, cn in cs:
                eps_star += E_RAD[:, None, None] * mask_at(ct, cn)
                halo += np.roll(one["ep"].astype(float), (int(round(cn / H)), int(round(ct / H))), (1, 2))
            halo *= ~hyd
            g_halo = float(g_of(elastic(eps_star + halo, s))[site].mean()) - g_app
            g_el = float(g_of(elastic(eps_star, s))[site].mean()) - g_app
            au = automaton(conf, s)
            r = dict(conf=conf, load=s, ref=g_ref, halo=g_halo, el=g_el, aut_halo=au[True], aut_el=au[False],
                     p_area_ref=float((ref["p"] > 1e-3).sum() * H * H), p_area_single=float((one["p"] > 1e-3).sum() * H * H))
            rows.append(r)
            print(f"{NAME[conf]:10s} σ {s:5.0f}: эталон {g_ref:7.0f}  ореолы сложены {g_halo:7.0f}  упругое {g_el:7.0f}  "
                  f"| автомат 0.4: с ореолами {au[True]:7.0f}, без {au[False]:7.0f}  | пласт. зона: стопка "
                  f"{r['p_area_ref']:6.1f}, одиночная {r['p_area_single']:5.1f} мкм²", flush=True)
    json.dump(rows, open(os.path.join(out, "halo_yield_check.json"), "w"), indent=1)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    jobs = [(out, "single", "side", s) for s in LOADS] + [(out, "stack", cf, s) for cf in CONF for s in LOADS]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        list(pool.imap_unordered(job, jobs))
    summary(out)
