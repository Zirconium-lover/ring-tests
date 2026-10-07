"""Складываются ли пластические ореолы? Поле трёх пластинок (колода, цепочка) собирается из одиночной:
собственная деформация ε* трёх пластинок + ε_p одиночной пластинки (halo_runs.py), сдвинутая в каждую, —
упругое решение той же схемой Фурье (hill_fft, без пластичности) при той же нагрузке. Сравнение с честным
упругопластическим расчётом трёх пластинок (stack_runs.py --fut-el --fields) и с тем, что делает автомат сейчас
(упругое поле соседей, обрезанное потолком ±σ_cap). Мера — g следующей пластинки той же ориентации
(σ:ε*/ε_n) в матрице и средним по местам следующей пластинки колоды (2.5, 1.2) и цепочки (5.8, 0.6) мкм.
Из g вычитается прямая добавка нагрузки Σ:ε*/ε_n — остаётся вклад соседей.
python halo_check.py папка_ореолов папка_стопок"""
import os
import sys
import json
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hill_fft as hf  # noqa: E402
from stack_compare import fft_tn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPS_N, EPS_T = hf.EPS_N, hf.EPS_T
STEP = {"deck": (2.5, 1.2), "chain": (5.8, 0.6)}
CAP = 180.0
E_STAR = np.array([EPS_T, EPS_N, EPS_T, 0, 0, 0])
INK, MUTED, BG, GRID, BLUE, ORANGE, GREEN = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc", "#2a78d6", "#eb6834", "#2f9e6e"


def g_of(s):
    """s — (6, n, n) в Манделе, оси (t, n): выгода следующей пластинки той же ориентации."""
    return (s[0] * EPS_T + s[1] * EPS_N + s[2] * EPS_T) / EPS_N


def elastic(eig, Sig, n, h):
    """Упругое решение схемой hill_fft: собственная деформация eig (6, n, n), среднее напряжение Sig."""
    c = hf.Cell((n, n, 1), h, sy=1e9)
    pl = hf.Plate(c, np.zeros((n, n, 1), bool), np.zeros(6))
    pl.eig0 = eig.reshape(6, -1)
    sig, _, _, _ = pl.solve(np.asarray(Sig, float), 0.0)
    return sig.reshape(6, n, n)


def automaton(eig, Sig, k):
    """Упругий Фурье автомата (ca_hydride.Elastic, спектральный, free_z) на сетке в k раз грубее: собственная
    деформация осредняется блоками k × k. Возвращает g (n/k × n/k), гидрид (доля > 0.5) и координаты."""
    sys.path.insert(0, os.path.dirname(HERE))
    from ca_hydride import Elastic
    n = eig.shape[1]; m = n // k; dx = 30.0 / m
    blk = lambda a: a.reshape(m, k, m, k).mean((1, 3))
    e = [blk(eig[i]) for i in range(6)]
    el = Elastic((m, m), dx, 90e3, 0.34, free_z=True)
    # массивы автомата: ось 0 — y (= n), ось 1 — x (= t); сдвиг тензорный (Мандель / √2)
    S11, S22, S12, S33 = el.stress(e[0].T, e[1].T, e[5].T / np.sqrt(2), e[2].T)
    tt, nn, zz = S11.T + Sig[0], S22.T + Sig[1], S33.T + Sig[2]
    g = (tt * EPS_T + nn * EPS_N + zz * EPS_T) / EPS_N
    x = (np.arange(m) + 0.5) * dx - 15.0
    T, N = np.meshgrid(x, x, indexing="ij")
    return g, T, N


def site_mean(a, T, N, conf, hyd):
    st, sn = STEP[conf]
    m = (np.abs(T - st) < 2.5) & (np.abs(N - sn) < 0.3) & ~hyd
    return float(np.mean(a[m]))


if __name__ == "__main__":
    DH, DS = sys.argv[1], sys.argv[2]
    hf.WORKERS = 4
    res = {}
    figrows = []
    for conf in ("deck", "chain"):
        for orient, al, lk in (("circ", 90, "0"), ("rad", 0, "U110"), ("circ", 90, "U110")):
            hz = os.path.join(DH, f"a2.5_al{al}_{lk}.npz" if lk != "0" else "a2.5_al0_0.npz")
            sz = os.path.join(DS, f"{conf}_{orient}_{lk}_h0.1_fe.npz")
            if not (os.path.exists(hz) and os.path.exists(sz)):
                continue
            H1 = np.load(hz); meta = json.load(open(hz[:-4] + ".json"))
            Sig = np.array(meta["Sig"])
            n = H1["p"].shape[0]; h = 30.0 / n
            x = (np.arange(n) + 0.5) * h - 15.0
            T, N = np.meshgrid(x, x, indexing="ij")
            st, sn = STEP[conf]
            shifts = [(0, 0), (-int(round(st / h)), -int(round(sn / h))), (-int(round(2 * st / h)), -int(round(2 * sn / h)))]
            m1 = H1["mask"].astype(bool)
            hyd = np.zeros((n, n), bool); eps_star = np.zeros((6, n, n)); halo = np.zeros((6, n, n))
            for dt, dn in shifts:
                mk = np.roll(m1, (dt, dn), (0, 1))
                hyd |= mk
                eps_star += E_STAR[:, None, None] * mk
                halo += np.roll(H1["ep"].astype(np.float64), (dt, dn), (1, 2))
            halo *= ~hyd                    # в эталоне (--fut-el) в гидриде пластической деформации нет
            # честный расчёт трёх пластинок (поля в осях пластинки)
            f = fft_tn(np.load(sz), orient)
            g_ref = (f["tt"] * EPS_T + f["nn"] * EPS_N + f["zz"] * EPS_T) / EPS_N
            g_app = float(Sig @ E_STAR) / EPS_N                      # прямая добавка нагрузки (Мандель: сдвиг 0)
            s_el = elastic(eps_star, Sig, n, h)
            s_halo = elastic(eps_star + halo, Sig, n, h)
            g_el = g_of(s_el); g_halo = g_of(s_halo)
            g_cap = np.clip(g_el - g_app, -CAP, CAP) + g_app            # как в автомате: потолок на поле соседей
            dist = ndi.distance_transform_edt(~hyd) * h
            r = {}
            k = 4                                                        # сетка автомата 0.4 мкм
            hyd_c = hyd.reshape(n // k, k, n // k, k).mean((1, 3)) > 0.5
            ga_h, Tc, Nc = automaton(eps_star + halo, Sig, k)
            ga_e, _, _ = automaton(eps_star, Sig, k)
            ga_c = np.clip(ga_e - g_app, -CAP, CAP) + g_app
            for name, g in (("автомат 0.4: ореолы", ga_h), ("автомат 0.4: упр. + потолок", ga_c)):
                r[name] = dict(site_deck=site_mean(g, Tc, Nc, "deck", hyd_c) - g_app,
                               site_chain=site_mean(g, Tc, Nc, "chain", hyd_c) - g_app, rms_near=np.nan, rms_mid=np.nan)
            for name, g in (("пластика, 3 пластинки", g_ref), ("ореолы сложены", g_halo), ("упругое", g_el),
                            ("упругое + потолок", g_cap)):
                d = g - g_ref
                zn = (dist > 0.15) & (dist < 1.0); zf = (dist >= 1.0) & (dist < 3.0)
                r[name] = dict(site_deck=site_mean(g, T, N, "deck", hyd) - g_app,
                               site_chain=site_mean(g, T, N, "chain", hyd) - g_app,
                               rms_near=float(np.sqrt(np.mean(d[zn] ** 2))), rms_mid=float(np.sqrt(np.mean(d[zf] ** 2))))
            key = f"{conf}_{orient}_{lk}"
            res[key] = r
            print(key, f"(нагрузка даёт {g_app:.0f})")
            for name, q in r.items():
                print(f"   {name:24s} место колоды {q['site_deck']:7.0f}  цепочки {q['site_chain']:7.0f}   "
                      f"Δrms 0.15–1 мкм {q['rms_near']:6.0f}, 1–3 мкм {q['rms_mid']:5.0f}")
            if lk == "U110" and orient == "rad":
                figrows.append((key, T, N, hyd, g_ref - g_app, g_halo - g_app, g_cap - g_app))
    json.dump(res, open(os.path.join(DH, "halo_check.json"), "w"), indent=1)
    if figrows:
        from matplotlib.colors import TwoSlopeNorm
        fig, axs = plt.subplots(len(figrows), 3, figsize=(17, 3.6 * len(figrows)), facecolor=BG, squeeze=False)
        for row, (key, T, N, hyd, a, b, c) in zip(axs, figrows):
            win = (T[:, 0] > -12) & (T[:, 0] < 8); wn = (N[0] > -3) & (N[0] < 2.5)
            ext = [T[win, 0][0], T[win, 0][-1], N[0, wn][0], N[0, wn][-1]]
            for ax, arr, ttl in ((row[0], a, "упругопластический расчёт трёх пластинок"),
                                 (row[1], b, "сумма ореолов одиночной + упругий Фурье"),
                                 (row[2], c, "автомат сейчас: упругое поле, потолок ±180")):
                im = ax.imshow(np.where(hyd, np.nan, arr)[np.ix_(win, wn)].T, origin="lower", extent=ext, cmap="RdBu_r",
                               norm=TwoSlopeNorm(0.0, -1200.0, 600.0))
                for conf in STEP:
                    st, sn = STEP[conf]
                    ax.add_patch(plt.Rectangle((st - 2.5, sn - 0.3), 5, 0.6, fill=False, ls="--", ec=GREEN, lw=1.0))
                ax.set_title(f"{key}: {ttl}", loc="left", fontsize=9.5, color=INK); ax.set_facecolor(BG)
            plt.colorbar(im, ax=row[2], fraction=0.03, label="g соседей, МПа")
        fig.suptitle("Пластические ореолы: складываются ли поля пластинок (g следующей пластинки, без вклада нагрузки)",
                     x=0.01, ha="left", fontsize=12, color=INK)
        fig.savefig(os.path.join(HERE, "figs", "fig_fe9_halo_check.png"), dpi=95, facecolor=BG, bbox_inches="tight")
