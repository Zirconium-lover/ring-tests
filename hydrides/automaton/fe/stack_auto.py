"""Поле соседей в автомате против МКЭ для стопок. В автомате (ca_hydride.Elastic) выгода следующей пластинки
g = σ:ε*/ε_n считается упругой микроупругостью на Фурье (плоская деформация, шаг dx, доля клетки по 3 × 3
подточкам) и обрезается потолком ±σ_cap. Здесь то же поле для трёх выпавших пластинок колоды или цепочки
(без нагрузки) сравнивается с упругим и упругопластическим МКЭ (stack_fe.py, случаи *_0_el и *_0) — в матрице
и в местах, где сядет следующая пластинка: продолжение колоды (t, n) = (2.5, 1.2) и цепочки (5.8, 0.6) мкм,
средним по площади этой будущей пластинки.
python stack_auto.py папка_мкэ"""
import os
import sys
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ca_hydride import Elastic  # noqa: E402
from stack_fe import plates_of, STEP, A_HALF, TH, L  # noqa: E402

EPS_N, EPS_T = 0.0720, 0.0458
CAP = 180.0
BLUE, ORANGE, GREEN, VIOLET, INK, MUTED, BG, GRID = ("#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc7", "#0b0b0b", "#52514e",
                                                     "#fcfcfb", "#e4e2dc")


def grid(dx):
    n = int(round(L / dx))
    x = (np.arange(n) + 0.5) * dx - L / 2
    return n, np.meshgrid(x, x, indexing="xy")          # массивы [n, t]: ось 0 — n (y), ось 1 — t (x)


def frac_rect(dx, ct, cn, sub=3):
    n, (T, N) = grid(dx)
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * dx
    fr = np.zeros((n, n))
    for a in off:
        for b in off:
            fr += (np.abs(T + a - ct) < A_HALF) & (np.abs(N + b - cn) < TH / 2)
    return fr / sub ** 2


def auto_g(conf, dx):
    n, (T, N) = grid(dx)
    fr = sum(frac_rect(dx, ct, cn) for ct, cn in plates_of(conf))
    el = Elastic((n, n), dx, 90e3, 0.34)
    S11, S22, S12, S33 = el.stress(EPS_T * fr, EPS_N * fr, 0.0 * fr, EPS_T * fr)
    g = (EPS_T * S11 + EPS_N * S22 + EPS_T * S33) / EPS_N
    return T, N, fr, g


def fe_g(z, T, N):
    m = z["own"] < 0
    S = z["S"][m].astype(np.float64)
    g = (EPS_T * S[:, 0] + EPS_N * S[:, 1] + EPS_T * S[:, 2]) / EPS_N
    return LinearNDInterpolator(z["xy"][m].astype(np.float64), g)(T, N)


def site_mean(T, N, a, conf):
    st, sn = STEP[conf]
    sel = (np.abs(T - st) < A_HALF) & (np.abs(N - sn) < TH / 2)
    return float(np.nanmean(a[sel]))


if __name__ == "__main__":
    D = sys.argv[1]
    out = {}
    fig, axs = plt.subplots(2, 4, figsize=(21, 8.4), facecolor=BG)
    for row, conf in zip(axs, ("deck", "chain")):
        fe_p = np.load(os.path.join(D, f"{conf}_circ_0.npz"))
        fe_e = np.load(os.path.join(D, f"{conf}_circ_0_el.npz")) if os.path.exists(os.path.join(D, f"{conf}_circ_0_el.npz")) else None
        res = {}
        for dx in (0.1, 0.4):
            T, N, fr, g = auto_g(conf, dx)
            gp = fe_g(fe_p, T, N)
            ge = fe_g(fe_e, T, N) if fe_e is not None else np.full_like(g, np.nan)
            dist = ndi.distance_transform_edt(fr < 0.01) * dx
            near = (dist > 0.15) & (dist < 1.0) & np.isfinite(gp)
            res[dx] = dict(site_auto=site_mean(T, N, g, conf), site_auto_cap=site_mean(T, N, np.clip(g, -CAP, CAP), conf),
                           site_fe_el=site_mean(T, N, ge, conf), site_fe_pl=site_mean(T, N, gp, conf),
                           rms_auto_vs_fe_el=float(np.sqrt(np.nanmean((g - ge)[near] ** 2))),
                           rms_auto_vs_fe_pl=float(np.sqrt(np.nanmean((g - gp)[near] ** 2))),
                           rms_cap_vs_fe_pl=float(np.sqrt(np.nanmean((np.clip(g, -CAP, CAP) - gp)[near] ** 2))),
                           max_auto=float(np.nanmax(np.abs(g[near]))), max_fe_pl=float(np.nanmax(np.abs(gp[near]))))
            if dx == 0.1:
                T1, N1, g1, gp1, ge1, fr1 = T, N, g, gp, ge, fr
        out[conf] = res
        win = (T1[0] > -9) & (T1[0] < 6)
        wn = (N1[:, 0] > -3.5) & (N1[:, 0] < 2.5)
        ext = [T1[0][win][0], T1[0][win][-1], N1[:, 0][wn][0], N1[:, 0][wn][-1]]
        sub = lambda a: a[np.ix_(wn, win)]
        hyd = fr1 > 0.5
        for ax, a, ttl in ((row[0], np.clip(g1, -CAP, CAP), "автомат: упругий Фурье, потолок ±180"),
                           (row[1], ge1, "МКЭ, упругая матрица"), (row[2], gp1, "МКЭ, Мизес")):
            im = ax.imshow(sub(np.where(hyd, np.nan, a)), origin="lower", extent=ext, cmap="RdBu_r", vmin=-400, vmax=400)
            ax.contour(sub(hyd.astype(float)), [0.5], colors=INK, linewidths=0.8, extent=ext)
            st, sn = STEP[conf]
            ax.add_patch(plt.Rectangle((st - A_HALF, sn - TH / 2), 2 * A_HALF, TH, fill=False, ls="--", ec=GREEN, lw=1.2))
            ax.set_title(f"{conf}: {ttl}", loc="left", fontsize=10, color=INK)
            ax.set_facecolor(BG)
        plt.colorbar(im, ax=row[2], fraction=0.03, label="g следующей, МПа")
        # профиль через место следующей пластинки (по n при t = st)
        st, sn = STEP[conf]
        j = int(np.argmin(np.abs(T1[0] - st)))
        sel = (N1[:, j] > -3.5) & (N1[:, j] < 3.0)
        ax = row[3]
        ax.plot(N1[sel, j], np.clip(g1[sel, j], -CAP, CAP), color=BLUE, lw=1.6, label="автомат (потолок)")
        ax.plot(N1[sel, j], g1[sel, j], color=BLUE, lw=0.9, ls=":", label="автомат без потолка")
        ax.plot(N1[sel, j], ge1[sel, j], color=MUTED, lw=1.4, ls="--", label="МКЭ упругий")
        ax.plot(N1[sel, j], gp1[sel, j], color=ORANGE, lw=1.8, label="МКЭ Мизес")
        ax.axvspan(sn - TH / 2, sn + TH / 2, color=GREEN, alpha=0.12, label="место следующей")
        ax.set_ylim(-700, 500); ax.grid(color=GRID); ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel(f"n, мкм (t = {st:g})"); ax.set_ylabel("g, МПа"); ax.set_facecolor(BG)
        ax.set_title("профиль через место следующей пластинки", loc="left", fontsize=10, color=INK)
    fig.suptitle("Поле соседей в автомате против МКЭ: три выпавшие пластинки, без нагрузки", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.savefig(os.path.join(HERE, "figs", "fig_fe8_stack_auto.png"), dpi=95, facecolor=BG, bbox_inches="tight")
    json.dump(out, open(os.path.join(D, "auto_compare.json"), "w"), indent=1)
    for conf, res in out.items():
        for dx, r in res.items():
            print(conf, dx, {k: round(v, 1) for k, v in r.items()})
