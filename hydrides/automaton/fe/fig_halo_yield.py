"""Рисунок: карта выгоды зарождения у стопки «бок о бок» при 200 МПа (Э635, σ_y 226) — эталоны (упругопластика на
Фурье, сетка 0.1 мкм; CalculiX) против коллективной пластичности и суммы таблиц ореолов на сетке автомата 0.4 мкм.
Цвет — вклад соседей в выгоду радиальной пластинки g − σθ (МПа); рамка — место следующей пластинки.

python fig_halo_yield.py папка_эталона_фурье папка_calculix
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import halo_yield_check as R  # noqa: E402
import proto_collective as P  # noqa: E402
from ca_hydride import Elastic, plate_eigen  # noqa: E402
from halo import HaloTable  # noqa: E402

S_APP = 200.0
EN, ET = R.hf.EPS_N, R.hf.EPS_T
WIN = (-6.0, 5.0, -4.0, 4.0)            # θ (по нормали к пластинкам) от, до; r (вдоль) от, до — мкм от центра новой


def automaton_maps():
    dx = 0.4; n = int(round(R.L / dx))
    cs, step = R.centres("side"); to_ = lambda c: np.array([R.L / 2 + c[0], R.L / 2 + c[1]])
    out = {}
    col = P.Collective((n, n), dx, R.SY, R.HARD, Sig=(S_APP, 0, 0, 0)); col.solve()
    for grp in (cs[:2], [cs[2]]):
        col.add_plates([(to_(c), np.pi / 2, R.A, R.TH) for c in grp]); col.solve()
    out["коллективная пластичность\n(сетка автомата 0.4 мкм)"] = (P.g_rad(col.stress(col.Ep)) - S_APP, col.hyd, dx)
    tab = HaloTable(path=os.path.join(os.path.dirname(HERE), "data_halo", "halo_tab_e635.npz")); tab.set_load(S_APP)
    E = [np.zeros((n, n)) for _ in range(4)]; hyd = np.zeros((n, n))
    for c in cs:
        ys, xs, fr, comps = plate_eigen((n, n), dx, to_(c), np.pi / 2, R.A, R.TH); sel = np.ix_(ys, xs)
        hyd[sel] = np.maximum(hyd[sel], fr)
        for a, v in zip(E, comps):
            a[sel] += fr * v
        ys, xs, hc = tab.patch((n, n), dx, to_(c), np.pi / 2, R.A)
        for a, v in zip(E, hc):
            a[np.ix_(ys, xs)] += v
    S = Elastic((n, n), dx, 90e3, 0.34, free_z=True).stress(*E)
    out["сумма таблиц ореолов\n(как было в автомате)"] = ((EN * S[0] + ET * S[1] + ET * S[3]) / EN, hyd, dx)
    return out


def main():
    DF, DC = sys.argv[1], sys.argv[2]
    maps = {}
    z = np.load(os.path.join(DF, "stack_side_200.npz")); sig = z["sig"].astype(float)
    cs, step = R.centres("side")
    hyd = (R.mask_at(*cs[0]) | R.mask_at(*cs[1]) | R.mask_at(*cs[2]))
    g = R.g_of(sig) - S_APP
    maps["эталон: упругопластика\nна Фурье, сетка 0.1 мкм"] = (g.T, hyd.T.astype(float), R.H)   # строки — r, столбцы — θ
    c = np.load(os.path.join(DC, "side_rad_U200.npz"))
    xy, Sc, own = c["xy"].astype(float), c["S"].astype(float), c["own"]
    gc = (ET * Sc[:, 0] + EN * Sc[:, 1] + ET * Sc[:, 2]) / EN - S_APP     # оси МКЭ: X вдоль (r), Y по нормали (θ)
    x = (np.arange(R.NN) + 0.5) * R.H - R.L / 2
    T_, N_ = np.meshgrid(x, x, indexing="ij")                             # строки — t (r), столбцы — n (θ)
    gcc = griddata(xy, gc, (T_, N_), method="linear")
    hc = griddata(xy, (own >= 0).astype(float), (T_, N_), method="nearest")
    maps["эталон: CalculiX\n(МКЭ, независимый код)"] = (gcc, hc, R.H)
    maps.update(automaton_maps())
    fig, axs = plt.subplots(1, 4, figsize=(17, 5.2))
    norm = TwoSlopeNorm(0.0, -500.0, 500.0)
    for ax, (name, (g, h, dx)) in zip(axs, maps.items()):
        n = g.shape[0]
        ext = (-R.L / 2, R.L / 2, R.L / 2, -R.L / 2)
        if dx == 0.4:   # массивы автомата: центр новой пластинки в (L/2, L/2) — та же система
            ext = (-R.L / 2, R.L / 2, R.L / 2, -R.L / 2)
        im = ax.imshow(np.where(h > 0.5, np.nan, g), cmap="RdBu_r", norm=norm, extent=ext, interpolation="nearest")
        ax.contour(np.linspace(-R.L / 2, R.L / 2, n), np.linspace(-R.L / 2, R.L / 2, n), h, [0.5], colors="k", linewidths=0.8)
        ax.add_patch(plt.Rectangle((step[1] - R.TH / 2, step[0] - R.A), R.TH, 2 * R.A, fill=False, ec="#1baf7a", lw=1.5, ls="--"))
        ax.set_xlim(WIN[0], WIN[1]); ax.set_ylim(WIN[3], WIN[2])
        site = np.isfinite(g)
        rng = (np.abs(np.linspace(-R.L / 2, R.L / 2, n))[None, :] < 99)
        ax.set_title(name + f"\nмакс. выгода у стопки {np.nanmax(np.where(h > 0.2, np.nan, g)[(np.abs(np.linspace(-R.L/2, R.L/2, n))[:, None] < 4) & (np.linspace(-R.L/2, R.L/2, n)[None, :] > -6) & (np.linspace(-R.L/2, R.L/2, n)[None, :] < 5)]):.0f} МПа",
                     loc="left", fontsize=9)
        ax.set_xlabel("окружное направление θ, мкм", fontsize=8)
    axs[0].set_ylabel("радиус r, мкм", fontsize=8)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("вклад соседей в выгоду радиальной пластинки, МПа", fontsize=8)
    fig.suptitle("Стопка из трёх радиальных пластинок «бок о бок», Э635, окружная нагрузка 200 МПа "
                 "(зелёная рамка — место следующей пластинки)", x=0.01, ha="left", fontsize=11)
    out = os.path.join(HERE, "figs", "fig_fe11_halo_yield.png")
    fig.savefig(out, dpi=120, facecolor="white", bbox_inches="tight")
    print("→", out)


if __name__ == "__main__":
    main()
