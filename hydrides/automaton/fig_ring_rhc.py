"""Рисунок: кольца Э635 (Плясов и др. 2023) — модель против опыта по трём мерам на каждом участке:
F_l по третям стенки (табл. 1), граница зон (табл. 2), RHC — RHCP и RHCF через всю толщину (рис. 3, окна 120 мкм,
≈3.5 мкм/пикс; модель «снята» так же: утолщение травлением 3 мкм, тот же пиксель и та же обработка).

python fig_ring_rhc.py папка_расчётов [--label "коллективная пластичность, фора 4 °C"] [--out файл]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhc_model as RM  # noqa: E402
from connectivity import rhc_mask  # noqa: E402

INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e6e6e3", "#2a78d6"
T_MM = 0.85
EXP_FL = {"S1_0": (0.39, 0.10, 0.01), "S1_90": (0.10, 0.69, 0.89), "S2_0": (0.35, 0.11, 0.04), "S2_90p": (0.15, 0.47, 0.71)}
EXP_BOUND = {"S1_0": 0.57, "S1_90": 0.50, "S2_0": 0.51, "S2_90p": 0.70}        # мм от внутренней поверхности
FIG3 = {"S1_0": "S1_after0", "S1_90": "S1_after90", "S2_0": "S2_after0", "S2_90p": "S2_after90"}
TITLE = {"S1_0": "200 Н, 0°", "S1_90": "200 Н, 90°", "S2_0": "350 Н, 0°", "S2_90p": "350 Н, 90°"}


def model_rhc(z, um_px=3.5, etch_um=3.0, win_um=120.0):
    import hydride_spec as hs
    hyd = np.asarray(z["hyd"], float); dx = float(z["dx"])
    g = RM.as_micrograph(hyd, dx, etch_um, um_px)
    s1 = max(1.5, 2 * um_px)
    _, D = hs.analyse("модель", um=um_px, field=True, g=g, scales_um=(s1, 2 * s1))
    nw = int(round(win_um / um_px))
    out = {}
    for key, m in (("any", D["hm"]), ("rad", D["hm_rad"])):
        v = [rhc_mask(m[:, c0:c0 + nw], um_px, periodic=False) for c0 in range(0, m.shape[1] - nw + 1, nw)]
        out[key] = {k: float(np.mean([x[k] for x in v])) for k in ("RHCP", "RHCF")}
    return out


LEVEL = 0.25           # граница зон — где F_l переходит уровень посередине между фоном до опыта (~0.1) и зоной (~0.4)


def boundary_mm(prof, site):
    """Граница зон по профилю F_l (12 слоёв от наружной поверхности): где F_l пересекает LEVEL; мм от внутренней."""
    p = np.array(prof, float); n = len(p)
    x_in = (n - 0.5 - np.arange(n)) / n * T_MM          # слой → расстояние от внутренней
    ok = np.isfinite(p)
    x, y = x_in[ok][::-1], p[ok][::-1]                  # по возрастанию расстояния от внутренней
    s = np.sign(y - LEVEL)
    idx = np.where(np.diff(s) != 0)[0]
    if not len(idx):
        return np.nan
    i = idx[0] if site.endswith("90") or site.endswith("90p") else idx[-1]
    return float(x[i] + (LEVEL - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--label", default="модель")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "fig28_ring_rhc.png"))
    a = ap.parse_args()
    ex = {r["image"]: r for r in json.load(open(os.path.join(HERE, "data_lit", "plyasov2023_fig3_rhc.json")))["rows"]
          if r["win_um"] == 120}
    rows = {}
    for f in glob.glob(os.path.join(a.folder, "*.json")):
        m = json.load(open(f))
        if "ring" in m and os.path.exists(f[:-5] + ".npz"):
            rows[m["ring"]] = (m, np.load(f[:-5] + ".npz"))
    sites = [s for s in ("S1_0", "S1_90", "S2_0", "S2_90p") if s in rows]
    fig = plt.figure(figsize=(4.2 * len(sites), 11))
    gs = fig.add_gridspec(3, len(sites), height_ratios=[2.2, 1.2, 1.2], hspace=0.45, wspace=0.35)
    summary = {}
    for j, site in enumerate(sites):
        m, z = rows[site]
        # снимок модели (наружная поверхность сверху)
        ax = fig.add_subplot(gs[0, j])
        h = np.asarray(z["hyd"], float); dx = float(z["dx"]); k = 3
        h = h[: h.shape[0] // k * k, : h.shape[1] // k * k].reshape(h.shape[0] // k, k, h.shape[1] // k, k).max((1, 3))
        ax.imshow(1 - 0.85 * np.clip(h * 1.5, 0, 1), cmap="gray", vmin=0, vmax=1,
                  extent=(0, h.shape[1] * k * dx, h.shape[0] * k * dx, 0), interpolation="antialiased")
        bm = boundary_mm(m["F_l_5_prof"], site)
        for xb, col, ls in ((EXP_BOUND[site], INK, "-"), (bm, BLUE, "--")):
            if np.isfinite(xb):
                ax.axhline((T_MM - xb) * 1000, color=col, ls=ls, lw=1.2)
        ax.set_title(f"{TITLE[site]}\nграница зон: опыт {EXP_BOUND[site]:.2f} мм, модель "
                     + (f"{bm:.2f} мм" if np.isfinite(bm) else "—"), loc="left", fontsize=9)
        ax.set_ylabel("от наружной поверхности, мкм", fontsize=8, color=MUTED); ax.tick_params(labelsize=7)
        # F_l по третям
        ax = fig.add_subplot(gs[1, j]); x = np.arange(3); w = 0.38
        mod = [m[f"F_l_5_{s}"] for s in ("out", "mid", "in")]
        ax.bar(x - w / 2, EXP_FL[site], w, color=INK, label="опыт (табл. 1)")
        ax.bar(x + w / 2, mod, w, color=BLUE, label=a.label)
        ax.set_xticks(x); ax.set_xticklabels(["нар.", "сред.", "внутр."]); ax.set_ylim(0, 1.1)
        ax.set_title("F_l по третям стенки", loc="left", fontsize=9); ax.grid(axis="y", color=GRID, lw=0.6)
        if j == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper center")
        # RHC
        ax = fig.add_subplot(gs[2, j])
        r = model_rhc(z)
        e = ex[FIG3[site]]
        labels = ["RHCP", "RHCP\nрадиальные", "RHCF"]
        ev = [e["RHCP_any"], e["RHCP_radial_only"], e["RHCF_any"]]
        mv = [r["any"]["RHCP"], r["rad"]["RHCP"], r["any"]["RHCF"]]
        x = np.arange(3)
        ax.bar(x - w / 2, [v[0] for v in ev], w, yerr=[v[1] for v in ev], color=INK, capsize=3, label="опыт (рис. 3)")
        ax.bar(x + w / 2, mv, w, color=BLUE, label=a.label)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0, 1.0)
        ax.set_title("RHC через всю стенку (окна 120 мкм)", loc="left", fontsize=9); ax.grid(axis="y", color=GRID, lw=0.6)
        for axx in fig.axes:
            for sp in ("top", "right"):
                axx.spines[sp].set_visible(False)
        summary[site] = dict(F_l=mod, boundary_mm=bm, RHC=r)
        print(site, "F_l", [round(v, 2) for v in mod], "опыт", EXP_FL[site], "| граница", round(bm, 2) if np.isfinite(bm) else None,
              "опыт", EXP_BOUND[site], "| RHCP", round(mv[0], 2), "/", round(mv[1], 2), "RHCF", round(mv[2], 2),
              "опыт", round(ev[0][0], 2), "/", round(ev[1][0], 2), round(ev[2][0], 2))
    fig.suptitle(f"Кольца Э635 (Плясов и др. 2023): {a.label} против опыта — F_l, граница зон, RHC",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(a.out, dpi=110, facecolor="white", bbox_inches="tight")
    json.dump(summary, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print("→", a.out)


if __name__ == "__main__":
    main()
