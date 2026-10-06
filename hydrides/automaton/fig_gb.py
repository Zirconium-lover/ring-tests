"""Рис. 18: межзёренный канал в кинетическом движке — RHF, доля межзёренных (против Son и др. 2026),
связность и снимок структуры. python fig_gb.py папка [σ снимка]"""
import os
import sys
import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_kinetic import KParams  # noqa: E402
from ca_hydride import make_grains  # noqa: E402
from connectivity import metrics_of  # noqa: E402
from calib_beta_report import LEP, cinbiz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, VIOLET, INK, MUTED, BG, GRID = ("#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc7",
                                                     "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
D = sys.argv[1]
S_SNAP = int(sys.argv[2]) if len(sys.argv) > 2 else 175
runs = defaultdict(lambda: defaultdict(list))
for f in glob.glob(D + "/*.json"):
    m = json.load(open(f))
    P = np.load(f[:-5] + ".npz")["plates"]
    for g in (1.0, 3.0):
        c = metrics_of(P, (600, 600), 0.4, gap_um=g)
        m[f"L{g:.0f}"] = c["L_cluster_max"]
    runs[(bool(m.get("gb", False)), bool(m.get("grow_kin", False)))][int(m["sigma_app"])].append(m)
VAR = [((False, False), MUTED, "в теле зерна, фора зёрен 12.4 °C"),
       ((True, False), BLUE, "+ границы: фора 1.5 °C, фора зёрен 10.9 °C"),
       ((True, True), ORANGE, "+ рост во времени и переход через границы")]


def curve(ax, key, field, **kw):
    s = sorted(runs[key])
    y = [np.nanmean([m[field] for m in runs[key][q]]) for q in s]
    e = [np.nanstd([m[field] for m in runs[key][q]]) for q in s]
    ax.errorbar(s, y, yerr=e, marker="o", ms=4, lw=1.8, capsize=2, **kw)


fig = plt.figure(figsize=(20, 5.4), facecolor=BG)
axs = [fig.add_axes([0.03 + 0.205 * i, 0.12, 0.17, 0.74]) for i in range(4)]
ax_img = fig.add_axes([0.83, 0.08, 0.17, 0.82])
for k, c, lab in VAR:
    if k in runs:
        curve(axs[0], k, "Fn45_image", color=c, label=lab)
        curve(axs[3], k, "L3", color=c, label=lab + ", зазор 3 мкм")
        curve(axs[3], k, "L1", color=c, ls="--", alpha=0.6)
        if k[0]:
            curve(axs[1], k, "GB_frac", color=c, label=lab)
            curve(axs[2], k, "GB_frac_rad", color=c, label=lab)
x = np.linspace(0, 260, 300)
axs[0].plot(x, cinbiz(x), color=INK, lw=1.2, ls="--", label="Cinbiz 2016, одноосно")
axs[0].set_title("доля длины на снимке под 45–135°", loc="left", fontsize=10.5, color=INK)
axs[0].set_ylim(0, 1.02)
for ax in axs[1:3]:
    ax.axhspan(0.90, 0.94, color=VIOLET, alpha=0.15, lw=0)
    ax.axhspan(0.62, 0.68, color=MUTED, alpha=0.12, lw=0)
    ax.text(5, 0.95, "Son 2026, CWSR (0.9 мкм): 92 %", fontsize=8, color=VIOLET)
    ax.text(5, 0.585, "PRXA (1.3 мкм): 68 % / 62 % у радиальных", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.05)
axs[1].set_title("доля длины межзёренных — все пластинки", loc="left", fontsize=10.5, color=INK)
axs[2].set_title("доля межзёренных среди радиальных", loc="left", fontsize=10.5, color=INK)
axs[3].set_title("длиннейший кластер, мкм (пунктир: зазор 1 мкм)", loc="left", fontsize=10.5, color=INK)
for ax in axs:
    ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_xlim(-5, 260); ax.grid(color=GRID)
    ax.set_facecolor(BG)
axs[0].legend(frameon=False, fontsize=8.5, loc="center left")
axs[1].legend(frameon=False, fontsize=8.5, loc="lower right")
# снимок: границы зёрен и пластинки, кусок 80 × 80 мкм
f = sorted(glob.glob(D + f"/bias_dT10.9_gbTrue_gb_dT1.5_seed1_sigma_app{S_SNAP}.json"))
if f:
    z = np.load(f[0][:-5] + ".npz")
    p = KParams(seed=1)
    grains, gpsi, gbd = make_grains(p, np.random.default_rng(1), gb=True)
    ny, nx = grains.shape
    W = int(80 / p.dx)
    gbm = np.zeros(ny * nx, bool); gbm[gbd["cells"]] = True; gbm = gbm.reshape(ny, nx)
    ax_img.imshow(np.where(gbm[:W, :W], 0.82, 1.0), cmap="gray", vmin=0, vmax=1, extent=(0, 80, 80, 0),
                  interpolation="nearest")
    for (cy, cx, psi, half), kd in zip(z["plates"], z["kind"]):
        if not (-5 < cy < 85 and -5 < cx < 85):
            continue
        ty, tx = -np.sin(psi), np.cos(psi)
        ax_img.plot([cx - half * tx, cx + half * tx], [cy - half * ty, cy + half * ty], lw=2.2,
                    color=ORANGE if kd == 1 else BLUE, solid_capstyle="butt")
    ax_img.set_xlim(0, 80); ax_img.set_ylim(80, 0); ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"{S_SNAP} МПа, с границами: межзёренные — оранжевые,\nв теле зерна — синие; "
                     f"по горизонтали TD, 80 мкм", loc="left", fontsize=9, color=INK)
fig.suptitle("Межзёренный канал зарождения: пластинка вдоль грани, след грани в пределах 15° от базисного следа "
             "одного из соседей (Qin 2011, Son 2026); сдвиг 0.08 °C/МПа (Vizcaíno 2014)", x=0.01, ha="left",
             fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig18_gb.png"), dpi=100, facecolor=BG, bbox_inches="tight")
for k, _, lab in VAR:
    if k not in runs:
        continue
    s = sorted(runs[k])
    print(lab)
    for fld in ("RHF_image", "Fn45_image", "GB_frac", "GB_frac_rad", "GB_frac_circ", "L1", "L3"):
        if fld in runs[k][s[0]][0]:
            print(f"  {fld:14s}" + " ".join(f"{q}:{np.nanmean([m[fld] for m in runs[k][q]]):.2f}" for q in s))
