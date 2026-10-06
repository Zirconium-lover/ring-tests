"""Рис. 18: межзёренный канал в кинетическом движке — RHF, доля межзёренных (против Son и др. 2026),
связность и снимок структуры. python fig_gb.py папка_без_границ папка_с_границами [σ снимка] [фора снимка]"""
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
D0, D1 = sys.argv[1], sys.argv[2]
S_SNAP = int(sys.argv[3]) if len(sys.argv) > 3 else 200
G_SNAP = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
runs = defaultdict(lambda: defaultdict(list))
for d, gb in ((D0, False), (D1, True)):
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        if m.get("bias_dT") != 12.4 or m.get("sigma_cap") != 180.0 or m.get("dT_s", 0.0) != 0.0:
            continue
        key = ("gb", m["gb_dT"]) if m.get("gb") else ("intra", None)
        if gb != (key[0] == "gb"):
            continue
        n = int(round(m.get("size_um", [240.0])[0] / m.get("dx", 0.4))) if isinstance(m.get("size_um"), list) else 600
        P = np.load(f[:-5] + ".npz")["plates"]
        c = metrics_of(P, (n, n), 0.4)
        m.update({k: v for k, v in c.items() if k != "extents"})
        runs[key][int(m["sigma_app"])].append(m)
VAR = [(("intra", None), MUTED, "только в теле зерна"),
       (("gb", 0.0), GREEN, "+ границы, фора 0"),
       (("gb", 1.0), BLUE, "+ границы, фора 1 °C"),
       (("gb", 3.0), ORANGE, "+ границы, фора 3 °C")]


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
        curve(axs[3], k, "L_cluster_max", color=c, label=lab)
        if k[0] == "gb":
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
axs[3].set_title("самый длинный связный кластер, мкм", loc="left", fontsize=10.5, color=INK)
for ax in axs:
    ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_xlim(-5, 260); ax.grid(color=GRID)
    ax.set_facecolor(BG)
axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
axs[1].legend(frameon=False, fontsize=8.5, loc="lower right")
# снимок: границы зёрен и пластинки, кусок 80 × 80 мкм
f = [g for g in glob.glob(D1 + "/*.json") if json.load(open(g)).get("gb_dT") == G_SNAP
     and int(json.load(open(g))["sigma_app"]) == S_SNAP and json.load(open(g))["seed"] == 1]
if f:
    m = json.load(open(f[0])); z = np.load(f[0][:-5] + ".npz")
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
    ax_img.set_title(f"{S_SNAP} МПа, фора границ {G_SNAP:g} °C: межзёренные — оранжевые,\nв теле зерна — синие; "
                     f"по горизонтали TD, 80 мкм", loc="left", fontsize=9, color=INK)
fig.suptitle("Межзёренный канал зарождения: пластинка вдоль грани, след грани в пределах 15° от базисного следа "
             "одного из соседей (Qin 2011, Son 2026); фора зёрен 12.4 °C, сдвиг 0.08 °C/МПа", x=0.01, ha="left",
             fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig18_gb.png"), dpi=100, facecolor=BG, bbox_inches="tight")
for k, _, lab in VAR:
    if k not in runs:
        continue
    s = sorted(runs[k])
    print(lab)
    for fld in ("RHF_image", "Fn45_image", "GB_frac", "GB_frac_rad", "GB_frac_circ", "L_cluster_max", "RHCF", "HCC_rad_mean", "RHCP"):
        if fld in runs[k][s[0]][0]:
            print(f"  {fld:14s}" + " ".join(f"{q}:{np.nanmean([m[fld] for m in runs[k][q]]):.2f}" for q in s))
