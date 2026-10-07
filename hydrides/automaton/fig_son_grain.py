"""Рис. 20: мелкое зерно в условиях опыта Son и др. 2026 — RHF по снимку, доля радиальных пластинок, доля
межзёренных и снимок. python fig_son_grain.py папка [папка ...] (первая — снимок)"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, VIOLET, INK, MUTED, BG, GRID = ("#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc7",
                                                     "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
R = defaultdict(lambda: defaultdict(list))
for d in sys.argv[1:]:                      # вариант — из первой папки, где он есть
    got = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        if m["variant"] in ("viz5", "ref02", "fine_sim", "fine_h06", "fine_h06x"):
            got[m["variant"]][int(m["sigma_app"])].append(m)
    for v, by in got.items():
        if v not in R:
            R[v] = by
VAR = [("viz5", MUTED, "зерно 2.5 × 4.5 мкм, сетка 0.4, поле 240 мкм"),
       ("ref02", BLUE, "зерно 2.5 × 4.5 мкм, сетка 0.2, поле 120 мкм"),
       ("fine_sim", GREEN, "зерно 0.66 × 1.2 мкм, всё в масштабе 0.27 (старые остатки)"),
       ("fine_h06", ORANGE, "зерно 0.66 × 1.2 мкм, пластинка 0.6 мкм"),
       ("fine_h06x", VIOLET, "то же + рост и переход через границы (старые остатки)")]
SON = {"CWSR": {0: 0.0, 83: 0.038, 96: 0.107, 146: 0.454}, "PRXA": {0: 0.0, 83: 0.115, 96: 0.263, 146: 0.540}}


def curve(ax, v, key, **kw):
    s = sorted(R[v])
    y = [np.nanmean([m[key] for m in R[v][q]]) for q in s]
    e = [np.nanstd([m[key] for m in R[v][q]]) for q in s]
    ax.errorbar(s, y, yerr=e, marker="o", ms=4, lw=1.8, capsize=2, **kw)


fig = plt.figure(figsize=(20, 5.4), facecolor=BG)
axs = [fig.add_axes([0.03 + 0.205 * i, 0.12, 0.17, 0.74]) for i in range(3)]
ax_img = fig.add_axes([0.655, 0.06, 0.33, 0.84])
for v, c, lab in VAR:
    if v in R:
        if v != "fine_sim":             # пластинки 0.16 мкм на оптическом снимке не разрешаются
            curve(axs[0], v, "RHF_image", color=c, label=lab)
        curve(axs[1], v, "RHF45_plates", color=c, label=lab)
        curve(axs[2], v, "GB_frac", color=c, label=lab)
for name, mk in (("CWSR", "s"), ("PRXA", "^")):
    q = sorted(SON[name])
    for ax in axs[:2]:
        ax.plot(q, [SON[name][k] for k in q], color=INK, marker=mk, ms=8, mfc="none", lw=1.0, ls=":",
                label=f"Son 2026, {name}")
axs[2].axhspan(0.90, 0.94, color=VIOLET, alpha=0.15, lw=0); axs[2].text(3, 0.95, "Son, CWSR: 92 %", fontsize=8.5, color=VIOLET)
for ax, t in zip(axs, ("RHF по снимку", "доля длины радиальных пластинок", "доля длины межзёренных")):
    ax.set_title(t, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("окружное напряжение, МПа"); ax.set_ylim(0, 1.02); ax.set_xlim(-5, 185)
    ax.grid(color=GRID); ax.set_facecolor(BG)
axs[0].legend(frameon=False, fontsize=7.8, loc="upper left")
# снимки: мелкое зерно, пластинка 0.6 мкм — 0 и 146 МПа рядом, по 30 × 30 мкм
W = 30.0
for k, (sig, x0) in enumerate(((0, 0.0), (146, W + 3.0))):
    f = os.path.join(sys.argv[1], f"fine_h06_s{sig}_seed1.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    p = KParams(seed=1, grain_um=(0.66, 1.2), dx=0.2, size_um=(120.0, 120.0))
    grains, gpsi, gbd = make_grains(p, np.random.default_rng(1), gb=True)
    ny, nx = grains.shape; n = int(W / p.dx)
    gbm = np.zeros(ny * nx, bool); gbm[gbd["cells"]] = True; gbm = gbm.reshape(ny, nx)
    # окно 30 × 30 мкм там, где больше всего пластинок
    P = z["plates"]
    y0 = float(np.clip(np.median(P[:, 0]) - W / 2, 0, 120 - W))
    xh = np.histogram(P[:, 1], bins=np.arange(0, 121 - W, 5.0) + 0.0)[0] if len(P) else [0]
    cnt = [((P[:, 1] >= a) & (P[:, 1] < a + W) & (P[:, 0] >= y0) & (P[:, 0] < y0 + W)).sum() for a in np.arange(0, 91, 5.0)]
    xa = float(np.arange(0, 91, 5.0)[int(np.argmax(cnt))])
    iy, ix = int(y0 / p.dx), int(xa / p.dx)
    ax_img.imshow(np.where(gbm[iy:iy + n, ix:ix + n], 0.85, 1.0), cmap="gray", vmin=0, vmax=1,
                  extent=(x0, x0 + W, W, 0), interpolation="nearest")
    for (cy, cx, psi, half), kd in zip(P, z["kind"]):
        cy, cx = cy - y0, cx - xa
        if not (-2 < cy < W + 2 and -2 < cx < W + 2):
            continue
        ty, tx = -np.sin(psi), np.cos(psi)
        xs = np.clip([cx - half * tx, cx + half * tx], 0, W) + x0
        ax_img.plot(xs, [cy - half * ty, cy + half * ty], lw=2.0, color=ORANGE if kd == 1 else BLUE,
                    solid_capstyle="butt")
    ax_img.text(x0, -0.8, f"{sig} МПа (окно y {y0:.0f}–{y0 + W:.0f}, x {xa:.0f}–{xa + W:.0f} мкм)", fontsize=9.5, color=INK)
ax_img.set_xlim(0, 2 * W + 3); ax_img.set_ylim(W, -2); ax_img.set_xticks([]); ax_img.set_yticks([])
ax_img.set_title("зерно 0.66 × 1.2 мкм, пластинка 0.6 мкм: межзёренные — оранжевые, в теле зерна — синие; 30 мкм",
                 loc="left", fontsize=9, color=INK)
fig.suptitle("Мелкое зерно в условиях Son 2026 (Zr-Nb, 187 ppm, 0.3 °C/мин; фора зёрен 5 °C, фора границы 1.5 °C)",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig20_son_grain.png"), dpi=100, facecolor=BG, bbox_inches="tight")
for v, _, lab in VAR:
    if v in R:
        s = sorted(R[v])
        print(lab)
        for k in ("RHF_image", "Fn45_image", "RHF45_plates", "GB_frac", "GB_frac_rad", "L_mean", "n_new"):
            if k in R[v][s[0]][0]:
                print(f"  {k:13s}" + " ".join(f"{q}:{np.nanmean([m[k] for m in R[v][q]]):.2f}" for q in s))
