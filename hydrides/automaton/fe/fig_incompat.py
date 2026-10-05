"""Рис. fig_fe5_incompat: напряжения несовместности между зёрнами и выгода собственной пластинки.
python fig_incompat.py папка_с_incompat_*.npz"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

D = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#9cc3ef", "#efeeea", "#f4b49a", ORANGE])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
c = (slice(24, -24), slice(24, -24))
z = np.load(os.path.join(D, "incompat_chi30_s1.npz"))
gr = z["grains"]; dx = float(z["dx"])
fig = plt.figure(figsize=(15.5, 4.9), facecolor=BG)
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.5)
ax = fig.add_subplot(gs[0])
g = z["thermal_g"][c]
ext = (0, g.shape[1] * dx, 0, g.shape[0] * dx)
im = ax.imshow(g, cmap=DIV, vmin=-50, vmax=50, extent=ext, interpolation="nearest")
b = np.zeros(gr[c].shape, bool); gg = gr[c]
b[:-1] |= gg[:-1] != gg[1:]; b[:, :-1] |= gg[:, :-1] != gg[:, 1:]
ax.imshow(np.ma.masked_where(~b, b), cmap=LinearSegmentedColormap.from_list("k", [INK, INK]), extent=ext, interpolation="nearest", alpha=0.5)
# след пластинки в зерне
psi = z["gpsi"]
for gid in np.unique(gg)[::2]:
    yy, xx = np.nonzero(gg == gid)
    if len(yy) < 60:
        continue
    cy, cx = yy.mean() * dx, xx.mean() * dx
    L = 1.2
    ax.plot([cx - L * np.cos(psi[gid]), cx + L * np.cos(psi[gid])],
            [ext[3] - cy - L * np.sin(psi[gid]), ext[3] - cy + L * np.sin(psi[gid])], color=INK, lw=1.2)
ax.set_xlabel("TD, мкм"); ax.set_ylabel("ND (r), мкм")
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("g своей пластинки, МПа"); cb.outline.set_edgecolor(MUTED)
ax.set_title("а) остывание на 100 К", loc="left", color=INK)

ax = fig.add_subplot(gs[1])
bins = np.arange(0, 91, 10)
for tag, col, lab in (("chi22_s1", BLUE, "χ₀ = 22°"), ("chi30_s1", INK, "χ₀ = 30°"), ("chi38.5_s1", ORANGE, "χ₀ = 38.5°")):
    zz = np.load(os.path.join(D, f"incompat_{tag}.npz"))
    ap = np.abs(np.degrees(zz["gpsi"])[zz["grains"]])[c].ravel(); gt = zz["thermal_g"][c].ravel()
    k = np.digitize(ap, bins) - 1
    m = np.array([gt[k == i].mean() if np.sum(k == i) > 200 else np.nan for i in range(len(bins) - 1)])
    sd = np.array([gt[k == i].std() if np.sum(k == i) > 200 else np.nan for i in range(len(bins) - 1)])
    xc = bins[:-1] + 5
    ax.plot(xc, m, "o-", color=col, lw=2, ms=5, label=lab)
    if tag == "chi30_s1":
        ax.fill_between(xc, m - sd, m + sd, color=col, alpha=0.08, lw=0)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("наклон пластинки в зерне |ψ|, ° от TD"); ax.set_ylabel("g своей пластинки, МПа (остывание 100 К)")
ax.set_xlim(0, 90); ax.set_ylim(-40, 40); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.text(88, -36, "полоса — разброс между зёрнами (χ₀ = 30°)", fontsize=8.5, color=MUTED, ha="right")
ax.set_title("б) крутым зёрнам остывание выгодно", loc="left", color=INK)

ax = fig.add_subplot(gs[2])
ap = np.abs(np.degrees(z["gpsi"])[gr])[c].ravel(); gm = z["mech_g"][c].ravel(); gf = z["gapp_formula"][c].ravel()
k = np.digitize(ap, bins) - 1
xc = bins[:-1] + 5
mm = np.array([gm[k == i].mean() if np.sum(k == i) > 200 else np.nan for i in range(len(bins) - 1)])
sd = np.array([gm[k == i].std() if np.sum(k == i) > 200 else np.nan for i in range(len(bins) - 1)])
mf = np.array([gf[k == i].mean() if np.sum(k == i) > 200 else np.nan for i in range(len(bins) - 1)])
off = np.nanmean(mm - mf)
ax.fill_between(xc, mm - off - sd, mm - off + sd, color=BLUE, alpha=0.12, lw=0)
ax.plot(xc, mm - off, "o-", color=BLUE, lw=2, ms=5, label="МКЭ (анизотропные зёрна)")
ax.plot(xc, mf, "--", color=ORANGE, lw=2, label="формула автомата σ(ε₁₁ − ε̄)/ε_n")
ax.set_xlabel("наклон пластинки в зерне |ψ|, ° от TD"); ax.set_ylabel("g своей пластинки, МПа (σ_TD = 100 МПа)")
ax.set_xlim(0, 90); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("в) нагрузка: формула верна, ±8 МПа разброс", loc="left", color=INK)
fig.suptitle("Напряжения несовместности между зёрнами циркония (МКЭ, сечение r–θ, ось c в плоскости сечения)",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig_fe5_incompat.png"), dpi=115, facecolor=BG, bbox_inches="tight")
print("ok")
