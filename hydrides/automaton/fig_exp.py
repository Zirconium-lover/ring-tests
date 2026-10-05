"""Рис. 5 — текстура и порог переориентации; рис. 6 — стенка Э635 с изгибом; числа абляции."""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figs import BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID, BG, LIGHT, tidy, draw_plates
from ca_analysis import simon_w

E = "/tmp/claude-0/-home-user/16339a70-3da3-5a74-97cf-f17fbd8cb16f/scratchpad/ca_runs/v2/"
rows = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(E + "*.json")}
MAGENTA = "#e87ba4"
CH = [(22, "≈ труба Zr-1Nb (χ₀≈24)", MAGENTA), (30, "лист Zry-4 (калибровка)", INK), (33, "Э635, внутренний слой", AQUA),
      (38.5, "Э635, средний и наружный", ORANGE), (41.5, "Э635, другой образец", BLUE)]
S = [0, 100, 150, 200, 250]
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.9), facecolor=BG, gridspec_kw=dict(wspace=0.28))
thr = []
for chi, lab, col in CH:
    m, lo, hi = [], [], []
    for s in S:
        v = np.array([rows[f"tex_chi{chi}_s{s}_seed{k}"]["RHF_image"] for k in (1, 2, 3) if f"tex_chi{chi}_s{s}_seed{k}" in rows], float)
        m.append(v.mean()); lo.append(v.min()); hi.append(v.max())
    m = np.array(m)
    axs[0].fill_between(S, lo, hi, color=col, alpha=0.12, lw=0)
    axs[0].plot(S, m, "o-", color=col, lw=2, ms=5, label=f"χ₀ = {chi:g}°: {lab}")
    lvl = 0.35
    k = np.argmax(m >= lvl)
    t = np.interp(lvl, m[k - 1:k + 1], S[k - 1:k + 1]) if k > 0 and m[k] >= lvl else np.nan
    thr.append((chi, lab, col, t, m[0]))
axs[0].axhline(0.35, color=GRID, lw=1, ls=(0, (3, 3)))
axs[0].set_xlabel("окружное напряжение при охлаждении, МПа"); axs[0].set_ylabel("RHF"); axs[0].set_ylim(0, 1)
axs[0].set_title("а) RHF при разной текстуре (≈180 wppm H)", loc="left", fontsize=10.5, color=INK)
axs[0].legend(frameon=False, fontsize=8.5, loc="upper left"); tidy(axs[0])
ax = axs[1]
for chi, lab, col, t, r0 in thr:
    ax.plot([chi], [t], "o", color=col, ms=9)
    ax.text(chi + 0.6, t + 4, f"{t:.0f} МПа", fontsize=9, color=INK)
ax.plot([c for c, *_ in thr], [t for *_, t, _ in thr], color=MUTED, lw=1, zorder=0)
ax.set_xlabel("наклон базисных полюсов от радиуса χ₀, ° (по параметрам Кернса)")
ax.set_ylabel("напряжение, при котором RHF = 0.35, МПа"); ax.set_xlim(18, 46); ax.set_ylim(0, 160)
ax.axvline(45, color=GRID, lw=1, ls=(0, (4, 3))); ax.text(44.5, 150, "45°", ha="right", fontsize=9, color=MUTED)
ax.set_title("б) Прогноз: порог переориентации от текстуры", loc="left", fontsize=10.5, color=INK); tidy(ax)
fig.savefig("figs/fig5_texture.png", dpi=115, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("пороги:", [(c, round(t), round(r0, 2)) for c, l, col, t, r0 in thr])


def depth_profile(name, nb=8, H=800.0):
    z = np.load(E + name + ".npz")["plates"]
    y, psi, half = z[:, 0], z[:, 2], z[:, 3]
    dev = np.degrees(np.abs(np.arctan(np.tan(psi))))
    L = 2 * half
    edges = np.linspace(0, H, nb + 1)
    out, amt = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        out.append((L[m] * simon_w(dev[m])).sum() / L[m].sum() if m.any() else np.nan)
        amt.append(L[m].sum())
    return 0.5 * (edges[1:] + edges[:-1]), np.array(out), np.array(amt) / np.sum(amt)


E = "/tmp/claude-0/-home-user/16339a70-3da3-5a74-97cf-f17fbd8cb16f/scratchpad/ca_runs/v2/"
rows = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(E + "*.json")}
fig = plt.figure(figsize=(14, 6.4), facecolor=BG)
gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 0.42, 0.42, 0.42], wspace=0.25)
ax = fig.add_subplot(gs[0])
for nm, col, lab, ls in (("wall_free", LIGHT, "без нагрузки", (0, (1.5, 2))), ("wall_unif200", BLUE, "растяжение +200 МПа", (0, (5, 2))),
                         ("wall_bend100", AQUA, "изгиб ±100 МПа", (0, (6, 2, 1, 2))),
                         ("wall_bend200", ORANGE, "изгиб: +200 снаружи → −200 внутри", "-")):
    P = [depth_profile(f"{nm}_seed{k}") for k in (1, 2)]
    yb = P[0][0]; v = np.nanmean([p[1] for p in P], axis=0)
    ax.plot(v, yb, "o", color=col, ls=ls, lw=2, ms=5, label=lab)
    print(nm, "RHF по слоям:", np.round(v, 2), " доля гидрида по слоям:", np.round(np.mean([p[2] for p in P], axis=0), 3))
ax.set_ylim(800, 0); ax.set_xlim(0, 1)
ax.set_xlabel("RHF в слое"); ax.set_ylabel("глубина от наружной поверхности, мкм")
ax.set_title("а) Стенка Э635 0.8 мм, текстура по слоям", loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower right"); tidy(ax)
for i, (nm, t) in enumerate((("wall_free_seed1", "б) без нагрузки"), ("wall_unif200_seed1", "в) +200 МПа"), ("wall_bend200_seed1", "г) изгиб ±200"))):
    ax = fig.add_subplot(gs[i + 1])
    draw_plates(ax, np.load(E + nm + ".npz")["plates"], (800, 240), lw=0.7)
    ax.set_title(t, loc="left", fontsize=10, color=INK); ax.set_xticks([0, 120, 240]); ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_xlabel("TD, мкм")
    if i: ax.set_yticklabels([])
fig.text(0.01, -0.02, "Текстура по слоям из параметров Кернса канала Э635 (наружный / средний / внутренний: χ₀ = 38.3 / 38.7 / 32.7°). "
         "Напряжение при охлаждении меняется по толщине линейно.", fontsize=9, color=MUTED)
fig.savefig("figs/fig6_wall.png", dpi=115, bbox_inches="tight", facecolor=BG)
for nm in ("dimple_free", "dimple_bend"):
    P = [depth_profile(f"{nm}_seed{k}", nb=3, H=250.0) for k in (1, 2)]
    print(nm, "RHF по третям (нар., сред., внутр.):", np.round(np.nanmean([p[1] for p in P], axis=0), 2))
for s_ in (100, 150, 200):
    v = np.array([rows[f"bist_s{s_}_seed{k}"]["packet_RHF"] for k in range(1, 9)], float)
    w = np.array([rows[f"bist_s{s_}_seed{k}"]["RHF_image"] for k in range(1, 9)], float)
    print(f"разброс при {s_} МПа: RHF пакетов", np.round(np.sort(v), 2), " RHF снимка", np.round(np.sort(w), 2))
