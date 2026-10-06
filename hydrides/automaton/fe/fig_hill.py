"""Рис. fe6: выгода радиальной пластинки над окружной Δg(σθ) — упругость, пластичность Мизеса и Хилла (2D и 3D),
и пластическая деформация у окружной и радиальной пластинки в матрице Хилла при 155 МПа.
python fig_hill.py папка_2D папка_3D"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D2, D3 = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#2f9e6e", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def table(d):
    r = {}
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        r[(m["hill"], m["load"], m["orient"])] = m
    out = {}
    for (h, lk, o), m in r.items():
        if o != "circ":
            continue
        q = r.get((h, lk, "rad"), m if (h == "iso" and lk == "0") else None)
        if q is not None:
            out[(h, lk)] = (m["s_theta"], m["s_z"], q["g"] - m["g"])
    return out


t2, t3 = table(D2), table(D3)
fig = plt.figure(figsize=(15, 5.4), facecolor=BG)
ax = fig.add_axes([0.05, 0.12, 0.42, 0.76])
x = np.linspace(0, 260, 50)
ax.plot(x, (0.0720 - 0.0458) / 0.0720 * x, color=MUTED, lw=1.5, ls="--", label="упругость: 0.36·σθ")
for (h, c, m, lab) in (("iso", BLUE, "o", "Мизес"), ("massih", ORANGE, "s", "Хилл (Massih)")):
    for tab, mfc, dim in ((t2, c, "2D"), (t3, "none", "3D")):
        pts = sorted(v for (hh, lk), v in tab.items() if hh == h and v[1] == 0)
        if pts:
            ax.plot([p[0] for p in pts], [p[2] for p in pts], marker=m, color=c, mfc=mfc, ms=7, lw=1.6 if dim == "2D" else 0,
                    ls="-" if dim == "2D" else "none", label=f"{lab}, {dim}")
        bi = [v for (hh, lk), v in tab.items() if hh == h and v[1] > 0]
        for v in bi:
            ax.plot([v[0]], [v[2]], marker="*", color=c, mfc=mfc, ms=12, ls="none")
ax.plot([], [], "k*", mfc="none", ms=10, ls="none", label="двухосно: 110/63 и 75/62 МПа (σθ/σz)")
ax.axhline(0, color=GRID, lw=1)
ax.set_xlabel("окружное напряжение σθ, МПа"); ax.set_ylabel("Δg = g_рад − g_окр, МПа")
ax.set_title("выгода радиальной пластинки над окружной (работа выделения / ε_n)", loc="left", fontsize=10.5, color=INK)
ax.grid(color=GRID); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.text(150, -15, "для порога Cinbiz (155 МПа) нужен сдвиг ≈ −150 МПа\nв пользу окружной; Хилл даёт −2 (2D) и +79 (3D)",
        fontsize=8.5, color=INK, va="top")
ax.set_ylim(-60, 260)
# поля пластической деформации у пластинки в матрице Хилла при 155 МПа (2D)
for k, (o, title) in enumerate((("circ", "окружная пластинка"), ("rad", "радиальная пластинка"))):
    f = os.path.join(D2, f"massih1_{o}_U155.npz")
    if not os.path.exists(f):
        continue
    p = np.load(f)["p"]
    n = p.shape[0]; h = 0.1; w = int(6.0 / h)
    sub = p[n // 2 - w:n // 2 + w, n // 2 - w:n // 2 + w].T          # ось 0 — x (θ), ось 1 — y (r) → картинка: x вправо, r вверх
    a2 = fig.add_axes([0.53 + 0.235 * k, 0.12, 0.21, 0.76])
    im = a2.imshow(sub * 100, origin="lower", extent=(-6, 6, -6, 6), cmap="magma", vmin=0, vmax=3)
    a2.set_title(f"{title}: p, %", loc="left", fontsize=10.5, color=INK)
    a2.set_xlabel("θ, мкм"); a2.set_ylabel("r, мкм" if k == 0 else "")
cb = fig.add_axes([0.985, 0.12, 0.008, 0.76]); fig.colorbar(im, cax=cb)
fig.suptitle("Анизотропия пластичности по Хиллу: пластичность усиливает выбор нагрузкой (Δg ≈ 0.9–1.0·σθ вместо 0.36·σθ), "
             "но не даёт ни порога, ни двухосности", x=0.01, ha="left", fontsize=12, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig_fe6_hill.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
