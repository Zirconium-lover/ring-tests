"""Рис. fig_fe4_gaps: разрывы вдоль пакетов — снимки Cinbiz против автомата с искусственными разрывами.
python fig_gaps.py папка_прогонов_v2 папка_снимков(lepine) [папка_вывода_для_версии_со_снимком]
Версия со снимком Cinbiz (кадр из диссертации) в репозиторий не кладётся."""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from gaps import measure_dark, summary, render_plates

RUNS, LEP = sys.argv[1], sys.argv[2]
PRIV = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, LEP)
from spec import load  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
IMGS = [("0 циклов", "p34-000.jpg", (370, 1585, 12, 860), 0.202), ("1 цикл", "p35-000.jpg", (372, 1585, 12, 865), 0.202),
        ("4 цикла", "p37-000.jpg", (345, 1490, 12, 800), 0.2174)]
res = {"img": {}, "ca": {}}
maps = {}
for name, f, crop, um in IMGS:
    g = load(f, crop, inpaint_red=False)
    o = measure_dark(g, um)
    res["img"][name] = summary(o)
    if f == "p35-000.jpg":
        maps["img"] = (g, o["_maps"])
for s, cls in ((0, "окр"), (250, "рад")):
    P = np.load(f"{RUNS}/tex_chi30_s{s}_seed1.npz")["plates"]
    for sh in (0.0, 0.5, 1.0, 1.5):
        g = render_plates(P, (240.0, 240.0), shrink_um=sh)
        o = measure_dark(g, 0.2)
        res["ca"][f"{s}_{2 * sh:g}"] = summary(o)
        if s == 250 and sh in (0.0, 1.0):
            maps[f"ca{2 * sh:g}"] = (g, o["_maps"])
json.dump(res, open(os.path.join(HERE, "figs", "gaps_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)


def overlay(ax, g, mp, box, title):
    skP, near, _ = mp
    rgb = np.stack([g, g, g], -1) / 255.0
    rgb[ndi.binary_dilation(skP & near, iterations=1)] = (0.11, 0.69, 0.48)
    rgb[ndi.binary_dilation(skP & ~near, iterations=1)] = (0.92, 0.41, 0.20)
    y0, y1, x0, x1 = box
    ax.imshow(rgb[y0:y1, x0:x1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(MUTED)
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def chart(axs):
    for ax, (cls, s, title) in zip(axs, (("окр", 0, "окружные пакеты"), ("рад", 250, "радиальные пакеты"))):
        xs = [res["ca"][f"{s}_{d:g}"][cls]["cov"] for d in (0, 1, 2, 3)]
        ys = [res["ca"][f"{s}_{d:g}"][cls]["big_per100"] for d in (0, 1, 2, 3)]
        ax.plot(xs, ys, color=MUTED, lw=1, zorder=1)
        for d, x, y, c in zip((0, 1, 2, 3), xs, ys, RAMP):
            ax.plot(x, y, "o", color=c, ms=9, zorder=2)
            ax.text(x + 0.015, y + 0.6, f"автомат +{d} мкм" if d else "автомат", fontsize=8.5, color=INK)
        for name, v in res["img"].items():
            vv = v[cls]
            if np.isfinite(vv["cov"]) and vv["L"] > 60:
                ax.plot(vv["cov"], vv["big_per100"], "s", color=ORANGE, ms=9, zorder=3)
                ax.text(vv["cov"] - 0.02, vv["big_per100"] + (0.9 if name == "4 цикла" else -0.9), f"Cinbiz, {name}",
                        fontsize=8.5, color=INK, ha="right", va="center")
        ax.set_xlim(0, 1); ax.set_ylim(0, 30)
        ax.set_xlabel("доля оси пакета, покрытая гидридом")
        ax.set_ylabel("разрывов ≥ 1.5 мкм на 100 мкм пакета")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)


def make(with_img, path):
    fig = plt.figure(figsize=(13, 9.2), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.28, wspace=0.18)
    top = [fig.add_subplot(gs[0, k]) for k in range(3)]
    box = (100, 520, 300, 860)
    if with_img:
        g, mp = maps["img"]
        overlay(top[0], g, mp, (80, 500, 380, 940), "снимок Cinbiz, 1 цикл при 225 МПа")
    else:
        top[0].axis("off")
        top[0].text(0.0, 0.5, "снимок Cinbiz — в версии\nне для репозитория", fontsize=10, color=MUTED, va="center")
    overlay(top[1], *maps["ca0"], box, "автомат, 250 МПа")
    overlay(top[2], *maps["ca2"], box, "автомат, пластинки укорочены:\n+2 мкм к каждому разрыву")
    bot = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.28, wspace=0.22)
    chart([fig.add_subplot(bot[1, 0]), fig.add_subplot(bot[1, 1])])
    fig.suptitle("Разрывы между пластинками вдоль пакетов: ореол из МКЭ дал бы разрывы 2.5–5 мкм у каждой пластинки",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "Ось пакета: зелёная — рядом гидрид (затемнение выше 97-го процентиля фона), оранжевая — разрыв. "
             "Синие точки — автомат с пластинками, укороченными на 0, 0.5, 1, 1.5 мкм с каждого конца.", fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=110, facecolor=BG, bbox_inches="tight")


make(False, os.path.join(HERE, "figs", "fig_fe4_gaps.png"))
if PRIV:
    make(True, os.path.join(PRIV, "fig_fe4_gaps_with_image.png"))
print(json.dumps(res, ensure_ascii=False, default=float)[:400])
