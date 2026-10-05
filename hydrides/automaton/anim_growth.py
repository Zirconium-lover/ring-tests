"""Анимация роста гидридов в автомате: без нагрузки и при окружном напряжении 250 МПа.

Фон — где сейчас выгоднее всего зародиться следующей пластинке (темнее — выгоднее).
Пластинки появляются в том порядке, в каком их выбирает автомат; последняя обведена.
Запуск: python anim_growth.py [папка_вывода]  → anim_growth.mp4 и anim_growth.gif
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_hydride import Params, run
from ca_analysis import simon_w

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
BLUE, ORANGE, INK, MUTED, BG, LIGHT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#a9a8a2"
FAV = LinearSegmentedColormap.from_list("fav", [BG, "#e7e5df", "#b9b6ad", "#6f6c64"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
EVERY = 3                       # кадр — каждые 3 пластинки
SIZE = (200.0, 200.0)
CASES = [("без нагрузки", 0.0), ("окружное напряжение 250 МПа", 250.0)]


def record(sigma):
    frames = []

    def cb(plates, expo, hyd, cH):
        n = len(plates)
        if n % EVERY == 0 and (not frames or frames[-1]["n"] != n):
            ok = np.isfinite(expo)
            lw = np.full(expo.shape, -8.0)
            lw[ok] = (expo[ok] - expo[ok].max()) / np.log(10)        # log10 веса
            lw = ndi.zoom(np.clip(lw, -6, 0), 0.5, order=1)
            frames.append(dict(n=n, plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in plates]),
                               fav=lw.astype(np.float32), frac=float(hyd.mean())))

    r = run(Params(size_um=SIZE, sigma_app=sigma, beta=0.12, sigma_cap=90, capture_um=35, seed=7), callback=cb)
    P = np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]])
    frames.append(dict(n=len(P), plates=P, fav=frames[-1]["fav"], frac=float(r["hyd"].mean())))
    return frames


def segs_colors(P):
    if len(P) == 0:
        return np.zeros((0, 2, 2)), []
    cy, cx, psi, half = P.T
    dy, dx = -np.sin(psi) * half, np.cos(psi) * half
    segs = np.stack([np.stack([cx - dx, cy - dy], 1), np.stack([cx + dx, cy + dy], 1)], 1)
    dev = np.degrees(np.abs(np.arctan(np.tan(psi))))
    cols = [BLUE if d <= 40 else (ORANGE if d >= 65 else LIGHT) for d in dev]
    return segs, cols


def rhf(P):
    if len(P) == 0:
        return 0.0
    dev = np.degrees(np.abs(np.arctan(np.tan(P[:, 2]))))
    L = 2 * P[:, 3]
    return float((L * simon_w(dev)).sum() / L.sum())


data = [record(s) for _, s in CASES]
# синхронизация панелей по доле гидрида: кадр i — одна и та же доля от итоговой
nfr = 90
hold = 24
for k, d in enumerate(data):
    fr = np.array([f["frac"] for f in d])
    tgt = np.linspace(0, fr[-1], nfr)
    data[k] = [d[int(np.argmin(np.abs(fr - t)))] for t in tgt]
fig, axs = plt.subplots(1, 2, figsize=(12.4, 7.0), facecolor=BG, gridspec_kw=dict(wspace=0.08))
fig.subplots_adjust(top=0.86, bottom=0.2, left=0.07, right=0.98)
fig.suptitle("Клеточный автомат: каждая новая пластинка гидрида зарождается там, где это выгоднее всего",
             x=0.07, ha="left", y=0.97, fontsize=13, color=INK)
arts = []
for ax, (title, s), d in zip(axs, CASES, data):
    im = ax.imshow(d[0]["fav"], cmap=FAV, vmin=-5, vmax=0, extent=(0, SIZE[1], SIZE[0], 0), interpolation="bilinear")
    lc = LineCollection([], linewidths=1.6, capstyle="butt"); ax.add_collection(lc)
    last = LineCollection([], linewidths=4.2, colors=INK, capstyle="round", zorder=1); ax.add_collection(last)
    ax.set_xlim(0, SIZE[1]); ax.set_ylim(SIZE[0], 0); ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel("TD — вдоль дуги, мкм")
    txt = ax.text(0.0, -0.12, "", transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
    arts.append((im, lc, last, txt))
axs[0].set_ylabel("ND — через стенку, мкм")
axs[1].set_yticklabels([])
fig.text(0.01, 0.015, "Фон — выгодность зарождения следующей пластинки (темнее — выгоднее; учтены поле соседних пластинок, "
         "текстура, напряжение и запас водорода).\nПластинки: синие — окружные (≤ 40° к дуге), серые — 40–65°, оранжевые — радиальные (≥ 65°). "
         "Чёрная — только что выросшая.", fontsize=9.5, color=MUTED)


def draw(i):
    for (im, lc, last, txt), d in zip(arts, data):
        f = d[min(i, len(d) - 1)]
        im.set_data(f["fav"])
        segs, cols = segs_colors(f["plates"])
        lc.set_segments(segs); lc.set_color(cols)
        k = min(EVERY, len(segs))
        last.set_segments(segs[-k:] if i < len(d) - 1 and len(segs) else [])
        txt.set_text(f"пластинок {f['n']:4d}   гидрида {f['frac'] * 100:.2f} %   RHF {rhf(f['plates']):.2f}")
    return []


ani = FuncAnimation(fig, draw, frames=nfr + hold, interval=80)
os.makedirs(OUT, exist_ok=True)
ani.save(os.path.join(OUT, "anim_growth.mp4"), writer=FFMpegWriter(fps=12, bitrate=2400), dpi=110,
         savefig_kwargs=dict(facecolor=BG))
ani.save(os.path.join(OUT, "anim_growth.gif"), writer=PillowWriter(fps=12), dpi=58,
         savefig_kwargs=dict(facecolor=BG))
draw(nfr + hold - 1)
fig.savefig(os.path.join(OUT, "anim_growth_last.png"), dpi=110, facecolor=BG)
print("кадров", nfr + hold, [len(d) for d in data])
