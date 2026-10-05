"""Рис. 11: карта притяжения следующей пластинки в 2D и в 3D (поле одной пластинки, микроупругость).
python fig_map3d.py (сначала map3d.py и map3d.py shapes — числа для нижней панели)"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import map_coordinates
import map3d as M

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#9cc3ef", "#efeeea", "#f4b49a", ORANGE])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
LIM, VMAX = 5.0, 150.0
NORM = TwoSlopeNorm(0, -VMAX, VMAX)


def plane(g, f, e1, e2):
    """Срез поля g через центр пластинки по плоскости с осями e1, e2 (мкм): (картинка, маска пластинки)."""
    s = np.linspace(-LIM, LIM, 161)
    A, B = np.meshgrid(s, s, indexing="ij")
    P = M.C0[:, None, None] + A * e1[:, None, None] + B * e2[:, None, None]
    idx = P / M.DX - 0.5
    gi = map_coordinates(g, idx, order=1, mode="wrap")
    fi = map_coordinates(f, idx, order=1, mode="wrap")
    return gi, fi


def show(ax, gi, fi, title, xl, yl, flip=True):
    img = np.where(fi > 0.3, np.nan, np.clip(gi, -VMAX, VMAX)).T
    ext = (-LIM, LIM, LIM, -LIM) if flip else (-LIM, LIM, -LIM, LIM)
    im = ax.imshow(img, cmap=DIV, norm=NORM, extent=ext, origin="upper" if flip else "lower")
    ax.imshow(np.where(fi.T > 0.3, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("k", [INK, INK]), extent=ext,
              origin="upper" if flip else "lower", vmin=0, vmax=1)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
    return im


ex, ey, ez = np.eye(3)
fig = plt.figure(figsize=(17, 12.5), facecolor=BG)
gs = fig.add_gridspec(3, 6, height_ratios=(1, 1, 1.05), width_ratios=(1, 1, 1, 1, 1, 0.06), hspace=0.38, wspace=0.34)
HEAD = ("2D: следующая ∥", "2D: следующая −ψ", "3D-диск: следующая ∥,\nв плоскости пластинки",
        "3D-диск: следующая −ψ,\nсечение r–θ", "3D-диск: следующая −ψ,\nсечение TD – ось трубы")
for row, psi in enumerate((48, 70)):
    n1, n2 = M.normal(psi), M.normal(-psi)
    t = np.array([n1[1], -n1[0], 0.0]); t /= np.linalg.norm(t)
    fs, fd = M.strip_fraction(n1), M.disc_field(n1)
    hd = HEAD if row == 0 else ("",) * 5
    g = M.gain(fs, n1, n1)
    a0 = fig.add_subplot(gs[row, 0])
    im = show(a0, *plane(g, fs, ex, ey), hd[0], "TD, мкм", "ND (r), мкм")
    a0.text(-0.42, 0.5, f"ψ = {psi}°" + ("\n(как у Lepine)" if psi == 48 else ""), transform=a0.transAxes, rotation=90,
            ha="center", va="center", fontsize=11.5, color=INK)
    g = M.gain(fs, n1, n2)
    show(fig.add_subplot(gs[row, 1]), *plane(g, fs, ex, ey), hd[1], "TD, мкм", "ND (r), мкм")
    g = M.gain(fd, n1, n1)
    show(fig.add_subplot(gs[row, 2]), *plane(g, fd, t, ez), hd[2], "вдоль следа в r–θ, мкм", "ось трубы, мкм", flip=False)
    g = M.gain(fd, n1, n2)
    show(fig.add_subplot(gs[row, 3]), *plane(g, fd, ex, ey), hd[3], "TD, мкм", "ND (r), мкм")
    show(fig.add_subplot(gs[row, 4]), *plane(g, fd, ex, ez), hd[4], "TD, мкм", "ось трубы, мкм", flip=False)
cb = fig.colorbar(im, cax=fig.add_subplot(gs[0:2, 5]))
cb.set_label("выгода зарождения g, МПа (оранжевый — выгодно, чёрное — пластинка)")

# нижний ряд: доля притяжения вдоль оси трубы против вытянутости пластинки
num = json.load(open(os.path.join(HERE, "figs", "map3d_numbers.json")))
ax = fig.add_subplot(gs[2, 0:2])
asp = [1, 2, 4, 8]
for psi, c in ((48, BLUE), (70, ORANGE)):
    for kind, ls in (("parallel", "-"), ("mirror", "--")):
        y = [num[f"ell_psi{psi}_az{az}_{kind}"]["L"] for az in (1.25, 2.5, 5.0, 10.0)]
        ax.plot(asp, y, ls, color=c, lw=2, marker="o", ms=5, label=f"ψ {psi}°, следующая {'∥' if kind == 'parallel' else '−ψ'}")
ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.93))
ax.axhline(0, color=MUTED, lw=0.8)
ax.text(1, 0.03, "2D: 0 — притяжение только в плоскости r–θ", fontsize=9, color=MUTED)
ax.set_xscale("log", base=2); ax.set_xticks(asp, ["1 (диск)", "2", "4", "8"])
ax.set_xlim(0.9, 9); ax.set_ylim(-0.05, 1.05); ax.grid(color=GRID)
ax.set_xlabel("вытянутость пластинки вдоль оси трубы, раз (полуось вдоль следа 1.25 мкм)")
ax.set_ylabel("доля вдоль оси трубы")
ax.set_title("Куда тянется цепочка: доля притяжения вдоль оси трубы", loc="left", fontsize=10.5, color=INK)

ax = fig.add_subplot(gs[2, 2:6]); ax.axis("off")
rows = [("", "2D", "3D-диск", "3D, лента ×8")]
for psi in (20, 48, 70):
    for kind, nm in (("parallel", "∥"), ("mirror", "зерк.")):
        a = num[f"2D_psi{psi}_{kind}"]; b = num[f"3D_psi{psi}_{kind}"]; c = num[f"ell_psi{psi}_az10.0_{kind}"]
        def fmt(d, three=True):
            if d["chain_anis"] < 0.1:
                return "в сечении нет"
            s = f"{d['chain_deg']:.0f}°"
            return s + (f", вдоль оси {d['L']:.0%}" if three else "")
        rows.append((f"ψ {psi}°, {nm}", fmt(a, False), fmt(b), fmt(c)))
tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left", colLoc="left",
              colWidths=(0.16, 0.18, 0.33, 0.33))
tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.7)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(GRID); cell.set_facecolor(BG)
    if r == 0:
        cell.set_text_props(color=INK, weight="bold")
ax.set_title("Направление цепочки в сечении r–θ (0° — по окружности, 90° — по радиусу)\nи доля притяжения вдоль оси трубы",
             loc="left", fontsize=10.5, color=INK)
fig.suptitle("Карта притяжения: где поле пластинки выгодно для следующей. Диск 3 мкм, толщина 0.6 мкм, β 0.12, потолок 90 МПа",
             x=0.01, y=0.95, ha="left", fontsize=12.5, color=INK)
fig.savefig(os.path.join(HERE, "figs", "fig11_map3d.png"), dpi=100, facecolor=BG, bbox_inches="tight")
print("ok")
