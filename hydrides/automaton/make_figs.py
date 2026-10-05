"""Рисунки по результатам перебора автомата."""
import sys, os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import find_peaks

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, BG, LIGHT = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb", "#a9a8a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
# Lepine 2026, табл. 3.1 (Zircaloy-4, 177.8 wppm, один цикл, напряжение вдоль TD)
LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}
# Cinbiz (по Lepine, табл. 3.2): медиана наклона пластинок от TD
CIN = [(0, 19.4, "0 циклов"), (225, 41.8, "1 цикл, 225 МПа"), (225, 44.0, "4 цикла, 225 МПа")]


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(d):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(d, "*.json"))]
    return rows


def agg(rows, key, by="sigma_app", **flt):
    sel = [r for r in rows if all(r.get(k) == v for k, v in flt.items())]
    xs = sorted({r[by] for r in sel})
    m, lo, hi = [], [], []
    for x in xs:
        v = np.array([r[key] for r in sel if r[by] == x and r[key] is not None], float)
        v = v[np.isfinite(v)]
        m.append(v.mean() if v.size else np.nan); lo.append(v.min() if v.size else np.nan); hi.append(v.max() if v.size else np.nan)
    return np.array(xs), np.array(m), np.array(lo), np.array(hi)


def fig_calibration(rows, out):
    betas = sorted({r["beta"] for r in rows})
    cols = [LIGHT, BLUE, AQUA, ORANGE, YELLOW, INK]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=BG, gridspec_kw=dict(wspace=0.25))
    err = {}
    for b, c in zip(betas, cols):
        x, m, lo, hi = agg(rows, "RHF_image", beta=b)
        axs[0].fill_between(x, lo, hi, color=c, alpha=0.18, lw=0)
        axs[0].plot(x, m, "o-", color=c, lw=2, ms=5, label=f"β = {b} 1/МПа")
        e = []
        for s, ref in LEP.items():
            if s in x:
                e.append((m[list(x).index(s)] - np.mean(ref)) ** 2)
        err[b] = np.sqrt(np.mean(e)) if e else np.nan
        x2, m2, lo2, hi2 = agg(rows, "packet_RHF", beta=b)
        axs[1].plot(x2, m2, "o-", color=c, lw=2, ms=5, label=f"β = {b}")
    for s, (a, b_) in LEP.items():
        for ax in axs:
            ax.plot([s], [a], "s", color=INK, ms=8, zorder=5)
            ax.plot([s], [b_], "D", color=INK, ms=7, mfc="none", zorder=5)
    axs[0].plot([], [], "s", color=INK, label="Lepine: MATLAB"); axs[0].plot([], [], "D", color=INK, mfc="none", label="Lepine: HAPPy")
    for ax, t in zip(axs, ("а) RHF по «снимку» автомата (как сверяли с MATLAB)", "б) RHF пакетов (по кластерам пластинок)")):
        ax.axvline(155, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=0)
        ax.text(158, 0.95, "порог ≈ 155 МПа\n(Lepine, рис. 1.3)", fontsize=8.5, color=MUTED, va="top")
        ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("RHF"); ax.set_ylim(0, 1)
        ax.set_title(t, loc="left", fontsize=10.5, color=INK); tidy(ax)
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return err


def draw_plates(ax, P, size, lw=1.0):
    for cy, cx, psi, half in P:
        dy, dx = -np.sin(psi) * half, np.cos(psi) * half
        dev = np.degrees(abs(np.arctan(np.tan(psi))))
        col = BLUE if dev <= 40 else (ORANGE if dev >= 65 else LIGHT)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=col, lw=lw, solid_capstyle="butt")
    ax.set_xlim(0, size[1]); ax.set_ylim(size[0], 0); ax.set_aspect("equal")


def spacing(hyd, dx, axis):
    """Шаг пакетов по автокорреляции профиля плотности вдоль оси (0 — ND, 1 — TD)."""
    prof = hyd.mean(axis=1 - axis)
    prof = ndi.gaussian_filter1d(prof - prof.mean(), 2.0 / dx, mode="wrap")
    F = np.fft.rfft(prof)
    ac = np.fft.irfft(np.abs(F) ** 2, n=prof.size)
    ac /= ac[0]
    lags = np.arange(prof.size) * dx
    half = prof.size // 2
    pk, pr = find_peaks(ac[:half], prominence=0.1)
    pk = pk[lags[pk] > 15]
    return (float(lags[pk[0]]), float(ac[pk[0]])) if pk.size else (np.nan, np.nan)
