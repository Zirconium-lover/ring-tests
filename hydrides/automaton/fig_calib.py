"""Рис. 2 — калибровка по Lepine (табл. 3.1) и проверка наклонов пластинок; рис. 4 — структуры."""
import json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figs import BLUE, ORANGE, AQUA, INK, MUTED, GRID, BG, LIGHT, tidy, draw_plates

R = "/tmp/claude-0/-home-user/16339a70-3da3-5a74-97cf-f17fbd8cb16f/scratchpad/ca_runs/"
rows = [json.load(open(f)) for f in glob.glob(R + "cal7/*.json")]
rows = [r for r in rows if r.get("capture_um") == 35.0]
rows += [json.load(open(f)) for f in glob.glob(R + "v2/bist_*.json")]
S = sorted({r["sigma_app"] for r in rows})


def stat(key):
    m, lo, hi = [], [], []
    for s in S:
        v = np.array([np.nan if r.get(key) is None else r[key] for r in rows if r["sigma_app"] == s], float)
        v = v[np.isfinite(v)]
        m.append(v.mean() if v.size else np.nan); lo.append(v.min() if v.size else np.nan); hi.append(v.max() if v.size else np.nan)
    return np.array(m), np.array(lo), np.array(hi)


fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.9), facecolor=BG, gridspec_kw=dict(wspace=0.27))
ax = axs[0]
for key, col, lab, mk in (("RHF_image", BLUE, "автомат: RHF по «снимку» (как MATLAB-сверка)", "o-"),
                          ("packet_RHF", AQUA, "автомат: RHF пакетов", "s--")):
    m, lo, hi = stat(key)
    ax.fill_between(S, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot(S, m, mk, color=col, lw=2, ms=5, label=lab)
LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}
for s, (a, b) in LEP.items():
    ax.plot([s], [a], "s", color=INK, ms=8, zorder=5); ax.plot([s], [b], "D", color=INK, mfc="none", ms=7, zorder=5)
ax.plot([], [], "s", color=INK, label="Lepine, MATLAB"); ax.plot([], [], "D", color=INK, mfc="none", label="Lepine, HAPPy")
ax.axvline(155, color=GRID, lw=1.2, ls=(0, (4, 3)))
ax.text(158, 0.06, "порог ≈ 155 МПа\n(образец переменного\nсечения, рис. 1.3)", fontsize=8.5, color=MUTED)
ax.set_ylim(0, 1); ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("RHF")
ax.set_title("а) Калибровка: β = 0.12 1/МПа, σ_cap = 90 МПа, ℓ = 35 мкм", loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper left"); tidy(ax)
ax = axs[1]
for key, col, lab, mk in (("dev_in_circ", BLUE, "автомат: в окружных пакетах", "o-"),
                          ("dev_in_radial", ORANGE, "автомат: в радиальных пакетах", "s-")):
    m, lo, hi = stat(key)
    ok = np.isfinite(m)
    ax.fill_between(np.array(S)[ok], lo[ok], hi[ok], color=col, alpha=0.15, lw=0)
    ax.plot(np.array(S)[ok], m[ok], mk, color=col, lw=2, ms=5, label=lab)
ax.axhspan(17, 21, color=BLUE, alpha=0.25, lw=0)
ax.text(5, 22, "снимки Cinbiz: окружные пакеты (≈19°)", fontsize=8.5, color=INK)
ax.axhspan(47, 52, color=ORANGE, alpha=0.25, lw=0)
ax.text(5, 53, "снимки Cinbiz: радиальные пакеты (≈47–52°)", fontsize=8.5, color=INK)
ax.axhline(45, color=MUTED, lw=1, ls=(0, (2, 3)))
ax.set_ylim(0, 90); ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("медиана наклона пластинок к TD, °")
ax.set_title("б) Проверка (не подгонялось): наклон пластинок в пакетах", loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower right"); tidy(ax)
fig.savefig("figs/fig2_calibration.png", dpi=115, bbox_inches="tight", facecolor=BG)
plt.close(fig)

# --- рис. 4: структуры при 0, 150, 250 МПа
fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=BG, gridspec_kw=dict(wspace=0.12))
for ax, s in zip(axs, (0, 150, 250)):
    f = glob.glob(R + f"cal7/beta0.12_capture_um35.0_seed1_sigma_app{s}_sigma_cap90.npz")[0]
    z = np.load(f)
    draw_plates(ax, z["plates"], (240, 240), lw=1.1)
    r = [q for q in rows if q["sigma_app"] == s and q["seed"] == 1][0]
    ax.set_title(f"{'абв'[[0, 150, 250].index(s)]}) {s} МПа: RHF = {r['RHF_image']:.2f}", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("TD (вдоль дуги), мкм")
    if s == 0: ax.set_ylabel("ND (через стенку), мкм")
fig.text(0.01, -0.02, "Каждая чёрточка — пластинка (синие ≤ 40° к TD, серые 40–65°, оранжевые ≥ 65°). Поле 240 × 240 мкм, ≈ 180 wppm H, текстура Zry-4 листа.",
         fontsize=9, color=MUTED)
fig.savefig("figs/fig4_structures.png", dpi=115, bbox_inches="tight", facecolor=BG)
print("ok")
