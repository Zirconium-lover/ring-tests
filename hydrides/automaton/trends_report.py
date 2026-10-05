"""Рисунок и таблица по проверкам trends.py: python trends_report.py папка"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, INK, MUTED, BG, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dc"
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
rows = [json.load(open(f)) for f in glob.glob(os.path.join(D, "*.json"))]
R = {}
for r in rows:
    c = r["config"]["case"]
    R.setdefault(c["name"].rsplit("_seed", 1)[0], []).append(r["metrics"])
SIG = [0, 50, 100, 150, 200, 250]
HS = [60, 100, 180, 300, 450]
LVL = 0.35


def curve(prefix, sig=SIG, key="RHF_image"):
    return np.array([np.mean([m[key] for m in R.get(f"{prefix}_s{s}", [{key: np.nan}])]) for s in sig])


def thr(m, sig=SIG):
    k = np.argmax(m >= LVL)
    return float(np.interp(LVL, m[k - 1:k + 1], sig[k - 1:k + 1])) if k > 0 and m[k] >= LVL else np.nan


fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=BG, gridspec_kw=dict(wspace=0.3))
ax = axs[0]
for H, col in zip(HS, RAMP):
    m = curve(f"H{H}", key="RHF_image") if False else np.array(
        [np.mean([x["RHF_image"] for x in R.get(f"H{H}_s{s}_const", [{"RHF_image": np.nan}])]) for s in SIG])
    ax.plot(SIG, m, "o-", color=col, lw=2, ms=4, label=f"{H} ppm")
ax.axhline(LVL, color=GRID, lw=1, ls=(0, (3, 3)))
ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("RHF"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, title="водород", title_fontsize=9)
ax.set_title("а) RHF при разном водороде (β, потолок постоянны)", loc="left", fontsize=10.5, color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = axs[1]
hh = np.linspace(30, 480, 100)
ax.plot(hh, 110 + 65 * (1 - np.exp(-hh / 65)), color=INK, lw=1.5, ls="--")
ax.text(470, 182, "Desquines и др. 2014\n(Zry-4, опыт)", fontsize=8.5, color=INK, ha="right", va="bottom")
table = {}
for mode, col, lab in (("const", BLUE, "автомат: β и потолок постоянны"), ("T", ORANGE, "автомат: β ∝ 1/T, потолок ∝ σ_y(T)")):
    t = []
    for H in HS:
        m = np.array([np.mean([x["RHF_image"] for x in R.get(f"H{H}_s{s}_{mode}", [{"RHF_image": np.nan}])]) for s in SIG])
        t.append(thr(m)); table[(mode, H)] = (m, t[-1])
    ax.plot(HS, t, "o-", color=col, lw=2, ms=6, label=lab)
ax.set_xlabel("водород, ppm"); ax.set_ylabel(f"порог (RHF = {LVL}), МПа"); ax.set_ylim(0, 220); ax.set_xlim(0, 480)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_title("б) порог переориентации от водорода", loc="left", fontsize=10.5, color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = axs[2]
TM = [350, 370, 390, 420]
for s, col in zip([100, 150, 200, 250], RAMP[1:]):
    m = [np.mean([x["RHF_image"] for x in R.get(f"Tmax{T}_s{s}", [{"RHF_image": np.nan}])]) for T in TM]
    ax.plot(TM, m, "o-", color=col, lw=2, ms=5, label=f"{s} МПа")
ax.axvline(403.7, color=GRID, lw=1, ls=(0, (4, 3)))
ax.text(401, 0.95, "TSSD(180 ppm)", fontsize=8.5, color=MUTED, ha="right")
ax.set_xlabel("максимальная температура цикла T_max, °C"); ax.set_ylabel("RHF"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, title="напряжение", title_fontsize=9, loc="lower right")
ax.set_title("в) неполное растворение (180 ppm)", loc="left", fontsize=10.5, color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.suptitle("Проверка инструмента на общие закономерности (Zry-4, χ₀ = 30°, поле 240 × 240 мкм, 2 затравки)",
             x=0.01, ha="left", fontsize=12, color=INK)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "fig7_trends.png"), dpi=115, facecolor=BG, bbox_inches="tight")
for (mode, H), (m, t) in sorted(table.items()):
    print(f"{mode:5s} H={H:3d}: RHF " + " ".join(f"{v:.2f}" for v in m) + f"  порог {t:.0f} МПа" + f"   опыт {110 + 65 * (1 - np.exp(-H / 65)):.0f}")
for s in [100, 150, 200, 250]:
    print(f"T_max, σ={s}: " + " ".join(f"{T}:{np.mean([x['RHF_image'] for x in R.get(f'Tmax{T}_s{s}', [{'RHF_image': np.nan}])]):.2f}" for T in TM))
