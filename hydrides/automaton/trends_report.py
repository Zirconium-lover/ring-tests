"""Рисунок и таблица по проверкам trends.py: python trends_report.py папка [имя_рисунка] [подпись]"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
FIGNAME = sys.argv[2] if len(sys.argv) > 2 else "fig7_trends.png"
TITLE = sys.argv[3] if len(sys.argv) > 3 else "последовательный движок"
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


MODES_ALL = sorted({r["config"]["case"]["mode"] for r in rows})
M0 = "const" if "const" in MODES_ALL else MODES_ALL[0]
HAS_RATE = any(r["config"]["case"]["name"].startswith("rate") for r in rows)
fig, axs = plt.subplots(1, 4 if HAS_RATE else 3, figsize=(21 if HAS_RATE else 16, 4.8), facecolor=BG, gridspec_kw=dict(wspace=0.3))
ax = axs[0]
for H, col in zip(HS, RAMP):
    m = curve(f"H{H}", key="RHF_image") if False else np.array(
        [np.mean([x["RHF_image"] for x in R.get(f"H{H}_s{s}_{M0}", [{"RHF_image": np.nan}])]) for s in SIG])
    ax.plot(SIG, m, "o-", color=col, lw=2, ms=4, label=f"{H} ppm")
ax.axhline(LVL, color=GRID, lw=1, ls=(0, (3, 3)))
ax.set_xlabel("окружное напряжение при охлаждении, МПа"); ax.set_ylabel("RHF"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, title="водород", title_fontsize=9)
ax.set_title("а) RHF при разном водороде", loc="left", fontsize=10.5, color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = axs[1]
hh = np.linspace(30, 480, 100)
ax.plot(hh, 110 + 65 * (1 - np.exp(-hh / 65)), color=INK, lw=1.5, ls="--")
ax.text(470, 182, "Desquines и др. 2014\n(Zry-4, опыт)", fontsize=8.5, color=INK, ha="right", va="bottom")
table = {}
LABS = {"const": (BLUE, "по одной: β и потолок постоянны"), "T": (ORANGE, "по одной: β ∝ 1/T, потолок ∝ σ_y(T)"),
        "kin": (AQUA, "кинетический: одновременное зарождение")}
for mode in MODES_ALL:
    col, lab = LABS[mode]
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
if HAS_RATE:
    ax = axs[3]
    RT = [1, 3, 10, 30]
    for s_, col in zip([100, 150, 200, 250], RAMP[1:]):
        m = [np.mean([x["RHF_image"] for x in R.get(f"rate{q}_s{s_}", [{"RHF_image": np.nan}])]) for q in RT]
        ax.plot(RT, m, "o-", color=col, lw=2, ms=5, label=f"{s_} МПа")
    ax.set_xscale("log"); ax.set_xticks(RT); ax.set_xticklabels([str(q) for q in RT])
    ax.set_xlabel("скорость охлаждения, °C/мин"); ax.set_ylabel("RHF"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, title="напряжение", title_fontsize=9, loc="lower right")
    ax.set_title("г) скорость охлаждения (180 ppm)", loc="left", fontsize=10.5, color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.suptitle(f"Проверка инструмента на общие закономерности: {TITLE} (Zry-4, χ₀ = 30°, поле 240 × 240 мкм, 2 затравки)",
             x=0.01, ha="left", fontsize=12, color=INK)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", FIGNAME), dpi=115, facecolor=BG, bbox_inches="tight")
for (mode, H), (m, t) in sorted(table.items()):
    print(f"{mode:5s} H={H:3d}: RHF " + " ".join(f"{v:.2f}" for v in m) + f"  порог {t:.0f} МПа" + f"   опыт {110 + 65 * (1 - np.exp(-H / 65)):.0f}")
for s in [100, 150, 200, 250]:
    if HAS_RATE:
        print(f"скорость, σ={s}: " + " ".join(f"{q}:{np.mean([x['RHF_image'] for x in R.get(f'rate{q}_s{s}', [{'RHF_image': np.nan}])]):.2f}" for q in [1, 3, 10, 30]))
    print(f"T_max, σ={s}: " + " ".join(f"{T}:{np.mean([x['RHF_image'] for x in R.get(f'Tmax{T}_s{s}', [{'RHF_image': np.nan}])]):.2f}" for T in TM))
