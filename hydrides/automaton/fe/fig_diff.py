"""Рисунок к fe/diff_check.py: шаг диффузии Фурье против точного решения и CalculiX; схема с напряжениями."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "diff_check.json")))
INK, MUTED = "#0b0b0b", "#52514e"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# (а) ошибка против точного решения: Фурье и МКЭ
A = R["A"]
lab = [f"{a['T']} °C, {a['t']:g} с\nL={a['Ldiff']:.1f} мкм" for a in A]
x = np.arange(len(A))
ff = [max(a["fft_far"], 1e-4) for a in A]
fe50 = [a.get("ccx50_far", np.nan) for a in A]
fe400 = [a.get("ccx400_far", np.nan) for a in A]
w = 0.26
ax[0].bar(x - w, ff, w - 0.03, color=C1, label="Фурье (один шаг)")
ax[0].bar(x, fe50, w - 0.03, color=C2, label="CalculiX, 50 шагов")
ax[0].bar(x + w, fe400, w - 0.03, color=C3, label="CalculiX, 400 шагов")
ax[0].set_yscale("log")
ax[0].set_xticks(x, lab, fontsize=8)
ax[0].set_ylabel("макс. ошибка дальше 1 мкм от пластинки, ppm", color=MUTED)
ax[0].set_title("а) Фик: ошибка против точного решения", loc="left", color=INK)
ax[0].legend(frameon=False, fontsize=8)
ax[0].grid(axis="y", color="#e6e6e3", lw=0.6)

# (б) профиль за кончиком пластинки при короткой диффузионной длине
a = next((a for a in A if a["T"] == 150 and abs(a["t"] - 0.1) < 1e-9 and a["d"] == "0deg" and "prof" in a), None)
if a is not None:
    p = a["prof"]
    xs = (np.arange(len(p["ex"])) - len(p["ex"]) / 2 + 0.5) * 0.4
    sel = xs > 2.5
    for key, col, lw_, name in (("ex", INK, 2.0, "точное"), ("ff", C1, 2.0, "Фурье"), ("fe", C3, 2.0, "CalculiX")):
        ax[1].plot(xs[sel], 100 - np.array(p[key])[sel], color=col, lw=lw_, label=name,
                   ls="--" if key == "ex" else "-", marker="o" if key != "ex" else None, ms=3)
    ax[1].axhline(100, color=MUTED, lw=0.8, ls=":")
    ax[1].set_xlabel("расстояние от центра пластинки вдоль неё, мкм (кончик — 2.5)", color=MUTED)
    ax[1].set_ylabel("водород в растворе, ppm", color=MUTED)
    ax[1].set_title(f"б) 150 °C, шаг 0.1 с: L = {a['Ldiff']:.2f} мкм ≈ 2 клетки", loc="left", color=INK)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].grid(color="#e6e6e3", lw=0.6)

# (в) диффузия с напряжениями: схема автомата против эталона
B = [b for b in R["B"] if b["cap"] == 300.0]
lab = [f"{b['T']} °C, {b['t']:g} с" for b in B]
x = np.arange(len(B))
ax[2].bar(x - w, [b["stress_effect"] for b in B], w - 0.03, color=MUTED, label="сам эффект напряжений")
ax[2].bar(x, [b["slotboom_err"] for b in B], w - 0.03, color=C2, label="ошибка: схема до исправления")
ax[2].bar(x + w, [b.get("masked_err", np.nan) for b in B], w - 0.03, color=C1, label="ошибка: φ = 0 в гидриде")
ax[2].set_yscale("log")
ax[2].set_xticks(x, lab, fontsize=8)
ax[2].set_ylabel("макс. дальше 1 мкм от пластинки, ppm", color=MUTED)
ax[2].set_title("в) σ_h в диффузии: эталон — конечные объёмы", loc="left", color=INK)
ax[2].legend(frameon=False, fontsize=8)
ax[2].grid(axis="y", color="#e6e6e3", lw=0.6)

for a_ in ax:
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "figs", "fig_fe10_diffusion.png")
fig.savefig(out, dpi=130, facecolor="white")
print("→", out)
