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

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (а) ошибка против точного решения: Фурье и МКЭ
A = R["A"]
lab = [f"{a['T']} °C, {a['t']:g} с" + (", 30°" if a["d"] != "0deg" else "") + f"\nL={a['Ldiff']:.1f} мкм" for a in A]
x = np.arange(len(A))
ff = [max(a["fft_far"], 1e-4) for a in A]          # 0 рисуем как 1e-4
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

# (б) ложное пересыщение: концентрация в матрице выше фона (диффузия такого дать не может)
ax[1].bar(x - w / 2, [a["fft_over"] for a in A], w - 0.03, color=C1, label="Фурье")
ax[1].bar(x + w / 2, [a.get("ccx400_over", a.get("ccx50_over", np.nan)) for a in A], w - 0.03, color=C3,
          label="CalculiX (400 шагов; 50 — для 30 с)")
for i, a in enumerate(A):
    for dxo, v in ((-w / 2, a["fft_over"]), (w / 2, a.get("ccx400_over", a.get("ccx50_over", 0.0)))):
        ax[1].text(i + dxo, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=MUTED)
ax[1].set_xticks(x, lab, fontsize=8)
ax[1].set_ylabel("макс. c − фон в матрице, ppm (точное: 0)", color=MUTED)
ax[1].set_title("б) ложное пересыщение у свежей пластинки", loc="left", color=INK)
ax[1].legend(frameon=False, fontsize=8)
ax[1].grid(axis="y", color="#e6e6e3", lw=0.6)

# (в) диффузия с напряжениями: схема автомата против эталона
B = [b for b in R["B"] if b["cap"] == 300.0]
lab = [f"{b['T']} °C, {b['t']:g} с" for b in B]
x = np.arange(len(B))
ax[2].bar(x - w, [b.get("masked_effect", b["stress_effect"]) for b in B], w - 0.03, color=MUTED, label="сам эффект напряжений")
ax[2].bar(x, [b["slotboom_err"] for b in B], w - 0.03, color=C2, label="ошибка: схема до исправления")
ax[2].bar(x + w, [b.get("masked_err", np.nan) for b in B], w - 0.03, color=C1, label="ошибка: φ = 0 в гидриде")
ax[2].set_yscale("log")
ax[2].set_xticks(x, lab, fontsize=8)
ax[2].set_ylabel("макс. дальше 1 мкм от пластинки, ppm", color=MUTED)
ax[2].set_title("в) σ_h в диффузии против эталона", loc="left", color=INK)
ax[2].legend(frameon=False, fontsize=8)
ax[2].grid(axis="y", color="#e6e6e3", lw=0.6)

for a_ in ax:
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "figs", "fig_fe10_diffusion.png")
fig.savefig(out, dpi=130, facecolor="white")
print("→", out)
