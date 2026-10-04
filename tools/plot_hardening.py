#!/usr/bin/env python3
"""Кривые упрочнения Э635 для модели: B (базовая) и A (чувствительность).

    python3 tools/plot_hardening.py -o notes/figs/hardening_AB.png

B: σ = 653·(e0 + ε_p)^0.082, e0 = 0.0116 (σ(0.002) = 459 МПа; лист Mat B, Zhang 2023).
A: σ = 586·(0.002 + ε_p)^0.049 (подбор по окружным σ_0.2 = 448, σ_в = 483 МПа,
   оболочка Э635, novikov2006).
Точки — σ_0.2 из источников; ромбы — максимум инженерного напряжения
(условие Консидера для закона Свифта: ε_p = n − e0).
"""
import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_B, C_A = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'

CURVES = [
    ('B (базовая): 653·(0.0116 + ε_p)^0.082', 653.0, 0.0116, 0.082, C_B),
    ('A (чувствительность): 586·(0.002 + ε_p)^0.049', 586.0, 0.002, 0.049, C_A),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    eps = [i / 1000.0 for i in range(0, 1001)]
    for label, K, e0, n, col in CURVES:
        ax.plot(eps, [K * (e0 + e) ** n for e in eps], color=col, lw=2, label=label)
        eu = n - e0
        s_true = K * n ** n
        s_eng = s_true * math.exp(-eu)
        ax.plot([eu], [s_true], marker='D', ms=8, color=col, mec='white', mew=1.5, zorder=4)
        ax.annotate('σ_в = %.0f МПа (инж.)\nпри ε_p = %.3f' % (s_eng, eu),
                    xy=(eu, s_true), xytext=((0.24, 482) if col == C_A else (eu + 0.06, s_true + 25)),
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    pts = [(0.002, 448, 'Э635, оболочка, окружн.\n(novikov2006): σ_0.2 = 448'),
           (0.002, 459, 'лист Mat B, TD (Zhang 2023):\nσ_0.2 = 459')]
    for (x, y, t), ty in zip(pts, (398, 432)):
        ax.plot([x], [y], 'o', ms=8, color=INK, mec='white', mew=1.5, zorder=5)
        ax.annotate(t, xy=(x, y), xytext=(0.06, ty), fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(380, 700)
    ax.set_xlabel('эквивалентная пластическая деформация ε_p', color=INK)
    ax.set_ylabel('истинное напряжение течения σ, МПа', color=INK)
    ax.set_title('Э635: кривые упрочнения для изотропной модели (таблица *Plastic до ε_p = 2)',
                 fontsize=11, color=INK, loc='left')
    ax.grid(color=GRID, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.savefig(a.o, dpi=140, bbox_inches='tight', facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
