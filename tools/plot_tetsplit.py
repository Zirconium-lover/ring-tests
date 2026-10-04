#!/usr/bin/env python3
"""Сравнение сеток C3D4 (разбиение шестигранника и шаг) у края сегмента.

    python3 tools/plot_tetsplit.py A.inp B.inp ... --labels "..." "..." -o fig.png

Верхний ряд — сечение z = 0 у края сегмента, нижний — развёртка внутренней
поверхности там же (ведомые узлы контакта). Колоды — tools/mksector.py.
"""
import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from plot_mesh import read_deck, boundary_faces, cyl, C_RING, C_SEG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('decks', nargs='+')
    ap.add_argument('--labels', nargs='+')
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    labels = a.labels or a.decks
    n = len(a.decks)
    fig, axs = plt.subplots(2, n, figsize=(5.2 * n, 9.6))
    for col, (deck, lab) in enumerate(zip(a.decks, labels)):
        nodes, elems, nsets = read_deck(deck)
        rf = boundary_faces(elems, 'RING')
        sf = boundary_faces(elems, 'SEGMENT')
        ring_nodes = {q for t, s, c in elems if s == 'RING' for q in c}
        nring = sum(1 for t, s, c in elems if s == 'RING')
        ri = min(cyl(nodes[q])[0] for q in ring_nodes)
        seg_nodes = {q for t, s, c in elems if s == 'SEGMENT' for q in c}
        phe = max(cyl(nodes[q])[1] for q in seg_nodes)

        ax = axs[0][col]
        z0 = [f for f in rf if all(abs(nodes[q][2]) < 1e-9 for q in f)]
        s0 = [f for f in sf if all(abs(nodes[q][2]) < 1e-9 for q in f)]
        ax.add_collection(PolyCollection([[nodes[q][:2] for q in f] for f in z0],
                                         facecolor=C_RING, edgecolor='k', linewidths=0.4))
        ax.add_collection(PolyCollection([[nodes[q][:2] for q in f] for f in s0],
                                         facecolor=C_SEG, edgecolor='#1d3d63', linewidths=0.3,
                                         alpha=0.7))
        t = math.radians(22.5)
        ax.plot([4.8 * math.cos(t), 6.5 * math.cos(t)], [4.8 * math.sin(t), 6.5 * math.sin(t)],
                'k--', lw=0.8)
        ax.set_xlim(4.95, 5.95); ax.set_ylim(1.55, 2.55); ax.set_aspect('equal')
        ax.set_title('%s\nкольцо: %d C3D4, %d узлов' % (lab, nring, len(ring_nodes)),
                     fontsize=10, loc='left')
        ax.set_xlabel('x, мм'); ax.set_ylabel('y, мм')
        if col == 0:
            ax.annotate('край сегмента', xy=(ri * math.cos(math.radians(phe)),
                                             ri * math.sin(math.radians(phe))),
                        xytext=(5.0, 2.4), fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.7))
            ax.text(5.55, 2.38, 'φ = 22.5°', fontsize=8, rotation=22.5)

        ax = axs[1][col]
        inner = [f for f in rf if all(abs(cyl(nodes[q])[0] - ri) < 1e-6 for q in f)]
        poly = [[(ri * math.radians(cyl(nodes[q])[1]), nodes[q][2]) for q in f] for f in inner]
        ax.add_collection(PolyCollection(poly, facecolor='#e8b7bd', edgecolor='k', linewidths=0.4))
        ax.axvline(ri * math.radians(phe), color=C_SEG, lw=1.5, ls='--')
        ax.text(ri * math.radians(phe) - 0.02, 0.62, 'край\nсегмента', fontsize=8, ha='right',
                color='#1d3d63')
        ax.set_xlim(1.55, ri * math.radians(22.5)); ax.set_ylim(0, 0.7); ax.set_aspect('equal')
        ax.set_xlabel('дуга по R_i, мм'); ax.set_ylabel('z, мм')
        ax.set_title('внутренняя поверхность (развёртка), z от 0', fontsize=10, loc='left')
    fig.suptitle('Сетки C3D4 у края сегмента: сечение z = 0 (вверху) и внутренняя поверхность '
                 '(внизу)', fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(a.o, dpi=140, bbox_inches='tight', facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
