#!/usr/bin/env python3
"""Картинка модели всего кольца (mksector --full-ring) до запуска.

    python3 tools/plot_ring_model.py runs/ring/decks/R12_H3_mu0.05_ecc1.inp -o notes/figs/ring_model.png

а) сечение z = 0: кольцо, все сегменты, узлы против жёсткого смещения,
   место самой тонкой стенки; б) сетка у кромки сегмента; в) толщина стенки
   по окружности (эксцентриситет); г) параметры модели.
"""
import argparse
import math
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post_stage1 import read_deck          # noqa: E402

INK, MUTED, MARK = '#1f1f1e', '#6b6a63', '#eb6834'
C_RING, C_SEG = '#d9898f', '#3f7fc4'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('deck')
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    d = read_deck(a.deck)
    head = open(a.deck).readlines()[1]
    p = dict(re.findall(r'(\w+)=(\[[^\]]*\]|\S+)', head))
    nodes, elems = d['nodes'], d['elems']
    seg_ids = set(d['sets']['SEGNODES'])
    nseg = d['nseg']
    phe = float(p['phi_edge_deg'])
    ecc, phi0 = float(p['ecc']), float(p['phi0'])
    ring_f, seg_f = [], []
    for e, c in elems.items():
        if len(c) < 8:
            continue                              # пружины
        f = c[:4]                                 # грань 1-2-3-4: z = z_min элемента
        if all(abs(nodes[q][2]) < 1e-9 for q in f):
            (seg_f if c[0] in seg_ids else ring_f).append([nodes[q][:2] for q in f])
    nring = sum(1 for e, c in elems.items() if c[0] not in seg_ids)

    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    ax.add_collection(PolyCollection(ring_f, facecolor=C_RING, edgecolor='none'))
    ax.add_collection(PolyCollection(seg_f, facecolor=C_SEG, edgecolor='none', alpha=0.8))
    for k in range(nseg):
        ph = math.radians(2 * d['sector'] * k)
        ax.text(4.3 * math.cos(ph), 4.3 * math.sin(ph), str(k), ha='center', va='center', fontsize=9, color='#1d3d63')
    for name in ('SPR_X', 'SPR_Y'):
        for n in d['sets'][name]:
            x, y = nodes[n][:2]
            ax.plot(x, y, 's', color=INK, ms=6)
            ax.annotate('пружина', (x, y), xytext=(x * 1.2, y * 1.2 + (0.25 if name == 'SPR_Y' else 0)), fontsize=8.5,
                        ha='center', arrowprops=dict(arrowstyle='-', lw=0.6))
    t0 = math.radians(phi0)
    ax.annotate('стенка тоньше всего\n(φ = %g°, −%g %%)' % (phi0, 100 * ecc), (6.3 * math.cos(t0), 6.3 * math.sin(t0)),
                xytext=(7.0, 1.6), fontsize=9.5, color=MARK, arrowprops=dict(arrowstyle='->', color=MARK, lw=1))
    ax.set_xlim(-7.8, 8.6); ax.set_ylim(-7.8, 7.8); ax.set_aspect('equal')
    ax.set_xlabel('x, мм', color=INK); ax.set_ylabel('y, мм', color=INK)
    ax.set_title('а) Сечение z = 0: кольцо (%d C3D8I) и %d жёстких сегментов\n(номера — середины сегментов, '
                 'каждый движется по своей биссектрисе)' % (nring, nseg), loc='left', fontsize=11, color=INK)
    ax.add_patch(plt.Rectangle((5.5 * math.cos(math.radians(phe)) - 0.35, 5.5 * math.sin(math.radians(phe)) - 0.35),
                               0.9, 0.8, fill=False, ec=MARK, lw=1))

    ax2 = fig.add_subplot(gs[0, 1:])
    ex, ey = 5.5 * math.cos(math.radians(phe)), 5.5 * math.sin(math.radians(phe))
    ax2.add_collection(PolyCollection(ring_f, facecolor=C_RING, edgecolor='k', linewidths=0.35))
    ax2.add_collection(PolyCollection(seg_f, facecolor=C_SEG, edgecolor='#1d3d63', linewidths=0.3, alpha=0.8))
    ax2.set_xlim(ex - 0.6, ex + 1.3); ax2.set_ylim(ey - 0.45, ey + 0.55); ax2.set_aspect('equal')
    ax2.annotate('кромка сегмента 0 (φ = %.2f°);\nпропил 0.2 мм, дальше — сегмент 1' % phe, (ex, ey),
                 xytext=(ex + 0.35, ey - 0.38), fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.7))
    ax2.set_xlabel('x, мм', color=INK); ax2.set_ylabel('y, мм', color=INK)
    ax2.set_title('б) Сетка у кромки: шаг 0.1 мм, 8 элементов по толщине', loc='left', fontsize=11, color=INK)

    ax3 = fig.add_subplot(gs[1, 1])
    phs = list(range(0, 361, 2))
    ax3.plot(phs, [0.8 * (1 - ecc * math.cos(math.radians(x - phi0))) for x in phs], color=MARK, lw=2)
    for k in range(nseg):
        c = 2 * d['sector'] * k
        ax3.axvspan(c - phe, c + phe, color=C_SEG, alpha=0.12, lw=0)
    ax3.axvspan(360 - phe, 360, color=C_SEG, alpha=0.12, lw=0)
    ax3.set_xlim(0, 360); ax3.set_xticks(range(0, 361, 60))
    ax3.set_xlabel('φ, град (голубое — сегменты)', color=INK); ax3.set_ylabel('толщина стенки, мм', color=INK)
    ax3.set_title('в) Несовершенство: эксцентриситет ±%g %%' % (100 * ecc), loc='left', fontsize=11, color=INK)
    for s_ in ('top', 'right'):
        ax3.spines[s_].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 2]); ax4.axis('off')
    rows = [
        ('Кольцо', 'Ø 11.0 / 12.6 мм, H = %s мм' % p['H']),
        ('Высота', 'половина, симметрия по z = 0'),
        ('Область', 'всё кольцо 360°, без симметрии по φ'),
        ('Оправка', '%d жёстких сегментов, пропил 0.2 мм' % nseg),
        ('Сетка', '%d C3D8I, %d узлов' % (nring, len(nodes))),
        ('Контакт', 'node-to-surface, μ = %s' % p['mu']),
        ('Материал', 'Мизес, кривая B, без повреждения'),
        ('Несоверш.', 't = 0.8·(1 − %g·cos(φ − %g°))' % (ecc, phi0)),
        ('Опоры', '4 пружины по касательной, %s Н/мм' % p.get('kspring', '1.0')),
        ('Нагружение', 'u_r до %s мм (ε_ном до %.2f)' % (p['ur'], float(p['ur']) / 5.5)),
    ]
    tab = ax4.table(cellText=[[k, v] for k, v in rows], colWidths=[0.3, 0.95], loc='upper left', cellLoc='left')
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.75)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#c9c8c0')
        if c == 0:
            cell.set_text_props(fontweight='bold')
    ax4.set_title('г) Параметры', loc='left', fontsize=11, color=INK)
    fig.suptitle('Модель всего кольца (до запуска): H = %s мм, %d сегментов, μ = %s' % (p['H'], nseg, p['mu']),
                 fontsize=14, color=INK, x=0.01, ha='left')
    fig.savefig(a.o, dpi=120, bbox_inches='tight', facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
