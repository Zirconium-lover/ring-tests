#!/usr/bin/env python3
"""Картинки сетки колоды этапа 1 (tools/mksector.py).

    python3 tools/plot_mesh.py runs/stage1/decks/H5_mu0.05.inp -o notes/figs/stage1_mesh_H5.png

Три панели: а) общий вид сектора с сегментом и условиями закрепления;
б) сечение z = 0 (плоскость симметрии) и увеличение у края сегмента;
в) развёртка внутренней поверхности (контакт и зазор, свободный торец).
"""
import argparse
import math
import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection, LineCollection

TET_F = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
HEX_F = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
C_RING, C_SEG = '#d1495b', '#5b8fc7'


def read_deck(path):
    nodes, elems, nsets, cur, etype, eset = {}, [], {}, None, None, None
    header = ''
    for line in open(path):
        s = line.strip()
        if s.startswith('**'):
            if not header:
                header = s
            continue
        if s.startswith('*'):
            kw = s.lower()
            cur = None
            if kw.startswith('*node') and not kw.startswith('*node print') \
                    and not kw.startswith('*node file'):
                cur = 'node'
            elif kw.startswith('*element'):
                cur = 'elem'
                etype = re.search(r'type=([a-z0-9]+)', kw).group(1).upper()
                eset = re.search(r'elset=([a-z0-9_]+)', kw).group(1).upper()
            elif kw.startswith('*nset'):
                cur = 'nset'
                eset = re.search(r'nset=([a-z0-9_]+)', kw).group(1).upper()
                nsets[eset] = []
            continue
        if not s:
            continue
        v = [x for x in s.split(',') if x.strip()]
        if cur == 'node':
            nodes[int(v[0])] = tuple(float(x) for x in v[1:4])
        elif cur == 'elem':
            elems.append((etype, eset, [int(x) for x in v[1:]]))
        elif cur == 'nset':
            nsets[eset] += [int(x) for x in v]
    return nodes, elems, nsets


def boundary_faces(elems, eset):
    cnt, keep = Counter(), {}
    for t, s, c in elems:
        if s != eset:
            continue
        faces = TET_F if t == 'C3D4' else HEX_F
        for f in faces:
            ff = tuple(c[i] for i in f)
            k = tuple(sorted(ff))
            cnt[k] += 1
            keep[k] = ff
    return [keep[k] for k, n in cnt.items() if n == 1]


def cyl(p):
    x, y, z = p
    return math.hypot(x, y), math.degrees(math.atan2(y, x)), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('deck')
    ap.add_argument('-o', required=True)
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    nodes, elems, nsets = read_deck(a.deck)
    rf = boundary_faces(elems, 'RING')
    sf = boundary_faces(elems, 'SEGMENT')
    zs = [p[2] for p in nodes.values()]
    ring_nodes = {n for t, s, c in elems if s == 'RING' for n in c}
    hz = max(nodes[n][2] for n in ring_nodes)          # H/2
    ri = min(cyl(nodes[n])[0] for n in ring_nodes)
    ro = max(cyl(nodes[n])[0] for n in ring_nodes)
    seg_nodes = {n for t, s, c in elems if s == 'SEGMENT' for n in c}
    phe = max(cyl(nodes[n])[1] for n in seg_nodes)
    zseg = max(nodes[n][2] for n in seg_nodes)
    nring = sum(1 for t, s, c in elems if s == 'RING')
    etype = next(t for t, s, c in elems if s == 'RING')

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1, 1],
                          hspace=0.22, wspace=0.22)

    # --- а) 3D ---
    ax = fig.add_subplot(gs[:, 0], projection='3d')
    ax.add_collection3d(Poly3DCollection([[nodes[n] for n in f] for f in rf],
                                         facecolor=C_RING, edgecolor='k',
                                         linewidths=0.08, alpha=0.95))
    ax.add_collection3d(Poly3DCollection([[nodes[n] for n in f] for f in sf],
                                         facecolor=C_SEG, edgecolor='#1d3d63',
                                         linewidths=0.05, alpha=0.55))
    ax.set_xlim(4.6, 6.6); ax.set_ylim(-0.2, 2.6); ax.set_zlim(0, max(zseg, 2.0))
    ax.set_box_aspect((2.0, 2.8, max(zseg, 2.0)))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel('x, мм'); ax.set_ylabel('y, мм'); ax.set_zlabel('z, мм')
    ax.set_title('а) Сектор 22.5° × H/2: кольцо (%d %s) и жёсткий сегмент'
                 % (nring, etype), fontsize=11)
    txt = [((6.3, 0.0, 0.2), 'φ = 0: u_y = 0'),
           ((6.3 * math.cos(math.radians(22.5)) - 0.9, 6.3 * math.sin(math.radians(22.5)), hz + 0.35),
            'φ = 22.5°: u_n = 0 (уравнение)'),
           ((5.9, 1.2, -0.05), 'z = 0: u_z = 0'),
           ((5.9, 1.2, hz + 0.05), 'торец свободен'),
           ((5.0, 0.0, zseg), 'сегмент: u_x = u_r(t)')]
    for (x, y, z), s in txt:
        ax.text(x, y, z, s, fontsize=9)

    # --- б) сечение z = 0 ---
    ax2 = fig.add_subplot(gs[0, 1])
    tri0 = [f for f in rf if all(abs(nodes[n][2]) < 1e-9 for n in f)]
    seg0 = [f for f in sf if all(abs(nodes[n][2]) < 1e-9 for n in f)]
    ax2.add_collection(PolyCollection([[nodes[n][:2] for n in f] for f in tri0],
                                      facecolor=C_RING, edgecolor='k', linewidths=0.25))
    ax2.add_collection(PolyCollection([[nodes[n][:2] for n in f] for f in seg0],
                                      facecolor=C_SEG, edgecolor='#1d3d63', linewidths=0.2,
                                      alpha=0.7))
    for ang in (0.0, 22.5):
        t = math.radians(ang)
        ax2.plot([4.8 * math.cos(t), 6.5 * math.cos(t)], [4.8 * math.sin(t), 6.5 * math.sin(t)],
                 'k--', lw=0.8)
    ax2.set_aspect('equal'); ax2.set_xlim(4.85, 6.5); ax2.set_ylim(-0.1, 2.6)
    ax2.set_title('б) Сечение z = 0 (плоскость симметрии)', fontsize=11)
    ax2.set_xlabel('x, мм'); ax2.set_ylabel('y, мм')
    # --- б') увеличение у края сегмента ---
    ins = fig.add_subplot(gs[0, 2])
    ins.add_collection(PolyCollection([[nodes[n][:2] for n in f] for f in tri0],
                                      facecolor=C_RING, edgecolor='k', linewidths=0.5))
    ins.add_collection(PolyCollection([[nodes[n][:2] for n in f] for f in seg0],
                                      facecolor=C_SEG, edgecolor='#1d3d63', linewidths=0.4,
                                      alpha=0.7))
    t = math.radians(phe)
    cx, cy = 5.75 * math.cos(t), 5.75 * math.sin(t)
    ins.set_xlim(cx - 0.55, cx + 0.55); ins.set_ylim(cy - 0.45, cy + 0.65)
    ins.set_aspect('equal')
    t2 = math.radians(22.5)
    ins.plot([5.0 * math.cos(t2), 6.5 * math.cos(t2)], [5.0 * math.sin(t2), 6.5 * math.sin(t2)],
             'k--', lw=0.8)
    ins.annotate('край сегмента φ_e = %.2f°' % phe, xy=(ri * math.cos(t), ri * math.sin(t)),
                 xytext=(cx - 0.5, cy + 0.5), fontsize=9,
                 arrowprops=dict(arrowstyle='->', lw=0.8))
    ins.text(6.3 * math.cos(t2) - 0.02, 6.3 * math.sin(t2) + 0.05, 'φ = 22.5°\n(середина зазора)',
             fontsize=8, ha='right')
    ins.set_xlabel('x, мм'); ins.set_ylabel('y, мм')
    ins.set_title("б') Увеличение: 8 элементов по толщине", fontsize=11)

    # --- в) развёртка внутренней поверхности ---
    ax3 = fig.add_subplot(gs[1, 1])
    inner = set(nsets['RING_INNER'])
    fin = [f for f in rf if all(n in inner for n in f)]
    poly = [[(math.radians(cyl(nodes[n])[1]) * ri, nodes[n][2]) for n in f] for f in fin]
    ax3.add_collection(PolyCollection(poly, facecolor='#f6d5da', edgecolor='k',
                                      linewidths=0.15))
    sarc = math.radians(phe) * ri
    ax3.add_patch(plt.Rectangle((0, 0), sarc, zseg, facecolor=C_SEG, alpha=0.25,
                                edgecolor=C_SEG))
    ax3.axhline(hz, color=C_RING, lw=1.2)
    ax3.text(0.05, hz + 0.08, 'свободный торец z = H/2 = %.2f мм' % hz, fontsize=9)
    ax3.text(0.05, zseg - 0.25, 'сегмент (выступает на %.1f мм)' % (zseg - hz), fontsize=9,
             color='#1d3d63')
    ax3.axvline(sarc, color=C_SEG, lw=1.0, ls='--')
    ax3.annotate('край сегмента', xy=(sarc, 1.0), xytext=(sarc - 0.9, 1.25), fontsize=8,
                 arrowprops=dict(arrowstyle='->', lw=0.7))
    ax3.annotate('полупропил\n0.1 мм', xy=(0.5 * (sarc + math.radians(22.5) * ri), 0.6),
                 xytext=(sarc - 0.9, 0.45), fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.7))
    ax3.set_xlim(0, math.radians(22.5) * ri + 0.02); ax3.set_ylim(0, zseg + 0.1)
    ax3.set_aspect('equal')
    ax3.set_xlabel('дуга по R_i = %.2f мм, мм' % ri); ax3.set_ylabel('z, мм')
    ax3.set_title('в) Развёртка внутренней поверхности\n(узлы — ведомые узлы контакта)', fontsize=11)

    axt = fig.add_subplot(gs[1, 2]); axt.axis('off')
    rows = [('H, мм', 'H/t', 'C3D4', 'узлов'),
            ('3', '3.75', '15 840', '4 236'), ('5', '6.25', '26 400', '6 642'),
            ('8', '10.0', '42 240', '10 251'), ('12', '15.0', '63 360', '15 063')]
    tb = axt.table(cellText=rows[1:], colLabels=rows[0], loc='upper center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1.0, 1.6)
    axt.set_title('г) Серия по высоте (сектор 22.5° × H/2,\nшаг 0.1 мм: 8 по толщине × 22 по дуге)',
                  fontsize=11)
    axt.text(0.0, 0.25, 'Каждая высота × μ = 0, 0.05, 0.2 → 12 расчётов.\n'
             'Контроль запирания: H = 5 мм, μ = 0.05 на C3D8I\n(4 400 шестигранников той же сетки).',
             fontsize=10, transform=axt.transAxes, va='top')

    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.savefig(a.o, dpi=140, bbox_inches='tight')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
