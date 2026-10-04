#!/usr/bin/env python3
"""Картинка модели этапа 2: полный сегмент 45° × полная высота, сетка C3D4 x24.

    python3 tools/plot_stage2_mesh.py runs/stage2/decks/H8_mu0.07_full_uf0.01.inp -o notes/figs/stage2_model.png

а) общий вид кольца и сегмента с условиями закрепления; б) сечение z = 0 по
всей дуге; в) увеличение сетки у кромки сегмента; г) развёртка внутренней
поверхности (контакт, зазоры, торцы); д) параметры модели (из заголовка колоды).
"""
import argparse
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_mesh import read_deck, boundary_faces, cyl, C_RING, C_SEG

INK, MUTED = '#1f1f1e', '#6b6a63'


def params(path):
    head = open(path).readlines()[1]
    return dict(re.findall(r'(\w+)=(\[[^\]]*\]|\S+)', head))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('deck')
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    nodes, elems, nsets = read_deck(a.deck)
    p = params(a.deck)
    rf = boundary_faces(elems, 'RING')
    sf = boundary_faces(elems, 'SEGMENT')
    ring_nodes = {q for t, s, c in elems if s == 'RING' for q in c}
    nring = sum(1 for t, s, c in elems if s == 'RING')
    ri = min(cyl(nodes[q])[0] for q in ring_nodes)
    ro = max(cyl(nodes[q])[0] for q in ring_nodes)
    hz = max(nodes[q][2] for q in ring_nodes)
    seg_nodes = {q for t, s, c in elems if s == 'SEGMENT' for q in c}
    phe_deg = max(cyl(nodes[q])[1] for q in seg_nodes)          # cyl даёт градусы
    zseg = max(nodes[q][2] for q in seg_nodes)
    alpha_deg = max(cyl(nodes[q])[1] for q in ring_nodes)
    phe, alpha = math.radians(phe_deg), math.radians(alpha_deg)

    def section(eset, faces_of):
        """Грани элементов, целиком лежащие в плоскости z = 0 (внутреннее сечение)."""
        seen = {}
        for t, s_, c in elems:
            if s_ != eset:
                continue
            for f in faces_of(t):
                ff = tuple(c[i] for i in f)
                if all(abs(nodes[q][2]) < 1e-9 for q in ff):
                    seen[tuple(sorted(ff))] = ff
        return list(seen.values())
    TET_F = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
    HEX_F = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    z0 = section('RING', lambda t: TET_F if t == 'C3D4' else HEX_F)
    s0 = section('SEGMENT', lambda t: HEX_F)

    fig = plt.figure(figsize=(20, 12.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1, 1.08],
                          hspace=0.25, wspace=0.2)

    # --- а) 3D: наружная поверхность, торцы, боковые плоскости; сегмент ---
    ax = fig.add_subplot(gs[:, 0], projection='3d')
    def on_outer(f):
        rs = [cyl(nodes[q])[0] for q in f]
        return all(abs(r - ro) < 1e-6 for r in rs)
    def on_side(f):
        phs = [cyl(nodes[q])[1] for q in f]
        return all(abs(abs(x) - alpha_deg) < 1e-4 for x in phs) and abs(phs[0] - phs[1]) < 1e-4
    def on_end(f):
        return all(abs(abs(nodes[q][2]) - hz) < 1e-9 for q in f)
    show = [f for f in rf if on_outer(f) or on_side(f) or (on_end(f) and nodes[f[0]][2] > 0)]
    ax.add_collection3d(Poly3DCollection([[nodes[q] for q in f] for f in show], facecolor=C_RING,
                                         edgecolor='k', linewidths=0.04, alpha=0.97))
    ax.add_collection3d(Poly3DCollection([[nodes[q] for q in f] for f in sf], facecolor=C_SEG,
                                         edgecolor='#1d3d63', linewidths=0.03, alpha=0.45))
    ax.set_xlim(4.4, 6.6); ax.set_ylim(-2.6, 2.6); ax.set_zlim(-zseg, zseg)
    ax.set_box_aspect((2.2, 5.2, 2 * zseg))
    ax.view_init(elev=18, azim=-28)
    ax.set_xticks([4.5, 5.5, 6.5])
    ax.set_xlabel('x, мм'); ax.set_ylabel('y, мм'); ax.set_zlabel('z, мм')
    ax.set_title('а) Полный сегмент 45° × полная высота: кольцо (%d C3D4) и жёсткий сегмент'
                 % nring, fontsize=11, loc='left')
    fig.text(0.1, 0.14, 'φ = ±22.5° (середины зазоров): нормальное перемещение 0\n'
              'торцы z = ±H/2 свободны; осевой сдвиг держит один узел\n'
              'сегмент (синий): все узлы u_x = u_r(t), выступ 1 мм за торцы',
             fontsize=10, color=INK, va='top')

    # --- б) сечение z = 0 по всей дуге ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.add_collection(PolyCollection([[nodes[q][:2] for q in f] for f in z0], facecolor=C_RING,
                                      edgecolor='k', linewidths=0.08))
    ax2.add_collection(PolyCollection([[nodes[q][:2] for q in f] for f in s0], facecolor=C_SEG,
                                      edgecolor='#1d3d63', linewidths=0.1, alpha=0.7))
    for ang in (-alpha_deg, 0.0, alpha_deg):
        t = math.radians(ang)
        ax2.plot([4.6 * math.cos(t), 6.6 * math.cos(t)], [4.6 * math.sin(t), 6.6 * math.sin(t)],
                 'k--', lw=0.7)
    ax2.text(6.45, 0.05, 'φ = 0\n(середина\nсегмента)', fontsize=8)
    ax2.set_aspect('equal'); ax2.set_xlim(4.6, 6.7); ax2.set_ylim(-2.6, 2.6)
    ax2.set_title('б) Сечение z = 0: вся дуга (кольцо — красный, сегмент — синий)', fontsize=10.5,
                  loc='left')
    ax2.set_xlabel('x, мм'); ax2.set_ylabel('y, мм')
    # рамка увеличения
    ex, ey = ri * math.cos(phe), ri * math.sin(phe)
    ax2.add_patch(plt.Rectangle((ex - 0.35, ey - 0.35), 0.9, 0.75, fill=False, lw=1, ec='#eb6834'))

    # --- в) увеличение у кромки ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.add_collection(PolyCollection([[nodes[q][:2] for q in f] for f in z0], facecolor=C_RING,
                                      edgecolor='k', linewidths=0.35))
    ax3.add_collection(PolyCollection([[nodes[q][:2] for q in f] for f in s0], facecolor=C_SEG,
                                      edgecolor='#1d3d63', linewidths=0.3, alpha=0.7))
    ax3.set_xlim(ex - 0.35, ex + 0.55); ax3.set_ylim(ey - 0.35, ey + 0.4); ax3.set_aspect('equal')
    ax3.annotate('кромка сегмента φ = %.2f°' % phe_deg, xy=(ex, ey), xytext=(ex - 0.3, ey + 0.3),
                 fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.7))
    ax3.set_title('в) Сетка x24 у кромки: шаг 0.1 мм, 8 шестигранников по толщине,\n'
                  '24 тетраэдра в каждом', fontsize=10.5, loc='left')
    ax3.set_xlabel('x, мм'); ax3.set_ylabel('y, мм')

    # --- г) развёртка внутренней поверхности ---
    ax4 = fig.add_subplot(gs[1, 1])
    inner = [f for f in rf if all(abs(cyl(nodes[q])[0] - ri) < 1e-6 for q in f)]
    poly = [[(ri * math.radians(cyl(nodes[q])[1]), nodes[q][2]) for q in f] for f in inner]
    ax4.add_collection(PolyCollection(poly, facecolor='#e8b7bd', edgecolor='k', linewidths=0.05))
    se = ri * phe
    sa = ri * alpha
    ax4.axvspan(-se, se, color=C_SEG, alpha=0.18, lw=0)
    for x in (-se, se):
        ax4.axvline(x, color=C_SEG, lw=1.2, ls='--')
    ax4.text(0, hz + 0.25, 'контакт с сегментом (μ = %s)' % p.get('mu'), ha='center', fontsize=9,
             color='#1d3d63')
    ax4.text(sa, -hz - 0.55, 'полузазор\n0.1 мм', ha='right', fontsize=8, color=MUTED)
    ax4.set_xlim(-sa - 0.05, sa + 0.05); ax4.set_ylim(-hz - 0.9, hz + 0.6)
    ax4.set_aspect('equal')
    ax4.set_xlabel('дуга по R_i, мм (0 — середина сегмента)'); ax4.set_ylabel('z, мм')
    ax4.set_title('г) Внутренняя поверхность (развёртка)',
                  fontsize=10.5, loc='left')

    # --- д) параметры ---
    ax5 = fig.add_subplot(gs[1, 2]); ax5.axis('off')
    eps = p.get('epsf_eta', '[]').strip('[]').split(', ')
    pairs = ', '.join('%s→%s' % (eps[i], eps[i + 1]) for i in range(0, len(eps) - 1, 2))
    rows = [
        ('Кольцо', 'Ø %.1f / %.1f мм, H = %s мм' % (2 * ri, 2 * ro, p.get('H'))),
        ('Область', 'φ от −22.5° до +22.5° (1/8 кольца), z от −H/2 до +H/2'),
        ('Сетка кольца', '%d C3D4 (x24), %d узлов всего' % (nring, len(nodes))),
        ('Сегмент', 'жёсткий, φ ±%.2f°, выступ 1 мм за оба торца' % phe_deg),
        ('Контакт', 'node-to-surface, μ = %s' % p.get('mu')),
        ('Материал', 'Мизес, кривая B; E = 90.5 ГПа, ν = 0.35'),
        ('Повреждение', 'ε_f(η): ' + pairs),
        ('', 'u_f = %s мм, удаление C3D4' % p.get('uf')),
        ('Нагружение', 'u_r = 0 → %s мм (ε_ном до %.2f)' % (p.get('ur'), float(p.get('ur', 0)) / ri)),
        ('Остановка', 'нет пути нагрузки между φ = −22.5° и +22.5°'),
    ]
    tab = ax5.table(cellText=[[k, v] for k, v in rows], colWidths=[0.26, 0.9], loc='upper left',
                    cellLoc='left')
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.8)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#c9c8c0')
        if c == 0:
            cell.set_text_props(fontweight='bold')
    ax5.set_title('д) Параметры модели', fontsize=10.5, loc='left')
    fig.suptitle('Этап 2: модель для расчёта разрушения (до запуска)', fontsize=14, x=0.01, ha='left')
    fig.savefig(a.o, dpi=130, bbox_inches='tight', facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
