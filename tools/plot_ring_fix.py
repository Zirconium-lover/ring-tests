#!/usr/bin/env python3
"""Чем отличалось закрепление модели всего кольца: жёсткие опоры против пружин.

    python3 tools/plot_ring_fix.py -o notes/figs/ring_fix.png

а), б) схемы (разностенность показана с увеличением): было — две жёсткие
опоры по касательной (90° и 180°), стало — четыре мягкие пружины;
в) окружное смещение наружной поверхности в середине высоты u_θ(φ) за
вычетом поворота кольца как целого; г) наибольшая ε_θ на внутренней
поверхности по сегментам — при одинаковом ходе в обоих прогонах.
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post_stage1 import read_deck          # noqa: E402
from plot_maps_stage1 import frd_frames    # noqa: E402

ROOT = os.path.join(HERE, '..', 'runs', 'ring')
OLD, NEW = 'R12_H3_mu0.05_ecc1_pins', 'R12_H3_mu0.05_ecc1_spr'
INK, MUTED, BAD, GOOD = '#1f1f1e', '#6b6a63', '#d03b3b', '#2a78d6'
C_RING, C_SEG = '#e8b7bd', '#bcd0e6'
RI, RO, T0, ECC, PHI0, PHE = 5.5, 6.3, 0.8, 0.01, 3.0, 13.958


def frames(run):
    d = read_deck(os.path.join(ROOT, run, 'm.inp'))
    nodes = d['nodes']
    out, inn = sorted(set(d['sets']['OUT_Z0'])), sorted(set(d['sets']['IN_Z0']))
    ang = lambda n: math.degrees(math.atan2(nodes[n][1], nodes[n][0])) % 360
    fr = frd_frames(os.path.join(ROOT, run, 'm.frd'), set(out) | set(inn))
    res = {}
    for t, f in fr.items():
        U = f.get('DISP', {})
        if len(U) < len(out) + len(inn):
            continue
        ut = [(ang(n), -math.sin(math.radians(ang(n))) * U[n][0] + math.cos(math.radians(ang(n))) * U[n][1])
              for n in out]
        m = sum(u for _, u in ut) / len(ut)            # поворот кольца как целого
        ut = sorted((a, 1000 * (u - m)) for a, u in ut)
        order = sorted(inn, key=ang)
        pos = lambda n: [nodes[n][k] + U[n][k] for k in range(3)]
        eps = [((ang(a) + 0.5 * ((ang(b) - ang(a)) % 360)) % 360,
                math.log(math.dist(pos(a), pos(b)) / math.dist(nodes[a], nodes[b])))
               for a, b in zip(order, order[1:] + order[:1])]
        seg = [max(e for p, e in eps if min(abs(p - 30 * k), 360 - abs(p - 30 * k)) <= 15) for k in range(12)]
        res[round(t, 6)] = dict(ut=ut, seg=seg, en=t * 2.5 / 5.5)
    return res


def scheme(ax, springs):
    k = 25.0                                           # увеличение разностенности на схеме
    ph = [math.radians(x) for x in range(0, 361, 2)]
    ro = [RO - T0 * k * ECC * math.cos(p - math.radians(PHI0)) + T0 * 0 for p in ph]
    ring = [(RI * math.cos(p), RI * math.sin(p)) for p in ph] + \
           [(r * math.cos(p), r * math.sin(p)) for p, r in zip(ph[::-1], ro[::-1])]
    ax.add_patch(Polygon(ring, closed=True, fc=C_RING, ec=INK, lw=0.8))
    for s in range(12):
        c = 30 * s
        ax.add_patch(Wedge((0, 0), RI, c - PHE, c + PHE, width=0.9, fc=C_SEG, ec='#6f8fb3', lw=0.6))
    ax.text(RO + 0.25, -0.15, 'тоньше', fontsize=9, color=MUTED, ha='left', va='top')
    ax.text(-RO - 0.25, -0.75, 'толще', fontsize=9, color=MUTED, ha='right', va='top')
    # куда смещается материал: от тонкой стороны к толстой (по расчёту, см. в)
    for a, sgn in ((45, 1), (135, 1), (225, -1), (315, -1)):
        r = RO + 0.55
        a0, a1 = math.radians(a - 14 * sgn), math.radians(a + 14 * sgn)
        ax.add_patch(FancyArrowPatch((r * math.cos(a0), r * math.sin(a0)), (r * math.cos(a1), r * math.sin(a1)),
                                     connectionstyle='arc3,rad=%.2f' % (0.12 * sgn), arrowstyle='-|>',
                                     mutation_scale=14, color=GOOD, lw=1.8))
    if not springs:
        # жёсткие опоры по касательной: при 90° — u_x = 0, при 180° — u_y = 0
        for (x, y, horiz, lab, bad) in ((0, RO, True, 'жёсткая опора: u_x = 0\n(по касательной)', True),
                                        (-RO, 0, False, 'жёсткая опора:\nu_y = 0', False)):
            ax.plot(x, y, 's', color=BAD if bad else INK, ms=9, zorder=5)
            if horiz:
                for dx in (-0.35, 0.35):
                    ax.plot([x + dx, x + dx], [y - 0.3, y + 0.3], color=INK, lw=2.2)
                    for yy in (-0.3, -0.1, 0.1, 0.3):
                        ax.plot([x + dx, x + dx + 0.15 * (1 if dx > 0 else -1)], [yy + y, yy + y + 0.12], color=INK, lw=0.8)
                ax.text(x, y + 0.75, lab, ha='center', va='bottom', fontsize=9.5, color=BAD)
            else:
                for dy in (-0.35, 0.35):
                    ax.plot([x - 0.3, x + 0.3], [y + dy, y + dy], color=INK, lw=2.2)
                ax.text(x - 0.5, y + 0.7, lab, ha='right', va='bottom', fontsize=9.5, color=INK)
        ax.text(0, 2.2, 'здесь материал\nсмещается по окружности,\nа опора держит\nодну точку на месте', ha='center',
                va='center', fontsize=9.5, color=BAD)
        ax.set_title('а) Было: две жёсткие опоры', loc='left', fontsize=12, color=INK)
    else:
        for a in (0, 90, 180, 270):
            x, y = RO * math.cos(math.radians(a)), RO * math.sin(math.radians(a))
            tx, ty = -math.sin(math.radians(a)), math.cos(math.radians(a))
            nx, ny = math.cos(math.radians(a)), math.sin(math.radians(a))
            pts = [(x, y)]
            L, nz = 1.3, 6
            for i in range(1, nz + 1):
                s = L * i / (nz + 1)
                w = 0.18 * (1 if i % 2 else -1)
                pts.append((x + tx * s + nx * w, y + ty * s + ny * w))
            pts.append((x + tx * L, y + ty * L))
            ax.plot(*zip(*pts), color=GOOD, lw=1.4)
            gx, gy = x + tx * L, y + ty * L
            ax.plot([gx - nx * 0.3, gx + nx * 0.3], [gy - ny * 0.3, gy + ny * 0.3], color=INK, lw=2.2)
            ax.plot(x, y, 'o', color=GOOD, ms=6, zorder=5)
        ax.text(0, 2.2, 'четыре мягкие пружины\nпо касательной, 1 Н/мм:\nсила < 1 Н при окружной\nсиле в кольце ≈ 600 Н',
                ha='center', va='center', fontsize=9.5, color=GOOD)
        ax.set_title('б) Стало: четыре мягкие пружины', loc='left', fontsize=12, color=INK)
    ax.set_xlim(-8.6, 8.6); ax.set_ylim(-8.0, 8.6); ax.set_aspect('equal'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    old, new = frames(OLD), frames(NEW)
    common = sorted(set(old) & set(new))
    tc = common[-1]
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.12, wspace=0.12)
    scheme(fig.add_subplot(gs[0, 0]), springs=False)
    scheme(fig.add_subplot(gs[0, 1]), springs=True)

    ax = fig.add_subplot(gs[1, 0])
    x = [p for p, _ in new[tc]['ut']]
    ax.plot(x, [u for _, u in new[tc]['ut']], color=GOOD, lw=2.4, label='стало (пружины), ε_ном = %.3f' % new[tc]['en'])
    ax.plot(x, [u for _, u in old[tc]['ut']], color=BAD, lw=2.0, ls='--', label='было (опоры), ε_ном = %.3f' % old[tc]['en'])
    tl = max(old)
    ax.plot([p for p, _ in old[tl]['ut']], [u / (old[tl]['en'] / old[tc]['en']) for _, u in old[tl]['ut']],
            color=BAD, lw=1.0, alpha=0.6, label='было, ε_ном = %.3f (в масштабе %.3f)' % (old[tl]['en'], old[tc]['en']))
    for xx, lab in ((90, 'опора u_x = 0'), (180, 'опора u_y = 0')):
        ax.axvline(xx, color=MUTED, lw=0.8, ls=':')
        ax.text(xx + 2, ax.get_ylim()[0] if False else -7.5, lab, fontsize=8.5, color=MUTED, rotation=90, va='bottom')
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
    ax.set_xlabel('φ, град (0 — тонкая сторона, 180 — толстая)', color=INK)
    ax.set_ylabel('u_θ, мкм (+ — против часовой стрелки)', color=INK)
    ax.set_title('в) Сдвиг материала по окружности (наружная поверхность, середина высоты)', loc='left',
                 fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc='lower left')

    ax = fig.add_subplot(gs[1, 1])
    w = 0.38
    ks = range(12)
    ax.bar([k - w / 2 for k in ks], new[tc]['seg'], w, color=GOOD, label='стало (пружины)')
    ax.bar([k + w / 2 for k in ks], old[tc]['seg'], w, color=BAD, alpha=0.75, label='было (опоры)')
    lo = min(min(new[tc]['seg']), min(old[tc]['seg']))
    hi = max(max(new[tc]['seg']), max(old[tc]['seg']))
    ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.set_xticks(list(ks)); ax.set_xticklabels(['%d\n%d°' % (k, 30 * k) for k in ks], fontsize=8.5)
    ax.annotate('сегмент 3 у опоры', (3 + w / 2, old[tc]['seg'][3]), xytext=(4.6, hi + 0.18 * (hi - lo)),
                fontsize=9.5, color=BAD, arrowprops=dict(arrowstyle='->', color=BAD, lw=1))
    ax.set_xlabel('сегмент (номер и середина)', color=INK)
    ax.set_ylabel('наибольшая ε_θ на внутренней поверхности', color=INK)
    ax.set_title('г) Деформация по сегментам при ε_ном = %.3f' % new[tc]['en'], loc='left', fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    for axx in fig.axes[2:]:
        for s_ in ('top', 'right'):
            axx.spines[s_].set_visible(False)
    fig.suptitle('Модель всего кольца H = 3 мм, 12 сегментов: что было не так с закреплением', fontsize=14,
                 color=INK, x=0.01, ha='left', y=0.995)
    fig.savefig(a.o, dpi=110, bbox_inches='tight', facecolor='white')
    print('wrote', a.o, 'common t =', tc)


if __name__ == '__main__':
    main()
