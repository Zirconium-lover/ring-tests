#!/usr/bin/env python3
"""Повреждение и удаление элементов в расчёте этапа 2 (по m.de1.vtk, m.de1stats, m.damage).

    python3 tools/plot_damage.py runs/stage2/test_coarse -o fig.png [--title "..."]

m.de1.vtk — последнее принятое состояние: деформированные узлы (точка k ↔
узел k+1 колоды), живые элементы и в них DE1_D (повреждение D) и DUCT_IP
(до зарождения — показатель ω = ∫dε_p/ε_f(η) < 1, после — 1 + D). Удалённые
элементы — в m.damage (номер, момент удаления).

а) сила на конусе F_z/H и отметки: зарождение повреждения, первое удаление;
б) рост повреждения: точек с D > 0.1 / 0.5 / 0.9 и удалённых элементов;
в) показатель ω / 1 + D на развёртке внутренней поверхности (вся дуга и
   высота), удалённые элементы — красным;
г) сечение z = 0 (деформированное) — то же, увеличение у кромки сегмента.
"""
import argparse
import csv
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post_stage1 import read_deck          # noqa: E402

INK, MUTED, ACC, DEL = '#1f1f1e', '#6b6a63', '#2a78d6', '#d03b3b'
PHI_EDGE = 21.458
TET_F = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))
# ω от 0 до 1 — синие оттенки; 1 + D от 1 до 2 — от жёлтого к тёмно-оранжевому
CMAP = LinearSegmentedColormap.from_list('omegaD', [(0.0, '#f4f7fb'), (0.25, '#86b6ef'), (0.4995, '#1c5cab'),
                                                    (0.5, '#f2c14e'), (0.75, '#eb6834'), (1.0, '#7a2a0c')])


def read_vtk(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('POINTS'))
    npts = int(lines[i].split()[1])
    pts = [tuple(map(float, l.split())) for l in lines[i + 1:i + 1 + npts]]
    i = next(k for k, l in enumerate(lines) if l.startswith('CELLS'))
    ncell = int(lines[i].split()[1])
    cells = [list(map(int, l.split()[1:])) for l in lines[i + 1:i + 1 + ncell]]
    data = {}
    for k, l in enumerate(lines):
        if l.startswith('SCALARS'):
            name = l.split()[1]
            data[name] = [float(x) for x in lines[k + 2:k + 2 + ncell]]
    head = lines[1]
    return pts, cells, data, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('-o', required=True)
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    run = a.run
    deck = read_deck(os.path.join(run, 'm.inp'))
    nodes, ri, ur_max = deck['nodes'], deck['ri'], deck['ur']
    pts, cells, data, head = read_vtk(os.path.join(run, 'm.de1.vtk'))
    eid = [int(x) for x in data['ELEMENT_ID']]
    val = dict(zip(eid, data['DUCT_IP']))           # ω < 1 или 1 + D
    dfm = lambda n: pts[n - 1]                       # деформированные координаты узла
    deleted = []
    if os.path.exists(os.path.join(run, 'm.damage')):
        for l in open(os.path.join(run, 'm.damage')):
            if l[:1].isdigit():
                f = l.split()
                deleted.append((int(f[0]), float(f[4])))
    dels = {e for e, _ in deleted}
    # кривые
    subprocess.run([sys.executable, os.path.join(HERE, 'post_stage1.py'), run], capture_output=True)
    ts = [{k: float(v) for k, v in r.items() if v != ''}
          for r in csv.DictReader(open(os.path.join(run, 'post', 'timeseries.csv')))]
    st = []
    if os.path.exists(os.path.join(run, 'm.de1stats')):
        st = [l.split() for l in open(os.path.join(run, 'm.de1stats')) if l[:1].isdigit()]
    en = lambda t: t * ur_max / ri
    t_init = float(st[0][3]) if st else None
    t_del = deleted[0][1] if deleted else None

    fig = plt.figure(figsize=(18, 12.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.28)
    top = outer[0].subgridspec(1, 2, wspace=0.22)
    bot = outer[1].subgridspec(1, 3, width_ratios=[0.6, 1.1, 1.0], wspace=0.25)

    ax = fig.add_subplot(top[0])
    ax.plot([r['eps_nom'] for r in ts], [r['Fz'] / deck['H'] for r in ts], color=ACC, lw=2.2)
    for t, lab in ((t_init, 'зарождение\nповреждения'), (t_del, 'первое\nудаление')):
        if t is not None:
            ax.axvline(en(t), color=MUTED, lw=0.9, ls='--')
            ax.text(en(t) - 0.004, 0.97, lab, fontsize=9, color=MUTED, ha='right', va='top',
                    transform=ax.get_xaxis_transform())
    ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
    ax.set_ylabel('сила на конусе / высота, Н/мм', color=INK)
    ax.set_title('а) Сила на конусе', loc='left', fontsize=11, color=INK)

    ax = fig.add_subplot(top[1])
    if st:
        x = [en(float(r[3])) for r in st]
        for col, lab, c in ((5, 'точек с D > 0 (зародилось)', '#86b6ef'), (6, 'D > 0.1', '#3987e5'),
                            (7, 'D > 0.5', '#1c5cab'), (8, 'D > 0.9', '#0d366b')):
            ax.plot(x, [int(r[col]) for r in st], color=c, lw=2, label=lab)
        xs = sorted({t for _, t in deleted})
        ax.step([en(t) for t in xs], [sum(1 for _, tt in deleted if tt <= t) for t in xs], where='post',
                color=DEL, lw=2.2, label='удалено элементов')
        ax.legend(frameon=False, fontsize=9, loc='upper left')
        ax.set_xlim(en(float(st[0][3])) - 0.01, en(float(st[-1][3])) + 0.005)
    ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
    ax.set_ylabel('число', color=INK)
    ax.set_title('б) Рост повреждения и удаление элементов', loc='left', fontsize=11, color=INK)

    norm = Normalize(0, 2)
    # в) развёртка внутренней поверхности: грани элементов на R_i (в исходной геометрии)
    ax = fig.add_subplot(bot[0])
    polys, cols, dpolys = [], [], []
    for e in deck['elems']:
        c = deck['elems'][e]
        if len(c) != 4:
            continue
        for f in TET_F:
            ff = [c[i] for i in f]
            if all(abs(math.hypot(nodes[q][0], nodes[q][1]) - ri) < 1e-6 for q in ff):
                poly = [(ri * math.atan2(nodes[q][1], nodes[q][0]), nodes[q][2]) for q in ff]
                if e in dels:
                    dpolys.append(poly)
                else:
                    polys.append(poly)
                    cols.append(val.get(e, 0.0))
    pc = PolyCollection(polys, array=cols, cmap=CMAP, norm=norm, edgecolor='none')
    ax.add_collection(pc)
    ax.add_collection(PolyCollection(dpolys, facecolor=DEL, edgecolor=DEL, linewidths=0.6))
    se = ri * math.radians(PHI_EDGE)
    for xx in (-se, se):
        ax.axvline(xx, color=MUTED, lw=0.8, ls='--')
    ax.set_xlim(-ri * math.radians(22.5), ri * math.radians(22.5)); ax.set_ylim(-0.5 * deck['H'], 0.5 * deck['H'])
    ax.set_aspect('equal')
    ax.set_xlabel('дуга по R_i, мм (пунктир — кромки)', color=INK); ax.set_ylabel('z, мм', color=INK)
    ax.set_title('в) Внутренняя поверхность\n(развёртка, исходная геометрия)', loc='left', fontsize=11,
                 color=INK)

    # г) сечение z = 0, деформированное — вся дуга и увеличение у кромки
    tris, tcol, dtris = [], [], []
    for e in deck['esets']['SEC_Z0']:
        c = deck['elems'][e]
        for f in TET_F:
            ff = [c[i] for i in f]
            if all(abs(nodes[q][2]) < 1e-9 for q in ff):
                poly = [dfm(q)[:2] for q in ff]
                if e in dels:
                    dtris.append(poly)
                else:
                    tris.append(poly)
                    tcol.append(val.get(e, 0.0))
    for k, (axspec, zoom) in enumerate(((bot[1], False), (bot[2], True))):
        ax = fig.add_subplot(axspec)
        ax.add_collection(PolyCollection(tris, array=tcol, cmap=CMAP, norm=norm, edgecolor='#ffffff',
                                         linewidths=0.15 if zoom else 0))
        ax.add_collection(PolyCollection(dtris, facecolor=DEL, edgecolor=DEL, linewidths=0.6))
        ur = en(float(head.split('time=')[1])) * ri if 'time=' in head else ur_max
        ph = [math.radians(-PHI_EDGE + 2 * PHI_EDGE * j / 120) for j in range(121)]
        ax.plot([ri * math.cos(p) + ur for p in ph], [ri * math.sin(p) for p in ph], color='#6b6a63', lw=1.2)
        xs = [p[0] for t in tris for p in t]; ys = [p[1] for t in tris for p in t]
        if zoom:
            # окно вокруг повреждения у одной кромки (y > 0): удалённые или наибольшее ω/1 + D
            cand = [t for t in dtris if sum(p[1] for p in t) > 0]
            if cand:
                cx = sum(p[0] for t in cand for p in t) / (3 * len(cand))
                cy = sum(p[1] for t in cand for p in t) / (3 * len(cand))
            else:
                j = max((i for i in range(len(tcol)) if sum(p[1] for p in tris[i]) > 0), key=lambda i: tcol[i])
                cx = sum(p[0] for p in tris[j]) / 3; cy = sum(p[1] for p in tris[j]) / 3
            ax.set_xlim(cx - 0.6, cx + 0.6); ax.set_ylim(cy - 0.5, cy + 0.5)
            ax.set_title('г) То же, увеличение у кромки: ω → 1 + D,\nудалённые элементы — красным',
                         loc='left', fontsize=11, color=INK)
        else:
            ax.set_xlim(min(xs) - 0.1, max(xs) + 0.1); ax.set_ylim(min(ys) - 0.1, max(ys) + 0.1)
            ax.set_title('г) Сечение z = 0, деформированное (серая дуга —\nповерхность сегмента)', loc='left',
                         fontsize=11, color=INK)
        ax.set_aspect('equal')
        ax.set_xlabel('x, мм', color=INK); ax.set_ylabel('y, мм', color=INK)
    cb = fig.colorbar(pc, ax=fig.axes[2:], shrink=0.75, pad=0.01)
    cb.set_label('ω = ∫dε_p/ε_f(η) (0…1, синий) → 1 + D (1…2, оранжевый)')
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    for axx in fig.axes:
        for s_ in ('top', 'right'):
            axx.spines[s_].set_visible(False)
    t_last = float(head.split('time=')[1]) if 'time=' in head else float('nan')
    fig.suptitle((a.title + ' — ' if a.title else '') + 'состояние при ε_ном = %.3f (u_r = %.2f мм): '
                 'удалено %d элементов, D_max = %.2f' % (en(t_last), en(t_last) * ri, len(dels),
                                                       max(data['DE1_D'])),
                 fontsize=13, color=INK, x=0.01, ha='left')
    fig.savefig(a.o, dpi=120, bbox_inches='tight', facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
