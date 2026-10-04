#!/usr/bin/env python3
"""Текущее состояние расчёта этапа 2 (можно запускать на идущем расчёте).

    python3 tools/plot_stage2_status.py runs/stage2/H8_mu0.07_full_uf0.01 -o fig.png

а) сила на конусе F_z/H от ε_ном — этап 2 против этапа 1 (C3D8I, H = 8 мм,
   μ = 0.05; сектор с симметрией);
б) наибольшая ε_θ на внутренней поверхности от ε_ном — то же сравнение;
в) карта ε_θ на развёртке внутренней поверхности (вся дуга и высота) в
   последнем записанном кадре .frd;
г) сечение z = 0 в последнем кадре, цвет — PEEQ;
е) ε_θ по дуге в середине высоты и на обоих торцах (последний кадр);
д) повреждение: число точек с D > 0.1 / 0.5 / 0.9 и удалённых элементов
   по m.de1stats и m.damage, плюс строка состояния (инкремент, время, память).
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post_stage1 import read_deck          # noqa: E402

INK, MUTED, ACC, REF = '#1f1f1e', '#6b6a63', '#2a78d6', '#9b9a92'
PHI_EDGE = 21.458
TET_F = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))


def last_frame(path, want_u, want_pe):
    """Последний полный кадр .frd: (время, U для want_u, PE для want_pe)."""
    cur_t, kind, block = None, None, None
    frames = {}
    with open(path) as f:
        for line in f:
            if line.startswith('  100CL'):
                cur_t = float(line[12:24])
            elif line.startswith(' -4'):
                kind = line.split()[1]
                block = frames.setdefault(cur_t, {}).setdefault(kind, {}) if kind in ('DISP', 'PE') else None
            elif block is not None and line.startswith(' -1'):
                n = int(line[3:13])
                if kind == 'DISP' and n in want_u:
                    block[n] = (float(line[13:25]), float(line[25:37]), float(line[37:49]))
                elif kind == 'PE' and n in want_pe:
                    block[n] = float(line[13:25])
            elif line.startswith(' -3'):
                block = None
    for t in sorted(frames, reverse=True):
        fr = frames[t]
        if len(fr.get('DISP', {})) == len(want_u) and len(fr.get('PE', {})) == len(want_pe):
            return t, fr['DISP'], fr['PE']
    return None, None, None


def status_line(run):
    sta = [l.split() for l in open(os.path.join(run, 'm.sta')) if l.strip()[:1].isdigit()]
    inc = int(sta[-1][1]) if sta else 0
    t = float(sta[-1][4]) if sta else 0.0
    prov = dict(l.strip().split('=', 1) for l in open(os.path.join(run, 'provenance.txt')) if '=' in l
                and not l.startswith('CCX_'))
    started = prov.get('started', '')
    rss = ''
    try:
        out = subprocess.run(['ps', '-o', 'rss=,etime=', '-C', 'ccx_2.23_pardis'], capture_output=True,
                             text=True).stdout.split()
        if out:
            rss = '%.1f ГБ, идёт %s' % (int(out[0]) / 1048576, out[1])
    except OSError:
        pass
    done = 'завершён, rc = %s' % prov['return_code'] if 'return_code' in prov else 'считается'
    return inc, t, '%s; инкремент %d, доля хода %.3f; %s' % (done, inc, t, rss), started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('-o', required=True)
    ap.add_argument('--ref', default=os.path.join(HERE, '..', 'runs', 'stage1', 'H8_mu0.05_C3D8I'))
    a = ap.parse_args()
    run = a.run
    # кривые: постобработка этапа 1 по m.dat (работает и на идущем расчёте)
    subprocess.run([sys.executable, os.path.join(HERE, 'post_stage1.py'), run], capture_output=True)
    subprocess.run([sys.executable, os.path.join(HERE, 'surfmap_stage1.py'), run], capture_output=True)
    ts = [{k: float(v) for k, v in r.items() if v != ''}
          for r in csv.DictReader(open(os.path.join(run, 'post', 'timeseries.csv')))]
    sm = json.load(open(os.path.join(run, 'post', 'surfmap.json')))
    rts = [{k: float(v) for k, v in r.items() if v != ''}
           for r in csv.DictReader(open(os.path.join(a.ref, 'post', 'timeseries.csv')))]
    rsm = json.load(open(os.path.join(a.ref, 'post', 'surfmap.json')))
    deck = read_deck(os.path.join(run, 'm.inp'))
    H = deck['H']
    nodes = deck['nodes']
    inner = set(deck['sets']['RING_INNER'])
    # сечение z = 0: грани тетраэдров слоя SEC_Z0, целиком в плоскости z = 0
    tris = {}
    for e in deck['esets']['SEC_Z0']:
        c = deck['elems'][e]
        for f in TET_F:
            ff = tuple(c[i] for i in f)
            if all(abs(nodes[q][2]) < 1e-9 for q in ff):
                tris[tuple(sorted(ff))] = ff
    tris = list(tris.values())
    sec_nodes = {q for f in tris for q in f}
    t_fr, U, PE = last_frame(os.path.join(run, 'm.frd'), inner | sec_nodes, sec_nodes)
    en_fr = t_fr * deck['ur'] / deck['ri'] if t_fr is not None else float('nan')
    inc, tfrac, status, started = status_line(run)

    fig = plt.figure(figsize=(18, 12))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.28)
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.28)
    bot = outer[1].subgridspec(1, 3, width_ratios=[0.62, 1.25, 1.0], wspace=0.3)

    # а) сила
    ax = fig.add_subplot(top[0])
    ax.plot([r['eps_nom'] for r in rts], [r['Fz'] / 8.0 for r in rts], color=REF, lw=1.6, ls='--',
            label='этап 1: C3D8I, сектор, μ = 0.05')
    ax.plot([r['eps_nom'] for r in ts], [r['Fz'] / H for r in ts], color=ACC, lw=2.4, marker='o', ms=3,
            label='этап 2: полный сегмент, x24, μ = 0.07')
    ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
    ax.set_ylabel('сила на конусе / высота, Н/мм', color=INK)
    ax.set_title('а) Сила на конусе', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.set_xlim(0, max(0.12, ts[-1]['eps_nom'] * 1.3) if ts else 0.12)

    # б) наибольшая ε_θ
    ax = fig.add_subplot(top[1])
    ax.plot([f['eps_nom'] for f in rsm['frames']], [max(f['emax']) for f in rsm['frames']], color=REF,
            lw=1.6, ls='--', label='этап 1 (C3D8I, μ = 0.05)')
    ax.plot([f['eps_nom'] for f in sm['frames']], [max(f['emax']) for f in sm['frames']], color=ACC,
            lw=2.4, marker='o', ms=3, label='этап 2')
    xx = [0, 0.6]
    ax.plot(xx, [math.log(1 + x) for x in xx], color=MUTED, lw=0.8, ls=':')
    ax.text(0.003, 0.0, 'пунктир — равномерная раздача', fontsize=8, color=MUTED, va='bottom')
    ax.set_xlabel('номинальная деформация u_r/R_i', color=INK)
    ax.set_ylabel('max ε_θ на внутренней поверхности', color=INK)
    ax.set_title('б) Наибольшая окружная деформация', loc='left', fontsize=11, color=INK)
    lim = max(0.12, sm['frames'][-1]['eps_nom'] * 1.3) if sm['frames'] else 0.12
    ax.set_xlim(0, lim)
    ax.set_ylim(0, max(0.15, lim * 1.6))
    ax.legend(frameon=False, fontsize=9, loc='upper left')

    # д) повреждение и состояние
    ax = fig.add_subplot(top[2]); ax.axis('off')
    stats = os.path.join(run, 'm.de1stats')
    dmg = [l.split() for l in open(stats)] if os.path.exists(stats) else []
    dmg = [r for r in dmg if r and r[0][0].isdigit()]
    ndel = sum(1 for l in open(os.path.join(run, 'm.damage')) if l[:1].isdigit()) \
        if os.path.exists(os.path.join(run, 'm.damage')) else 0
    lines = ['Состояние: %s' % status,
             'Запущен: %s' % started.replace('T', ' ').replace('Z', ' UTC'),
             'Последний кадр полей: ε_ном = %.3f (u_r = %.3f мм)' % (en_fr, en_fr * deck['ri']),
             '']
    if dmg:
        last = dmg[-1]
        lines += ['Повреждение (последний принятый инкремент):',
                  '  точек с D > 0: %s;  D > 0.1: %s;  D > 0.5: %s;  D > 0.9: %s' % tuple(last[5:9]),
                  '  D_max = %.3f' % float(last[10]),
                  '  удалено элементов: %d' % ndel]
    else:
        lines += ['Повреждения пока нет (m.de1stats пуст):',
                  '  по грубой проверке оно начнётся около ε_ном ≈ 0.35–0.40,',
                  '  удаление и сквозная трещина — дальше.']
    lines += ['', 'Модель: полный сегмент 45° × H = 8 мм, 675 840 C3D4 x24,',
              'μ = 0.07, ε_f(η) базовая, u_f = 0.01 мм, ход до u_r = 3.3 мм.']
    ax.text(0.0, 1.0, '\n'.join(lines), va='top', fontsize=10.5, color=INK, family='DejaVu Sans',
            linespacing=1.5)
    ax.set_title('д) Ход расчёта и повреждение', loc='left', fontsize=11, color=INK)

    # в) карта ε_θ на внутренней поверхности
    ax = fig.add_subplot(bot[0])
    prof = {}
    if U:
        rows = {}
        for n in inner:
            rows.setdefault(round(nodes[n][2], 6), []).append(n)
        zs = sorted(rows)
        grid = []
        for z in zs:
            ids = sorted(rows[z], key=lambda n: math.atan2(nodes[n][1], nodes[n][0]))
            sl, el = [], []
            for p_, q in zip(ids, ids[1:]):
                P, Q = nodes[p_], nodes[q]
                l0 = math.dist(P, Q)
                l = math.dist([P[k] + U[p_][k] for k in range(3)], [Q[k] + U[q][k] for k in range(3)])
                ph = 0.5 * (math.atan2(P[1], P[0]) + math.atan2(Q[1], Q[0]))
                sl.append(deck['ri'] * ph)
                el.append(math.log(l / l0))
            grid.append(el)
        # строки z с разным числом узлов (у x24 ряды через один реже) — берём полные
        full = max(len(g) for g in grid)
        zz = [z for z, g in zip(zs, grid) if len(g) == full]
        gg = [g for g in grid if len(g) == full]
        ids0 = sorted(rows[zz[0]], key=lambda n: math.atan2(nodes[n][1], nodes[n][0]))
        ss = [deck['ri'] * 0.5 * (math.atan2(nodes[p_][1], nodes[p_][0]) + math.atan2(nodes[q][1], nodes[q][0]))
              for p_, q in zip(ids0, ids0[1:])]
        cs = ax.pcolormesh(ss, zz, gg, cmap='Blues', shading='nearest')
        for lab, z_ in (('середина высоты z = 0', min(zz, key=abs)), ('торец z = +H/2', zz[-1]),
                        ('торец z = −H/2', zz[0])):
            prof[lab] = gg[zz.index(z_)]
        se = deck['ri'] * math.radians(PHI_EDGE)
        for x in (-se, se):
            ax.axvline(x, color='#eb6834', lw=1, ls='--')
        ax.text(se, zz[-1] + 0.15, 'кромка', color='#eb6834', fontsize=8, ha='right')
        ax.text(-se, zz[-1] + 0.15, 'кромка', color='#eb6834', fontsize=8, ha='left')
        i, j = max(((i, j) for i in range(len(zz)) for j in range(len(ss))), key=lambda ij: gg[ij[0]][ij[1]])
        ax.plot(ss[j], zz[i], 'o', mfc='none', mec='#eb6834', ms=10, mew=2)
        ax.annotate('max %.3f' % gg[i][j], xy=(ss[j], zz[i]), xytext=(-0.8, zz[i] + (1.2 if zz[i] < 0 else -1.2)),
                    fontsize=10, color=INK, arrowprops=dict(arrowstyle='->', lw=0.8))
        cb = fig.colorbar(cs, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label('ε_θ (лог.)')
        ax.set_aspect('equal')
        ax.set_ylim(zz[0] - 0.2, zz[-1] + 0.5)
    ax.set_xlabel('дуга по R_i, мм (0 — середина сегмента)', color=INK)
    ax.set_ylabel('z, мм', color=INK)
    ax.set_title('в) ε_θ на внутренней поверхности\n(развёртка), ε_ном = %.3f' % en_fr,
                 loc='left', fontsize=11, color=INK)

    # е) профили ε_θ по дуге: середина высоты и оба торца
    ax = fig.add_subplot(bot[1])
    if prof:
        for (lab, vals), c, ls in zip(prof.items(), (ACC, INK, MUTED), ('-', '--', ':')):
            ax.plot(ss, vals, color=c, lw=2 if ls == '-' else 1.5, ls=ls, label=lab)
        se = deck['ri'] * math.radians(PHI_EDGE)
        for x in (-se, se):
            ax.axvline(x, color='#eb6834', lw=1, ls='--')
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.legend(frameon=False, fontsize=9, loc='upper center')
    ax.set_xlabel('дуга по R_i, мм (0 — середина сегмента; пунктир — кромки)', color=INK)
    ax.set_ylabel('ε_θ (лог.), внутренняя поверхность', color=INK)
    ax.set_title('е) ε_θ по дуге: середина высоты и торцы', loc='left', fontsize=11, color=INK)

    # г) сечение z = 0 с PEEQ
    ax = fig.add_subplot(bot[2])
    if U:
        dfm = [[(nodes[q][0] + U[q][0], nodes[q][1] + U[q][1]) for q in f] for f in tris]
        val = [sum(PE[q] for q in f) / 3 for f in tris]
        pc = PolyCollection(dfm, array=val, cmap='Blues', edgecolor='none')
        ax.add_collection(pc)
        ur = en_fr * deck['ri']
        ph = [math.radians(-PHI_EDGE + 2 * PHI_EDGE * k / 80) for k in range(81)]
        ax.plot([deck['ri'] * math.cos(p) + ur for p in ph], [deck['ri'] * math.sin(p) for p in ph],
                color='#eb6834', lw=1.5)
        ax.set_aspect('equal')
        xs = [p[0] for f in dfm for p in f]; ys = [p[1] for f in dfm for p in f]
        ax.set_xlim(min(xs) - 0.1, max(xs) + 0.1); ax.set_ylim(min(ys) - 0.1, max(ys) + 0.1)
        cb = fig.colorbar(pc, ax=ax, pad=0.01, shrink=0.9)
        cb.set_label('PEEQ')
    ax.set_xlabel('x, мм', color=INK); ax.set_ylabel('y, мм', color=INK)
    ax.set_title('г) Сечение z = 0 (деформированное), PEEQ;\nоранжевый — поверхность сегмента', loc='left',
                 fontsize=11, color=INK)
    for axx in fig.axes:
        if hasattr(axx, 'spines'):
            for s_ in ('top', 'right'):
                axx.spines[s_].set_visible(False)
    fig.suptitle('Этап 2, H = 8 мм, μ = 0.07: текущее состояние расчёта (%s)'
                 % time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), fontsize=13, color=INK, x=0.01,
                 ha='left')
    fig.savefig(a.o, dpi=120, bbox_inches='tight', facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
