#!/usr/bin/env python3
"""Рисунки отчёта о гармоническом анализе деформации кольца (notes/report_harmonics.md).

    python3 tools/plot_harmonics.py [-o notes/figs] [--only curve,build,...]

Кривая — окружная деформация ε_θ (лог.) по хорде между соседними узлами
в середине высоты, на внутренней (или наружной) поверхности, как функция
угла φ по всей окружности. Гармоника n — составляющая с n волнами на
оборот: A_n·cos(n·φ − фаза). A_0 — среднее; A_n = 2|c_n|, c_n — коэффициенты
дискретного преобразования Фурье, делённые на число точек.

Данные: модель всего кольца runs/ring/R12_H3_mu0.05_ecc1_spr (12 сегментов,
H = 3 мм, μ = 0.05, разностенность ±1 %); секторы этапа 1 и runs/nseg
(8 и 12 сегментов, μ = 0…0.2) — профиль сектора зеркалится и повторяется
по окружности; сжатие кольца — аналитический изгибающий момент тонкого кольца.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post_stage1 import read_deck          # noqa: E402
from plot_maps_stage1 import frd_frames    # noqa: E402

RUNS = os.path.normpath(os.path.join(HERE, '..', 'runs'))
RING = os.path.join(RUNS, 'ring', 'R12_H3_mu0.05_ecc1_spr')
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
C0, C1, C12, CSB, CREST = '#6b6a63', '#eb6834', '#2a78d6', '#1baf7a', '#eda100'
C_RING, C_SEG = '#e8b7bd', '#bcd0e6'
RI, RO, PHE12 = 5.5, 6.3, 13.958
UR_RING = 2.5


def style(ax, grid=True):
    if grid:
        ax.grid(color=GRID, lw=0.7)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def spectrum(x):
    c = np.fft.rfft(np.asarray(x)) / len(x)
    a = 2 * np.abs(c)
    a[0] /= 2
    return a, np.angle(c)


def component(x, ns):
    """Сумма гармоник ns (список номеров) сигнала x на той же сетке."""
    c = np.fft.rfft(np.asarray(x))
    keep = np.zeros_like(c)
    for n in ns:
        keep[n] = c[n]
    return np.fft.irfft(keep, len(x))


# ---------- данные модели всего кольца (кэш) ----------
def ring_frames():
    cache = os.path.join(RING, 'post', 'harm_profiles.json')
    frd = os.path.join(RING, 'm.frd')
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(frd):
        return json.load(open(cache))
    d = read_deck(os.path.join(RING, 'm.inp'))
    nodes = d['nodes']
    inn, out = sorted(set(d['sets']['IN_Z0'])), sorted(set(d['sets']['OUT_Z0']))
    ang = lambda n: math.degrees(math.atan2(nodes[n][1], nodes[n][0])) % 360
    fr = frd_frames(frd, set(inn) | set(out))
    res = []
    for t in sorted(fr):
        U = fr[t].get('DISP', {})
        if len(U) < len(inn) + len(out):
            continue
        pos = lambda n: [nodes[n][k] + U[n][k] for k in range(3)]
        row = dict(t=t, en=t * UR_RING / RI)
        for name, line in (('in', inn), ('out', out)):
            order = sorted(line, key=ang)
            row['phi'] = [(ang(a) + 0.5 * ((ang(b) - ang(a)) % 360)) % 360 for a, b in zip(order, order[1:] + order[:1])]
            row[name] = [math.log(math.dist(pos(a), pos(b)) / math.dist(nodes[a], nodes[b]))
                         for a, b in zip(order, order[1:] + order[:1])]
        res.append(row)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(res, open(cache, 'w'))
    return res


def at(frames, x):
    return min(frames, key=lambda f: abs(f['en'] - x))


# ---------- секторы: профиль в середине высоты, зеркало и повтор ----------
def sector_profile(run, x, surf='out'):
    d = read_deck(os.path.join(run, 'm.inp'))
    nodes, sector = d['nodes'], d['sector']
    line = d['sets']['IN_Z0' if surf == 'in' else 'OUT_Z0']
    ang = lambda n: math.degrees(math.atan2(nodes[n][1], nodes[n][0]))
    order = sorted(set(line), key=ang)
    fr = frd_frames(os.path.join(run, 'm.frd'), set(order))
    ts = [t for t in fr if len(fr[t].get('DISP', {})) == len(order)]
    t = min(ts, key=lambda t: abs(t * d['ur'] / d['ri'] - x))
    U = fr[t]['DISP']
    pos = lambda n: [nodes[n][k] + U[n][k] for k in range(3)]
    half = [math.log(math.dist(pos(a), pos(b)) / math.dist(nodes[a], nodes[b])) for a, b in zip(order, order[1:])]
    period = half + half[::-1]
    nseg = round(180 / sector)
    return np.array(period * nseg), nseg, t * d['ur'] / d['ri']


# ---------- рисунки ----------
def fig_curve(fr, out):
    f = at(fr, 0.09)
    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.75, 1.6], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ph = np.radians(np.arange(0, 361, 2))
    k = 25
    ro = RO - 0.8 * k * 0.01 * np.cos(ph - math.radians(3))
    ax.add_patch(Polygon(list(zip(RI * np.cos(ph), RI * np.sin(ph))) + list(zip(ro[::-1] * np.cos(ph[::-1]),
                                                                                  ro[::-1] * np.sin(ph[::-1]))),
                         closed=True, fc=C_RING, ec=INK, lw=0.8))
    for s in range(12):
        ax.add_patch(Wedge((0, 0), RI, 30 * s - PHE12, 30 * s + PHE12, width=0.9, fc=C_SEG, ec='#6f8fb3', lw=0.6))
    ax.plot(RI * np.cos(ph), RI * np.sin(ph), color=C1, lw=3)
    ax.annotate('кривая снимается здесь:\nвнутренняя поверхность,\nсередина высоты, весь круг', (RI * math.cos(3.6), RI * math.sin(3.6)),
                xytext=(-9.0, -8.6), fontsize=9.5, color=C1, arrowprops=dict(arrowstyle='->', color=C1, lw=1))
    for a, lab in ((0, '0°\nстенка\nтоньше'), (90, '90°'), (180, '180°\nстенка\nтолще'), (270, '270°')):
        ax.text(7.6 * math.cos(math.radians(a)), 7.3 * math.sin(math.radians(a)), lab, ha='center', va='center',
                fontsize=8.5, color=MUTED)
    ax.text(0, 0, '12 сегментов\n(голубые)\nраздвигаются\nконусом', ha='center', va='center', fontsize=9.5, color='#41658f')
    ax.set_xlim(-9.5, 9.5); ax.set_ylim(-10.2, 8.6); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('а) Где берём кривую\n(разностенность ±1 % показана с увеличением)', loc='left', fontsize=11, color=INK)

    ax = fig.add_subplot(gs[1])
    phi, e = np.array(f['phi']), np.array(f['in'])
    for s in range(12):
        for c in (30 * s, 30 * s + 360):
            ax.axvspan(c - PHE12, c + PHE12, color=C_SEG, alpha=0.35, lw=0)
    o = np.argsort(phi)
    ax.plot(phi[o], e[o], color=INK, lw=1.6)
    ax.axhline(e.mean(), color=C0, lw=1.2, ls='--')
    ax.text(362, e.mean(), 'среднее\n%.3f' % e.mean(), fontsize=9, color=C0, va='center')
    j = int(np.argmax(e))
    ax.annotate('пики у кромок сегментов (между парой пиков — зазор)', (phi[j], e[j]), xytext=(60, e.max() + 0.004), fontsize=9.5,
                arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.annotate('провалы под серединами сегментов', (phi[o][np.argmin(e[o][150:200]) + 150], e[o][150:200].min()),
                xytext=(195, e.min() - 0.006), fontsize=9.5, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.text(5, e.max() + 0.009, 'тонкая сторона: всё выше', fontsize=9.5, color=C1)
    ax.text(150, e.max() + 0.009, 'толстая сторона: всё ниже', fontsize=9.5, color=C1)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
    ax.set_ylim(e.min() - 0.009, e.max() + 0.013)
    ax.set_xlabel('φ — угол по окружности, град (голубые полосы — где стоят сегменты)', color=INK)
    ax.set_ylabel('окружная деформация ε_θ', color=INK)
    ax.set_title('б) Сама кривая: деформация по всему кругу при ε_ном = %.3f (у максимума силы)' % f['en'],
                 loc='left', fontsize=11, color=INK)
    style(ax)
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')


def ring_icon(ax, comp, color, scale, title):
    phi = np.linspace(0, 2 * np.pi, len(comp), endpoint=False)
    r = 1 + scale * comp
    ax.fill(np.append(np.cos(phi), 1) * 1.0, np.append(np.sin(phi), 0) * 1.0, color='#f1f0eb', zorder=0)
    ax.plot(np.append(r * np.cos(phi), r[0]), np.append(r * np.sin(phi), 0), color=color, lw=1.8)
    ax.plot(np.cos(phi), np.sin(phi), color=MUTED, lw=0.6, ls=':')
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=8.5, color=MUTED)


def fig_build(fr, out):
    f = at(fr, 0.09)
    phi = np.array(f['phi']); o = np.argsort(phi); phi = phi[o]
    e = np.array(f['in'])[o]
    a, _ = spectrum(e)
    rows = [
        ('настоящая кривая', e, INK, None, ''),
        ('n = 0: среднее — равномерная раздача', component(e, [0]), C0, None, 'одна и та же раздача по всему кругу'),
        ('n = 1: одна волна на оборот — разностенность', component(e, [1]), C1, 1,
         'тонкая сторона растянута сильнее толстой'),
        ('n = 12: двенадцать волн — сегменты', component(e, [12]), C12, 12, 'у кромок больше, под серединой меньше'),
        ('n = 11 и 13: боковые полосы — сегменты на тонкой стороне сильнее', component(e, [11, 13]), CSB, 11,
         'взаимодействие разностенности и сегментов'),
        ('n = 24, 36, 48, …: острота пиков у кромок', component(e, list(range(24, len(a), 12))), CREST, 24,
         'пики острые, а не синусоида'),
    ]
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(len(rows) + 1, 2, width_ratios=[6, 1], hspace=0.55, wspace=0.02, top=0.95)
    for i, (lab, y, col, n, note) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(phi, y, color=col, lw=1.8)
        if i == 0:
            ax.axhline(e.mean(), color=C0, lw=0.8, ls='--')
        else:
            ax.axhline(0 if i > 1 else e.mean(), color=MUTED, lw=0.5)
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
        if i < len(rows):
            ax.set_xticklabels([])
        amp = (y.max() - y.min()) / 2 if i > 1 else (a[0] if i == 1 else None)
        ttl = lab + ('' if i < 2 else '   (размах ±%.4f)' % amp)
        ax.set_title(ttl, loc='left', fontsize=10.5, color=col if i else INK)
        if note:
            ax.text(1.0, 1.02, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=MUTED)
        style(ax)
        if i >= 1:
            axi = fig.add_subplot(gs[i, 1])
            comp = y - (e.mean() if i == 1 else 0)
            sc = 0 if i == 1 else 0.35 / max(abs(comp).max(), 1e-9)
            ring_icon(axi, comp if i > 1 else np.zeros_like(comp), col, sc,
                      'кольцо, искажённое\nэтой составляющей' if i == 2 else '')
    ax = fig.add_subplot(gs[len(rows), 0])
    rec = component(e, [0, 1, 11, 12, 13] + list(range(24, len(a), 12)))
    ax.plot(phi, e, color=INK, lw=2.6, label='настоящая кривая')
    ax.plot(phi, rec, color=C12, lw=1.3, ls='--', label='сумма составляющих выше')
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
    ax.set_xlabel('φ, град', color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.62))
    ax.set_title('Сложили составляющие — получили исходную кривую (остаток %.1f %% размаха)'
                 % (100 * np.abs(e - rec).max() / (e.max() - e.min())), loc='left', fontsize=10.5, color=INK)
    style(ax)
    fig.suptitle('Кривая при ε_ном = %.3f, разложенная на простые волны' % f['en'], fontsize=13.5, color=INK,
                 x=0.01, ha='left', y=0.985)
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white')


def fig_spectrum(fr, out):
    from matplotlib.patches import Patch
    f = at(fr, 0.09)
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw=dict(wspace=0.15))
    for ax, surf, ttl in ((axs[0], 'in', 'а) внутренняя поверхность'), (axs[1], 'out', 'б) наружная поверхность')):
        a, _ = spectrum(f[surf])
        n = np.arange(len(a))
        col = lambda k: (C0 if k == 0 else C1 if k in (1, 2) else C12 if k == 12 else CSB if k in (11, 13)
                         else CREST if k % 12 == 0 else '#c9c8c0')
        lim = 100
        ax.bar(n[:lim], a[:lim], color=[col(k) for k in n[:lim]], width=0.8)
        ax.set_yscale('log'); ax.set_ylim(1e-6, 0.4)
        ax.set_xlim(-1, lim)
        ax.annotate('n = 2 (овальность) ≈ 0:\nв модели её нет', (2, max(a[2], 1.2e-6)), xytext=(18, 1.6e-6),
                    fontsize=8.5, color=C1, arrowprops=dict(arrowstyle='->', color=C1, lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.95))
        ax.set_xlabel('n — число волн на оборот', color=INK)
        ax.set_ylabel('амплитуда, лог. шкала', color=INK)
        ax.set_title('%s, ε_ном = %.3f' % (ttl, f['en']), loc='left', fontsize=11, color=INK)
        style(ax)
    axs[1].legend(handles=[Patch(color=C0, label='n = 0 — среднее (равномерная раздача)'),
                           Patch(color=C1, label='n = 1 — разностенность'),
                           Patch(color=C12, label='n = 12 — сегменты'),
                           Patch(color=CSB, label='n = 11, 13 — боковые полосы'),
                           Patch(color=CREST, label='n = 24, 36, … — острота пиков у кромок'),
                           Patch(color='#c9c8c0', label='прочие — мелочь')],
                  frameon=True, framealpha=0.95, edgecolor='#e4e3dc', fontsize=8.5, loc='upper right')
    fig.suptitle('Спектр: сколько «весит» каждая волна (столбик — амплитуда гармоники n)', fontsize=13, color=INK,
                 x=0.01, ha='left', y=1.02)
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')


def fig_modulation(fr, out):
    phi = np.linspace(0, 360, 720, endpoint=False)
    p = np.radians(phi)
    plain = np.cos(12 * p)
    mod = (1 + 0.5 * np.cos(p)) * np.cos(12 * p)
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.2, 1], hspace=0.6, wspace=0.18)
    f = at(fr, 0.16)
    e = np.array(f['in'])[np.argsort(f['phi'])]
    real = component(e, [11, 12, 13])
    data = [(plain, 'а) двенадцать одинаковых волн', 'одна линия: n = 12'),
            (mod, 'б) те же волны, но справа (около 0°) выше, слева ниже', 'n = 12 и две боковые: 11 и 13'),
            (real, 'в) наша модель: составляющие 11 + 12 + 13 при ε_ном = %.3f' % f['en'], 'то же самое в расчёте')]
    for i, (y, ttl, note) in enumerate(data):
        ax = fig.add_subplot(gs[i, 0])
        x = phi if i < 2 else np.sort(f['phi'])
        ax.plot(x, y, color=C12 if i == 0 else CSB, lw=1.4)
        if i == 1:
            ax.plot(phi, 1 + 0.5 * np.cos(p), color=C1, lw=1, ls='--')
            ax.plot(phi, -(1 + 0.5 * np.cos(p)), color=C1, lw=1, ls='--')
            ax.text(185, 1.25, 'огибающая — одна волна на оборот (n = 1)', fontsize=9, color=C1)
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
        ax.set_title(ttl, loc='left', fontsize=10.5, color=INK)
        style(ax)
        axs = fig.add_subplot(gs[i, 1])
        a, _ = spectrum(y)
        axs.bar(range(8, 17), a[8:17], color=[CSB if k in (11, 13) else C12 if k == 12 else '#c9c8c0' for k in range(8, 17)])
        axs.set_xticks(range(8, 17)); axs.set_title(note, fontsize=9, color=MUTED)
        axs.set_xlabel('n', color=INK)
        style(axs)
    fig.suptitle('Почему в спектре рядом с n = 12 стоят n = 11 и 13: это «модуляция» — волны разной высоты',
                 fontsize=13, color=INK, x=0.01, ha='left', y=0.995)
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')


def fig_growth(fr, out):
    en = [f['en'] for f in fr]
    A = {k: [] for k in ('a0', 'a1', 'a12s', 'sb', 'hi', 'phimax', 'smax', 'emax')}
    for f in fr:
        e = np.array(f['in'])
        c = np.fft.rfft(e) / len(e)
        a = 2 * np.abs(c)
        A['a0'].append(c[0].real); A['a1'].append(a[1])
        A['a12s'].append(2 * c[12].real)              # знак: + больше под серединами сегментов, − у кромок
        A['sb'].append(math.hypot(a[11], a[13]))
        A['hi'].append(math.sqrt(sum(a[k] ** 2 for k in range(24, len(a), 12))))
        jm = int(np.argmax(e)); ph = f['phi'][jm]
        A['phimax'].append(ph); A['smax'].append(((ph + 15) % 30) - 15); A['emax'].append(e[jm])
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw=dict(wspace=0.25, width_ratios=[1.1, 1.1, 1]))
    ax = axs[0]
    ax.plot(en, A['a1'], color=C1, lw=2.2, marker='o', ms=3, label='n = 1 (разностенность)')
    ax.plot(en, A['a12s'], color=C12, lw=2.2, marker='o', ms=3, label='n = 12 со знаком (сегменты)')
    ax.plot(en, A['sb'], color=CSB, lw=2.2, marker='o', ms=3, label='n = 11 и 13 (боковые полосы)')
    ax.plot(en, A['hi'], color=CREST, lw=2.2, marker='o', ms=3, label='n = 24, 36, … (острота пиков)')
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_ylim(min(A['a12s']) - 0.004, max(max(A['a12s']), max(A['a1'])) * 1.12)
    ax.text(0.10, min(A['a12s']) - 0.0028, 'n = 12 < 0: больше у кромок', fontsize=8.5, color=C12)
    ax.text(0.012, max(A['a12s']) * 0.85, 'n = 12 > 0: больше под серединами', fontsize=8.5, color=C12)
    ax.set_ylabel('амплитуда', color=INK)
    ax.set_title('а) Амплитуды гармоник', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='center left')
    ax = axs[1]
    ax.plot(en, [100 * v / m for v, m in zip(A['a1'], A['a0'])], color=C1, lw=2.2, marker='o', ms=3, label='n = 1')
    ax.plot(en, [100 * v / m for v, m in zip(A['a12s'], A['a0'])], color=C12, lw=2.2, marker='o', ms=3, label='n = 12 со знаком')
    ax.plot(en, [100 * v / m for v, m in zip(A['sb'], A['a0'])], color=CSB, lw=2.2, marker='o', ms=3, label='n = 11 и 13')
    ax.axhline(0, color=MUTED, lw=0.7); ax.axhline(100, color=MUTED, lw=0.5, ls=':')
    ax.set_ylabel('% от средней деформации (n = 0)', color=INK)
    ax.set_title('б) Относительно средней деформации', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    ax = axs[2]
    ax.plot(en, [abs(v) for v in A['smax']], color=INK, lw=2, marker='o', ms=4)
    ax.axhline(PHE12, color=C12, lw=0.9, ls='--')
    ax.axhline(15, color=MUTED, lw=0.9, ls=':')
    ax.set_ylim(-0.8, 16.5)
    ax.text(0.012, PHE12 - 0.9, 'кромка сегмента (13.96°)', fontsize=8.5, color=C12)
    ax.text(0.012, 15.25, 'середина зазора (15°)', fontsize=8.5, color=MUTED)
    ax.text(0.012, 0.4, 'середина сегмента', fontsize=8.5, color=MUTED)
    ax.text(0.012, 5.2, 'всё время сегмент 0 —\nна тонкой стороне\n(самая тонкая стенка при 3°)',
            fontsize=9, color=INK)
    ax.set_ylabel('расстояние максимума от середины сегмента, град', color=INK)
    ax.set_title('в) Где самая большая деформация', loc='left', fontsize=11, color=INK)
    for ax in axs:
        ax.axvline(0.092, color=MUTED, lw=0.9, ls='--')
        ax.text(0.094, 0.98, 'максимум силы', transform=ax.get_xaxis_transform(), fontsize=8.5, color=MUTED, va='top')
        ax.set_xlabel('номинальная деформация ε_ном', color=INK)
        style(ax)
    fig.suptitle('Как гармоники растут при раздаче: модель всего кольца, H = 3 мм, 12 сегментов, μ = 0.05 '
                 '(расчёт до ε_ном = %.3f)' % en[-1], fontsize=13, color=INK, x=0.01, ha='left', y=1.02)
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')


def fig_nseg(out):
    cases = [('H = 8 мм', os.path.join(RUNS, 'nseg', 'N8_H8_mu0.07_C3D8I'), os.path.join(RUNS, 'nseg', 'N12_H8_mu0.07_C3D8I')),
             ('H = 3 мм', os.path.join(RUNS, 'nseg', 'N8_H3_mu0.07_C3D8I'), os.path.join(RUNS, 'nseg', 'N12_H3_mu0.07_C3D8I'))]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw=dict(hspace=0.45, wspace=0.15))
    for i, (lab, r8, r12) in enumerate(cases):
        for j, x in enumerate((0.10, 0.20)):
            ax = axs[i, j]
            for run, col, dx in ((r8, '#eb6834', -0.4), (r12, C12, 0.4)):
                y, N, en = sector_profile(run, x, 'in')
                a, _ = spectrum(y)
                ks = [k for k in range(1, 61) if a[k] > 1e-4 * a[0]]
                ax.bar([k + dx for k in ks], [100 * a[k] / a[0] for k in ks], width=0.8, color=col,
                       label='%d сегментов' % N)
            ax.set_xlim(0, 60); ax.set_xticks(range(0, 61, 4))
            ax.set_xlabel('n — число волн на оборот', color=INK); ax.set_ylabel('% от средней', color=INK)
            ax.set_title('%s, μ = 0.07, ε_ном ≈ %.2f (внутренняя поверхность)' % (lab, x), loc='left', fontsize=10.5,
                         color=INK)
            ax.legend(frameon=False, fontsize=9)
            style(ax)
    fig.suptitle('8 сегментов против 12 в спектре: гармоники стоят только на кратных N (8, 16, 24… или 12, 24, 36…); '
                 'больше сегментов — паразитные волны выше по n', fontsize=12.5, color=INK, x=0.01, ha='left', y=1.0)
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')


def fig_friction(out):
    runs = {'H = 8 мм': [('0', os.path.join(RUNS, 'stage1', 'H8_mu0_C3D8I')), ('0.05', os.path.join(RUNS, 'stage1', 'H8_mu0.05_C3D8I')),
                         ('0.07', os.path.join(RUNS, 'nseg', 'N8_H8_mu0.07_C3D8I')), ('0.2', os.path.join(RUNS, 'stage1', 'H8_mu0.2_C3D8I'))],
            'H = 3 мм': [('0', os.path.join(RUNS, 'stage1', 'H3_mu0_C3D8I')), ('0.05', os.path.join(RUNS, 'stage1', 'H3_mu0.05_C3D8I')),
                         ('0.07', os.path.join(RUNS, 'nseg', 'N8_H3_mu0.07_C3D8I')), ('0.2', os.path.join(RUNS, 'stage1', 'H3_mu0.2_C3D8I'))]}
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw=dict(wspace=0.2))
    for ax, surf, ttl in ((axs[0], 'out', 'а) наружная поверхность (то, что видит камера DIC)'),
                          (axs[1], 'in', 'б) внутренняя поверхность (там зарождается трещина)')):
        for (lab, rr), col in zip(runs.items(), ('#2a78d6', '#eb6834')):
            for x, ls in ((0.05, ':'), (0.10, '-')):
                mus, vals = [], []
                for mu, run in rr:
                    y, N, en = sector_profile(run, x, surf)
                    c = np.fft.rfft(y) / len(y)
                    # со знаком: + деформация больше у кромок и над зазорами, − под серединами сегментов
                    mus.append(float(mu)); vals.append(-100 * 2 * c[8].real / c[0].real)
                ax.plot(mus, vals, color=col, ls=ls, lw=2.2, marker='o', label='%s, ε_ном ≈ %.2f' % (lab, x))
        ax.set_xlabel('коэффициент трения сегмент–кольцо μ', color=INK)
        ax.set_ylabel('амплитуда n = 8 со знаком, % от средней', color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.text(0.003, ax.get_ylim()[1] * 0.97 if False else 0.5, '', fontsize=8)
        ax.set_title(ttl, loc='left', fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    fig.text(0.5, -0.04, 'Знак: «+» — деформация больше у кромок и над зазорами, «−» — больше под серединами сегментов.',
             ha='center', fontsize=9.5, color=MUTED)
    fig.suptitle('Гармоника n = N как «датчик трения» (8 сегментов, расчёты этапа 1): по её амплитуде можно оценить μ',
                 fontsize=13, color=INK, x=0.01, ha='left', y=1.02)
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')


def fig_rct(fr, out):
    phi = np.linspace(0, 360, 720, endpoint=False)
    p = np.radians(phi)
    M = 1 / np.pi - 0.5 * np.abs(np.sin(p))           # изгибающий момент тонкого кольца, ×P·R; φ от линии нагрузки
    f = at(fr, 0.09)
    e = np.array(f['in'])[np.argsort(f['phi'])]
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.8, 1.6, 1.2], hspace=0.45, wspace=0.28)
    # схемы
    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(np.append(1.15 * np.cos(t), 1.15) , np.append(1.15 * np.sin(t), 0), color=C_RING)
    ax.fill(np.cos(t), np.sin(t), color='white')
    for y0 in (1.32, -1.32):
        ax.add_patch(plt.Rectangle((-1.4, y0 - 0.08 if y0 > 0 else y0 - 0.08), 2.8, 0.16, color='#9aa5b1'))
    for y0, dy in ((1.75, -0.3), (-1.75, 0.3)):
        ax.annotate('', (0, y0 + dy), xytext=(0, y0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=2))
    for a in (90, 270):
        ax.plot(0.95 * math.cos(math.radians(a)), 0.95 * math.sin(math.radians(a)), 'o', color=C1, ms=7)
    for a in (0, 180):
        ax.plot(1.2 * math.cos(math.radians(a)), 1.2 * math.sin(math.radians(a)), 'o', color=C1, ms=7)
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('а) Сжатие кольца:\n4 опасных места (оранжевые)', loc='left', fontsize=10.5, color=INK)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(phi, M, color=C1, lw=2)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel('φ от точки нагрузки, град', color=INK); ax.set_ylabel('изгибающий момент, ×P·R', color=INK)
    ax.set_title('б) Изгибающий момент по кругу (тонкое кольцо)', loc='left', fontsize=10.5, color=INK)
    style(ax)
    ax = fig.add_subplot(gs[0, 2])
    a, _ = spectrum(M)
    ax.bar(range(0, 25), a[:25], color=[C1 if k % 2 == 0 and k else '#c9c8c0' for k in range(25)])
    ax.set_xlabel('n', color=INK); ax.set_ylabel('амплитуда, ×P·R', color=INK)
    ax.set_title('в) Спектр: среднего (n = 0) нет,\nглавная — вторая гармоника, затем 4, 6, …', loc='left', fontsize=10.5, color=INK)
    style(ax)
    ax = fig.add_subplot(gs[1, 0])
    ax.fill(np.append(1.15 * np.cos(t), 1.15), np.append(1.15 * np.sin(t), 0), color=C_RING)
    ax.fill(np.cos(t), np.sin(t), color='white')
    for s in range(12):
        ax.add_patch(Wedge((0, 0), 1.0, 30 * s - PHE12, 30 * s + PHE12, width=0.18, fc=C_SEG, ec='#6f8fb3', lw=0.5))
    for s in range(0, 12, 2):
        a0 = math.radians(30 * s)
        ax.annotate('', (1.0 * math.cos(a0), 1.0 * math.sin(a0)), xytext=(0.55 * math.cos(a0), 0.55 * math.sin(a0)),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2))
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('г) Раздача на сегментах:\nвсё кольцо почти одинаково', loc='left', fontsize=10.5, color=INK)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.sort(f['phi']), e, color=C12, lw=1.6)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel('φ, град', color=INK); ax.set_ylabel('ε_θ', color=INK)
    ax.set_title('д) Деформация по кругу (наша модель, ε_ном = %.3f)' % f['en'], loc='left', fontsize=10.5, color=INK)
    style(ax)
    ax = fig.add_subplot(gs[1, 2])
    a, _ = spectrum(e)
    ax.bar(range(0, 25), a[:25], color=[C0 if k == 0 else C12 if k == 12 else C1 if k == 1 else CSB if k in (11, 13) else '#c9c8c0'
                                         for k in range(25)])
    ax.set_xlabel('n', color=INK); ax.set_ylabel('амплитуда', color=INK)
    ax.set_title('е) Спектр: главное — среднее (n = 0),\nпаразитные n = 12 и n = 1 малы', loc='left', fontsize=10.5, color=INK)
    style(ax)
    fig.suptitle('Два испытания на одном языке: сжатие кольца нагружает второй гармоникой, раздача — нулевой',
                 fontsize=13, color=INK, x=0.01, ha='left', y=1.0)
    fig.savefig(out, dpi=115, bbox_inches='tight', facecolor='white')


def fig_residual(fr, out):
    f = at(fr, 0.09)
    phi = np.sort(f['phi'])
    model = np.array(f['in'])[np.argsort(f['phi'])]
    rng = np.random.default_rng(3)
    M = 4096
    x = np.linspace(0, 360, M, endpoint=False)
    mech = np.interp(x, np.append(phi, phi[0] + 360), np.append(model, model[0]))
    # «опыт»: та же механика, но трение чуть больше (n = 12 на 15 % выше) и разностенность 2 % вместо 1 %,
    # плюс мелкомасштабная неоднородность («гидриды»: длина корреляции ≈ 0.1 мм) и шум измерения
    mech_true = mech + 0.15 * component(mech, [12]) + 1.0 * component(mech, [1])
    white = rng.normal(size=M)
    k = np.fft.rfftfreq(M, d=1.0 / M)                 # номер гармоники
    lam = 0.1 / (2 * math.pi * RI) * M                # 0.1 мм в точках
    shaped = np.fft.irfft(np.fft.rfft(white) * np.exp(-0.5 * (k / (M / lam)) ** 2) * (k > 200), M)
    hyd = 0.004 * shaped / shaped.std()
    meas = mech_true + hyd + 0.0008 * rng.normal(size=M)
    resid = meas - mech
    low = component(resid, list(range(0, 60)))
    high = resid - low
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], hspace=0.55, wspace=0.18)
    items = [(meas, 'а) «опыт»: что измерила бы камера (синтетический пример)', INK),
             (mech, 'б) модель без гидридов: механика испытания', C12),
             (resid, 'в) остаток = опыт − модель', C1)]
    for i, (y, ttl, col) in enumerate(items):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(x, y, color=col, lw=0.8)
        if i == 2:
            ax.plot(x, low, color=C12, lw=2, label='низкие гармоники остатка (n < 60): ошибки модели')
            ax.legend(frameon=False, fontsize=9, loc='upper right')
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
        ax.set_title(ttl, loc='left', fontsize=10.5, color=INK)
        style(ax)
    ax = fig.add_subplot(gs[:, 1])
    a_m, _ = spectrum(model)                          # модель — на своей сетке (336 точек, до n ≈ 168)
    ax.loglog(np.arange(1, len(a_m)), a_m[1:], color=C12, lw=1.4, label='модель (сетка 0.1 мм: до n ≈ 170)')
    a_r, _ = spectrum(resid)
    ax.loglog(np.arange(1, len(a_r)), a_r[1:], color=C1, lw=1.0, label='остаток')
    ax.axvspan(1, 110, color=C12, alpha=0.07, lw=0)
    ax.axvspan(200, 2048, color=C1, alpha=0.07, lw=0)
    ax.text(1.3, 3e-2, 'механика испытания\n(n < 100):\nтрение, разностенность,\nсегменты — сюда же\nпопадают ошибки модели',
            fontsize=9, color=C12, va='top')
    ax.text(215, 3e-2, 'микроструктура\n(n > 200, < 0.2 мм):\nгидриды, текстура', fontsize=9, color=C1, va='top')
    ax.set_xlabel('n (лог.)', color=INK); ax.set_ylabel('амплитуда (лог.)', color=INK)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_title('г) Спектры: масштабы не перекрываются', loc='left', fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    style(ax)
    fig.suptitle('Идея: модель вычитает механику испытания, остаток делится по масштабу — '
                 'низкие n = ошибки модели, высокие n = гидриды (ИЛЛЮСТРАЦИЯ на синтетических данных)',
                 fontsize=12.5, color=INK, x=0.01, ha='left', y=1.0)
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', default=os.path.join(HERE, '..', 'notes', 'figs'))
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    fr = ring_frames()
    print('ring frames:', len(fr), 'last eps_nom = %.3f' % fr[-1]['en'])
    jobs = [('curve', lambda o: fig_curve(fr, o)), ('build', lambda o: fig_build(fr, o)),
            ('spectrum', lambda o: fig_spectrum(fr, o)), ('modulation', lambda o: fig_modulation(fr, o)),
            ('growth', lambda o: fig_growth(fr, o)), ('nseg', fig_nseg), ('friction', fig_friction),
            ('rct', lambda o: fig_rct(fr, o)), ('residual', lambda o: fig_residual(fr, o))]
    for name, fn in jobs:
        if only and name not in only:
            continue
        o = os.path.join(a.o, 'harm_%s.png' % name)
        fn(o)
        print('wrote', o)


if __name__ == '__main__':
    main()
