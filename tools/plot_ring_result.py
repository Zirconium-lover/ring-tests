#!/usr/bin/env python3
"""Итог модели всего кольца (mksector --full-ring): где и как пошли шейки.

    python3 tools/plot_ring_result.py runs/ring/R12_H3_mu0.05_ecc1_spr -o notes/figs/ring_result_R12_H3.png

а) ε_θ на внутренней поверхности, развёртка φ × z, последний кадр; б) толщина
стенки в середине высоты: наименьшая по каждому сегменту в зависимости от
ε_ном (чётные и нечётные сегменты разным цветом); в) гармоники деформации
внутренней поверхности (середина высоты) в % от средней: n = 1, 3, 6, 12;
г) трёхосность η в точке наибольшей PEEQ (середина высоты и торец) и сила.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post_stage1 import read_deck          # noqa: E402
from plot_maps_stage1 import frd_frames    # noqa: E402

INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
C_EVEN, C_ODD = '#d03b3b', '#2a78d6'


def style(ax):
    ax.grid(color=GRID, lw=0.7)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    d = read_deck(os.path.join(a.run, 'm.inp'))
    nodes, sector = d['nodes'], d['sector']
    nseg = round(180 / sector)
    head = open(os.path.join(a.run, 'm.inp')).readlines()[1]
    phe = float(head.split('phi_edge_deg=')[1].split()[0])
    inner = set(d['sets']['RING_INNER'])
    key = lambda n: (round(math.degrees(math.atan2(nodes[n][1], nodes[n][0])) % 360, 4), round(nodes[n][2], 6))
    kin = {key(n): n for n in inner}
    phis = sorted({k[0] for k in kin}); zs = sorted({k[1] for k in kin})
    fr = frd_frames(os.path.join(a.run, 'm.frd'), inner)
    t = max(t for t in fr if len(fr[t].get('DISP', {})) == len(inner))
    U = fr[t]['DISP']
    pos = lambda n: [nodes[n][k] + U[n][k] for k in range(3)]
    grid = []
    for z in zs:
        row = []
        for p, q in zip(phis, phis[1:] + phis[:1]):
            na, nb = kin[(p, z)], kin[(q, z)]
            row.append(math.log(math.dist(pos(na), pos(nb)) / math.dist(nodes[na], nodes[nb])))
        grid.append(row)
    pc = [(p + 0.5 * ((q - p) % 360)) % 360 for p, q in zip(phis, phis[1:] + phis[:1])]
    o = np.argsort(pc)
    pc = np.array(pc)[o]; grid = np.array(grid)[:, o]
    en_last = t * d['ur'] / d['ri']

    prof = json.load(open(os.path.join(a.run, 'post', 'profiles.json')))
    ts = sorted(prof, key=float)
    tmin = {k: [] for k in range(nseg)}
    ens = []
    for tt in ts:
        T = prof[tt]['T_Z0']
        ph = np.array(T['phi']) % 360; th = np.array(T['t'])
        ens.append(float(tt) * d['ur'] / d['ri'])
        for k in range(nseg):
            dd = np.abs(((ph - 2 * sector * k + 180) % 360) - 180)
            tmin[k].append(th[dd <= sector].min())
    harm = json.load(open(os.path.join(a.run, 'post', 'harm_profiles.json')))
    hen, H = [], {n: [] for n in (1, 3, 6, 12)}
    for f in harm:
        e = np.array(f['in']); c = np.fft.rfft(e) / len(e)
        hen.append(f['en'])
        for n in H:
            H[n].append(100 * 2 * abs(c[n]) / c[0].real)
    rows = list(csv.DictReader(open(os.path.join(a.run, 'post', 'timeseries.csv'))))
    g = lambda r, k: float(r[k]) if r.get(k) else float('nan')

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1], hspace=0.38, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    lo, hi = np.percentile(grid, 0.5), np.percentile(grid, 99.5)
    m = ax.pcolormesh(pc, zs, grid, cmap='Blues', vmin=lo, vmax=hi, shading='gouraud')
    for k in range(nseg):
        c = 2 * sector * k
        for x in (c - phe, c + phe):
            ax.axvline(x % 360, color=MUTED, lw=0.6, ls='--')
        ax.text(c, zs[-1] * 1.04, str(k), ha='center', va='bottom', fontsize=9,
                color=C_EVEN if k % 2 == 0 else C_ODD)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
    ax.set_xlabel('φ, град (номера — середины сегментов; пунктир — кромки)', color=INK)
    ax.set_ylabel('z, мм (0 — середина высоты)', color=INK)
    ax.set_title('а) ε_θ на внутренней поверхности при ε_ном = %.3f: шейки в серединах чётных сегментов' % en_last,
                 loc='left', fontsize=11, color=INK, pad=18)
    cb = fig.colorbar(m, ax=ax, pad=0.01, fraction=0.03); cb.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for k in range(nseg):
        ax.plot(ens, tmin[k], color=C_EVEN if k % 2 == 0 else C_ODD, lw=1.6 if k % 2 == 0 else 1.0,
                label=('чётные сегменты (0, 2, …, 10)' if k == 0 else 'нечётные (1, 3, …, 11)' if k == 1 else None))
    k_last = {k: tmin[k][-1] for k in range(0, nseg, 2)}
    for k, v in k_last.items():
        ax.text(ens[-1] + 0.003, v, str(k), fontsize=8, color=C_EVEN, va='center')
    ax.set_xlabel('ε_ном', color=INK); ax.set_ylabel('наименьшая t/t₀ в сегменте (середина высоты)', color=INK)
    ax.set_title('б) Утонение по сегментам', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    style(ax)

    ax = fig.add_subplot(gs[1, 1])
    for n, col in ((1, '#eb6834'), (3, '#8e5bd6'), (6, '#d03b3b'), (12, '#2a78d6')):
        ax.plot(hen, H[n], color=col, lw=2, label='n = %d' % n)
    ax.axvline(0.092, color=MUTED, lw=0.8, ls='--')
    ax.set_yscale('log'); ax.set_ylim(0.05, 200)
    ax.set_xlabel('ε_ном', color=INK); ax.set_ylabel('амплитуда, % от средней (лог.)', color=INK)
    ax.set_title('в) Гармоники: n = 6 — шейки через сегмент,\nn = 3 — следующий отбор', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    style(ax)

    ax = fig.add_subplot(gs[1, 2])
    en_r = [g(r, 'eps_nom') for r in rows]
    ax.plot(en_r, [g(r, 'hs_z0_eta_avg') for r in rows], color=INK, lw=2, label='η, середина высоты')
    ax.plot(en_r, [g(r, 'hs_ztop_eta_avg') for r in rows], color=MUTED, lw=1.5, ls='--', label='η, торец')
    ax.axhline(1 / 3, color=MUTED, lw=0.8, ls=':'); ax.text(0.01, 1 / 3 + 0.01, '1/3', fontsize=8, color=MUTED)
    ax.axhline(1 / math.sqrt(3), color=MUTED, lw=0.8, ls=':'); ax.text(0.01, 0.587, '1/√3', fontsize=8, color=MUTED)
    ax.set_ylim(0.15, 0.7)
    ax.set_xlabel('ε_ном', color=INK); ax.set_ylabel('трёхосность η в точке наибольшей PEEQ', color=INK)
    ax.set_title('г) Трёхосность в опасной точке', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    style(ax)
    fig.suptitle('Модель всего кольца: H = 3 мм, 12 сегментов, μ = 0.05, разностенность ±1 %% (тонко при 3°) — расчёт '
                 'встал при ε_ном = %.3f' % ens[-1], fontsize=13, color=INK, x=0.01, ha='left', y=1.0)
    fig.savefig(a.o, dpi=110, bbox_inches='tight', facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
