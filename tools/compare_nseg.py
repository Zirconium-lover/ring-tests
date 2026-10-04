#!/usr/bin/env python3
"""Сравнение оправки на 12 сегментов (сектор 15°) с 8 (22.5°): H = 3 и 8 мм,
μ = 0, 0.05, 0.07; постановка и сетка этапа 1 (C3D8I, полсегмента × полвысоты).

    python3 tools/compare_nseg.py [-o notes/figs]

8 сегментов при μ = 0 и 0.05 — расчёты этапа 1 (runs/stage1), остальное —
runs/nseg (tools/mknseg.sh). По m.frd каждого кадра: ε_θ на внутренней
поверхности (по хорде между соседними узлами в ряду z), её наибольшее
значение и место; неравномерность ε_max/ε_min по дуге в середине высоты
(критерий [R2008]); сила на конусе — из post/timeseries.csv.

Угол везде в долях периода: s = φ/(180°/N): 0 — середина сегмента, 1 —
середина зазора, кромка сегмента — s_e = φ_e/(180°/N) (0.954 при N = 8,
0.931 при N = 12).

Рисунки: nseg_curves.png (кривые), nseg_maps_H3.png и nseg_maps_H8.png
(карты ε_θ в плоскости s–z), таблица — runs/nseg/summary.md.
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from post_stage1 import read_deck          # noqa: E402
from plot_maps_stage1 import frd_frames    # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, '..', 'runs'))
INK, MUTED, GRID, MARK = '#1f1f1e', '#6b6a63', '#e4e3dc', '#eb6834'
MU_COL = {'0': '#2a78d6', '0.05': '#eb6834', '0.07': '#1baf7a'}
N_LS = {8: (0, (5, 3)), 12: '-'}
MUS = ('0', '0.05', '0.07')


def run_dir(N, H, mu):
    if N == 8 and mu != '0.07':
        return os.path.join(ROOT, 'stage1', 'H%d_mu%s_C3D8I' % (H, mu))
    return os.path.join(ROOT, 'nseg', 'N%d_H%d_mu%s_C3D8I' % (N, H, mu))


def analyse(run):
    """Кадры: ε_ном, сетка ε_θ(z, s) на внутренней поверхности и сводка по кадрам."""
    deck = read_deck(os.path.join(run, 'm.inp'))
    nodes, sector = deck['nodes'], deck['sector']
    inner = set(deck['sets']['RING_INNER'])
    key = lambda n: (round(math.degrees(math.atan2(nodes[n][1], nodes[n][0])), 5), round(nodes[n][2], 6))
    kin = {key(n): n for n in inner}
    phis = sorted({k[0] for k in kin})
    zs = sorted({k[1] for k in kin})
    fr = frd_frames(os.path.join(run, 'm.frd'), inner)
    sc = [0.5 * (p + q) / sector for p, q in zip(phis, phis[1:])]
    out = []
    for t in sorted(fr):
        U = fr[t].get('DISP', {})
        if len(U) != len(inner):
            continue
        pos = lambda n: [nodes[n][k] + U[n][k] for k in range(3)]
        grid = []
        for z in zs:
            row = []
            for p, q in zip(phis, phis[1:]):
                a, b = kin[(p, z)], kin[(q, z)]
                row.append(math.log(math.dist(pos(a), pos(b)) / math.dist(nodes[a], nodes[b])))
            grid.append(row)
        ii, jj = max(((i, j) for i in range(len(zs)) for j in range(len(sc))), key=lambda ij: grid[ij[0]][ij[1]])
        mid = grid[0]
        out.append(dict(t=t, en=t * deck['ur'] / deck['ri'], grid=grid, emax=grid[ii][jj], s_max=sc[jj],
                        z_max=zs[ii], ratio_mid=max(mid) / min(mid) if min(mid) > 0 else float('nan'),
                        smax_mid=sc[mid.index(max(mid))]))
    ts = [{k: float(v) for k, v in r.items() if v != ''}
          for r in csv.DictReader(open(os.path.join(run, 'post', 'timeseries.csv')))]
    return dict(deck=deck, s=sc, z=zs, frames=out, ts=ts, se=deck_edge(run) / sector, H=deck['H'])


def deck_edge(run):
    head = open(os.path.join(run, 'm.inp')).readlines()[1]
    return float(head.split('phi_edge_deg=')[1].split()[0])


def at(res, x):
    return min(res['frames'], key=lambda f: abs(f['en'] - x))


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def curves(data, out):
    fig, axs = plt.subplots(4, 2, figsize=(14, 17), sharex=True)
    for j, H in enumerate((3, 8)):
        for mu in MUS:
            for N in (8, 12):
                r = data.get((N, H, mu))
                if not r:
                    continue
                kw = dict(color=MU_COL[mu], ls=N_LS[N], lw=2.0 if N == 12 else 1.7,
                          label='N = %d, μ = %s' % (N, mu))
                axs[0, j].plot([p['eps_nom'] for p in r['ts']], [p['Fz'] / H / 1000 for p in r['ts']], **kw)
                x = [f['en'] for f in r['frames']]
                axs[1, j].plot(x, [f['emax'] for f in r['frames']], **kw)
                axs[2, j].plot(x, [f['ratio_mid'] for f in r['frames']], **kw)
                axs[3, j].plot(x, [f['s_max'] for f in r['frames']], **kw)
        axs[0, j].set_title('H = %d мм\nа) сила на конусе / высота, кН/мм' % H, loc='left', fontsize=11.5,
                            color=INK)
        axs[1, j].set_title('б) наибольшая ε_θ на внутренней поверхности', loc='left', fontsize=11, color=INK)
        axs[1, j].plot([0, 0.37], [0, 0.37], color=MUTED, lw=0.9, ls=':')
        axs[1, j].text(0.30, 0.255, 'ε_θ = ε_ном', fontsize=8, color=MUTED, rotation=0)
        axs[2, j].set_title('в) неравномерность по дуге в середине высоты: ε_max / ε_min', loc='left',
                            fontsize=11, color=INK)
        axs[3, j].set_title('г) где максимум ε_θ: s = φ / (180°/N)\n(0 — середина сегмента, 1 — середина '
                            'зазора, полоса — кромки сегментов)', loc='left', fontsize=11, color=INK)
        axs[3, j].axhspan(0.931, 0.954, color=MUTED, alpha=0.25, lw=0)
        axs[3, j].set_ylim(-0.03, 1.03)
        axs[3, j].set_xlabel('номинальная деформация ε_ном = u_r/R_i', color=INK)
        axs[2, j].set_ylim(1, None)
        for ax in axs[:, j]:
            style(ax)
            ax.axvline(0.092, color=MUTED, lw=0.8, ls='--')
        axs[0, j].text(0.094, 0.02, 'максимум силы', fontsize=8, color=MUTED, transform=axs[0, j].get_xaxis_transform())
    axs[0, 1].legend(frameon=False, fontsize=9, ncol=3, loc='lower right')
    fig.suptitle('12 сегментов против 8: этап 1, C3D8I (полсегмента × полвысоты); пунктир — 8 сегментов, '
                 'сплошные — 12, цвет — μ', fontsize=13, color=INK, x=0.01, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=120, facecolor='white')
    print('wrote', out)


def maps(data, H, out, targets=(0.20, 0.30)):
    cols = [(x, N) for x in targets for N in (8, 12)]
    rows = [mu for mu in MUS if all((N, H, mu) in data for N in (8, 12))]
    if not rows:
        return
    W, GX, L, R = 3.6, 0.45, 1.1, 0.2
    h = max(0.42 * H / 2, 1.05)
    GY, TOP, BOT = 0.45, 1.55, 1.25
    FW = L + len(cols) * W + (len(cols) - 1) * GX + 0.35 * (len(targets) - 1) + R
    FH = TOP + len(rows) * h + (len(rows) - 1) * GY + BOT
    fig = plt.figure(figsize=(FW, FH))
    box = lambda x0, y0, w, hh: fig.add_axes([x0 / FW, y0 / FH, w / FW, hh / FH])
    for ci, (x, N) in enumerate(cols):
        x0 = L + ci * (W + GX) + 0.35 * (ci // 2)
        grp = [at(data[(NN, H, mu)], x)['grid'] for NN in (8, 12) for mu in rows]
        vals = sorted(v for g in grp for row in g for v in row)
        lo, hi = vals[int(0.005 * len(vals))], vals[int(0.995 * len(vals)) - 1]
        levels = MaxNLocator(nbins=7).tick_values(lo, hi)
        for ri_, mu in enumerate(rows):
            r = data[(N, H, mu)]
            f = at(r, x)
            ax = box(x0, FH - TOP - (ri_ + 1) * h - ri_ * GY, W, h)
            xs = [0.0] + r['s'] + [1.0]
            grid = [[row[0]] + row + [row[-1]] for row in f['grid']]
            m = ax.pcolormesh(xs, r['z'], grid, cmap='Blues', vmin=lo, vmax=hi, shading='gouraud')
            cs = ax.contour(xs, r['z'], grid, levels=levels, colors=INK, linewidths=0.5, alpha=0.6)
            ax.clabel(cs, fmt='%.2f', fontsize=7, inline=True, inline_spacing=2)
            ax.axvline(r['se'], color=INK, lw=1.0, ls='--', alpha=0.75)
            ax.plot(f['s_max'], f['z_max'], 'o', mfc='none', mec=MARK, mew=2, ms=9, clip_on=False)
            ax.text(1, 1.03, 'макс. %.3f: s = %.2f, z = %.1f мм' % (f['emax'], f['s_max'], f['z_max']),
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color=INK)
            ax.set_xlim(0, 1); ax.set_ylim(0, H / 2)
            ax.set_yticks([0, H / 2] if H <= 5 else list(range(0, int(H / 2) + 1, 2)))
            ax.tick_params(labelsize=8, colors=MUTED)
            if ci == 0:
                ax.set_ylabel('μ = %s\nz, мм' % mu, color=INK, fontsize=10)
            if ri_ == 0:
                ax.text(0, 1.03 + 0.3 / h, '%d сегментов, ε_ном ≈ %.2f' % (N, f['en']), transform=ax.transAxes,
                        ha='left', va='bottom', fontsize=11.5, color=INK, fontweight='bold')
            if ri_ == len(rows) - 1:
                ax.set_xlabel('s = φ / (180°/N)', color=INK, fontsize=9.5)
            else:
                ax.set_xticklabels([])
        if N == 12:
            cax = box(x0 - 0.5 * W - 0.5 * GX, 0.42, W + GX, 0.13)
            cb = fig.colorbar(m, cax=cax, orientation='horizontal', ticks=levels, extend='both')
            cb.ax.tick_params(labelsize=8)
            cb.set_label('ε_θ (лог.) при ε_ном ≈ %.2f (шкала общая для пары столбцов)' % x, fontsize=8.5,
                         color=MUTED)
    fig.text(0.01, 1 - 0.18 / FH, 'H = %d мм: ε_θ на внутренней поверхности, 8 и 12 сегментов' % H,
             fontsize=13.5, color=INK, va='top', fontweight='bold')
    fig.text(0.01, 1 - 0.5 / FH, 'По горизонтали — доля периода s = φ/(180°/N): 0 — середина сегмента, 1 — середина '
             'зазора; пунктир — кромка сегмента. Низ — середина высоты,\nверх — свободный торец. '
             'Линии — изолинии, кружок — максимум (значение над панелью). 8 сегментов: период 22.5°, '
             '12 сегментов: 15°.', fontsize=9.5, color=MUTED, va='top')
    fig.savefig(out, dpi=110, facecolor='white')
    print('wrote', out)


def table(data, out):
    L = ['# 12 сегментов против 8 (C3D8I, этап 1)', '',
         'ε_θ — наибольшая на внутренней поверхности; s — её место в долях периода',
         '(0 — середина сегмента, 1 — середина зазора, кромка ≈ 0.93–0.95);',
         'ε_max/ε_min — по дуге в середине высоты. `tools/compare_nseg.py`.', '',
         '| H, мм | μ | N | F_z/H max, Н/мм | ε_θ / s при 0.10 | ε_θ / s при 0.20 | ε_θ / s при 0.30 |'
         ' ε_max/ε_min 0.10 / 0.20 / 0.30 |',
         '|---|---|---|---|---|---|---|---|']
    for H in (3, 8):
        for mu in MUS:
            for N in (8, 12):
                r = data.get((N, H, mu))
                if not r:
                    continue
                fz = max(p['Fz'] for p in r['ts']) / H
                cells = []
                for x in (0.10, 0.20, 0.30):
                    f = at(r, x)
                    cells.append('%.3f / %.2f' % (f['emax'], f['s_max']) if abs(f['en'] - x) < 0.02 else '—')
                rat = ' / '.join('%.2f' % at(r, x)['ratio_mid'] if abs(at(r, x)['en'] - x) < 0.02 else '—'
                                 for x in (0.10, 0.20, 0.30))
                L.append('| %d | %s | %d | %.0f | %s | %s |' % (H, mu, N, fz, ' | '.join(cells), rat))
    open(out, 'w').write('\n'.join(L) + '\n')
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', default=os.path.join(HERE, '..', 'notes', 'figs'))
    a = ap.parse_args()
    data = {}
    for H in (3, 8):
        for mu in MUS:
            for N in (8, 12):
                d = run_dir(N, H, mu)
                if os.path.exists(os.path.join(d, 'post', 'timeseries.csv')):
                    data[(N, H, mu)] = analyse(d)
    print('cases:', sorted(data))
    curves(data, os.path.join(a.o, 'nseg_curves.png'))
    for H in (3, 8):
        if any(k[1] == H for k in data):
            maps(data, H, os.path.join(a.o, 'nseg_maps_H%d.png' % H))
    table(data, os.path.join(ROOT, 'nseg', 'summary.md'))


if __name__ == '__main__':
    main()
