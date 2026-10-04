#!/usr/bin/env python3
"""Карты полей на внутренней поверхности в плоскости φ–z для серии высот этапа 1.

    python3 tools/plot_maps_stage1.py --mu 0.05 --what eps  -o notes/figs/stage1_maps_eps_mu0.05.png
    python3 tools/plot_maps_stage1.py --mu 0.05 --what fields -o notes/figs/stage1_maps_fields_mu0.05.png

Строки — высоты кольца H (высота строки пропорциональна H/2), по горизонтали
φ от 0 (середина сегмента) до 22.5° (середина зазора), по вертикали z от
середины высоты (низ, плоскость симметрии) до свободного торца (верх).
--what eps    — ε_θ (лог.) при ε_ном = 0.10, 0.20, 0.30 (столбцы, общая шкала в столбце);
--what fields — при ε_ном = 0.20: ε_θ, PEEQ и утонение t/t₀ (столбцы).
ε_θ — по хорде между соседними узлами внутренней поверхности в ряду z;
PEEQ — узловая (как пишет решатель); t/t₀ — расстояние между узлом
внутренней и наружной поверхности с теми же φ и z. Расчёты C3D8I.
"""
import argparse
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

ROOT = os.path.join(HERE, '..', 'runs', 'stage1')
INK, MUTED, MARK = '#1f1f1e', '#6b6a63', '#eb6834'
PHI_EDGE = 21.458


def frd_frames(path, want):
    out, t, kind, cur = {}, None, None, None
    with open(path) as f:
        for line in f:
            if line.startswith('  100CL'):
                t = float(line[12:24])
            elif line.startswith(' -4'):
                kind = line.split()[1]
                cur = out.setdefault(t, {}).setdefault(kind, {}) if kind in ('DISP', 'PE') else None
            elif cur is not None and line.startswith(' -1'):
                n = int(line[3:13])
                if n in want:
                    cur[n] = (float(line[13:25]), float(line[25:37]), float(line[37:49])) if kind == 'DISP' \
                        else float(line[13:25])
            elif line.startswith(' -3'):
                cur = None
    return out


def fields(run, targets):
    """{ε_ном: dict(phi_c, z, eps, phi_n, pe, thin)} для ближайших кадров."""
    deck = read_deck(os.path.join(ROOT, run, 'm.inp'))
    nodes = deck['nodes']
    inner, outer = set(deck['sets']['RING_INNER']), set(deck['sets']['RING_OUTER'])
    key = lambda n: (round(math.degrees(math.atan2(nodes[n][1], nodes[n][0])), 5), round(nodes[n][2], 6))
    kin = {key(n): n for n in inner}
    kout = {key(n): n for n in outer}
    phis = sorted({k[0] for k in kin})
    zs = sorted({k[1] for k in kin})
    fr = frd_frames(os.path.join(ROOT, run, 'm.frd'), inner | outer)
    times = [t for t in fr if len(fr[t].get('DISP', {})) == len(inner | outer)]
    res = {}
    for x in targets:
        t = min(times, key=lambda t: abs(t * deck['ur'] / deck['ri'] - x))
        U, PE = fr[t]['DISP'], fr[t].get('PE', {})
        pos = lambda n: [nodes[n][k] + U[n][k] for k in range(3)]
        eps, pe, thin = [], [], []
        for z in zs:
            row_e, row_p, row_t = [], [], []
            for i, ph in enumerate(phis):
                a = kin[(ph, z)]
                b = kout.get((ph, z))
                row_p.append(PE.get(a, float('nan')))
                row_t.append(math.dist(pos(a), pos(b)) / math.dist(nodes[a], nodes[b]) if b else float('nan'))
                if i + 1 < len(phis):
                    c = kin[(phis[i + 1], z)]
                    row_e.append(math.log(math.dist(pos(a), pos(c)) / math.dist(nodes[a], nodes[c])))
            eps.append(row_e); pe.append(row_p); thin.append(row_t)
        res[x] = dict(en=t * deck['ur'] / deck['ri'], phi_c=[0.5 * (p + q) for p, q in zip(phis, phis[1:])],
                      phi_n=phis, z=zs, eps=eps, pe=pe, thin=thin, H=deck['H'])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', default='0.05')
    ap.add_argument('--what', choices=('eps', 'fields'), default='eps')
    ap.add_argument('-o', required=True)
    a = ap.parse_args()
    Hs = [H for H in (3, 5, 8, 12, 16, 20)
          if os.path.exists(os.path.join(ROOT, 'H%d_mu%s_C3D8I' % (H, a.mu), 'm.frd'))]
    names = {'eps': 'ε_θ (лог.)', 'pe': 'PEEQ', 'thin': 'утонение t/t₀'}
    if a.what == 'eps':
        targets = (0.10, 0.20, 0.30)
        cols = [('eps', x) for x in targets]
    else:
        targets = (0.20,)
        cols = [('eps', 0.20), ('pe', 0.20), ('thin', 0.20)]
    data = {H: fields('H%d_mu%s_C3D8I' % (H, a.mu), targets) for H in Hs}

    # раскладка в дюймах: строки — высоты (высота строки ∝ H/2), столбцы — поля/уровни ε_ном
    W, GX, L, R = 4.2, 0.7, 1.05, 0.2
    SC, MINH, GY = 0.42, 1.0, 0.42
    TOP, BOT = 1.6, 1.3
    hs = [max(SC * H / 2, MINH) for H in Hs]
    FW = L + len(cols) * W + (len(cols) - 1) * GX + R
    FH = TOP + sum(hs) + GY * (len(Hs) - 1) + BOT
    fig = plt.figure(figsize=(FW, FH))
    box = lambda x0, y0, w, h: fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])
    for j, (field, x) in enumerate(cols):
        x0 = L + j * (W + GX)
        vals = sorted(v for H in Hs for row in data[H][x][field] for v in row if v == v)
        lo, hi = vals[int(0.005 * len(vals))], vals[int(0.995 * len(vals)) - 1]
        ext = {(False, False): 'neither', (True, False): 'min', (False, True): 'max'}.get(
            (lo > vals[0], hi < vals[-1]), 'both')
        levels = MaxNLocator(nbins=7).tick_values(lo, hi)
        cmap = 'Blues_r' if field == 'thin' else 'Blues'
        pick, word = (min, 'мин.') if field == 'thin' else (max, 'макс.')
        ytop = FH - TOP
        for i, H in enumerate(Hs):
            d = data[H][x]
            ax = box(x0, ytop - hs[i], W, hs[i])
            ytop -= hs[i] + GY
            xs, grid = (d['phi_c'], d['eps']) if field == 'eps' else (d['phi_n'], d[field])
            if field == 'eps':
                # φ = 0 и 22.5° — плоскости симметрии: значение у края = значению в крайней хорде
                xs = [0.0] + xs + [22.5]
                grid = [[r[0]] + r + [r[-1]] for r in grid]
            m = ax.pcolormesh(xs, d['z'], grid, cmap=cmap, vmin=lo, vmax=hi, shading='gouraud')
            cs = ax.contour(xs, d['z'], grid, levels=levels, colors=INK, linewidths=0.5, alpha=0.6)
            ax.clabel(cs, fmt='%.2f', fontsize=7, inline=True, inline_spacing=2)
            ax.axvline(PHI_EDGE, color=INK, lw=1.0, ls='--', alpha=0.75)
            ii, jj = pick(((r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] == grid[r][c]),
                          key=lambda rc: grid[rc[0]][rc[1]])
            ax.plot(xs[jj], d['z'][ii], 'o', mfc='none', mec=MARK, mew=2, ms=9, clip_on=False)
            ax.text(1, 1.03, '%s %.3f: φ = %.1f°, z = %.1f мм' % (word, grid[ii][jj], xs[jj], d['z'][ii]),
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color=INK)
            ax.set_xlim(0, 22.5); ax.set_ylim(0, H / 2)
            ax.set_xticks([0, 5, 10, 15, 20, 22.5]); ax.set_xticklabels(['0', '5', '10', '15', '20', ''])
            ax.set_yticks([0, H / 2] if H <= 5 else list(range(0, int(H / 2) + 1, 2)))
            if j == 0:
                ax.set_ylabel('H = %g мм\nz, мм' % H, color=INK, fontsize=10)
            if i == 0:
                ax.text(0, 1.03 + 0.3 / hs[0], '%s, ε_ном ≈ %.2f' % (names[field], d['en']), transform=ax.transAxes,
                        ha='left', va='bottom', fontsize=11.5, color=INK, fontweight='bold')
            if i == len(Hs) - 1:
                ax.set_xlabel('φ, град', color=INK, fontsize=9.5)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=8, colors=MUTED)
        cax = box(x0 + 0.1 * W, 0.42, 0.8 * W, 0.13)
        cb = fig.colorbar(m, cax=cax, orientation='horizontal', extend=ext, ticks=levels)
        cb.ax.tick_params(labelsize=8)
        cb.set_label('%s (шкала общая для столбца)' % names[field], fontsize=8.5, color=MUTED)
    what = 'ε_θ' if a.what == 'eps' else 'ε_θ, PEEQ и утонение'
    fig.text(0.01, 1 - 0.18 / FH, 'Этап 1, C3D8I, μ = %s: %s на внутренней поверхности в плоскости φ–z' % (a.mu, what),
             fontsize=13.5, color=INK, va='top', fontweight='bold')
    fig.text(0.01, 1 - 0.5 / FH, 'Развёртка половины высоты кольца: низ — середина высоты (z = 0), верх — свободный '
             'торец; φ = 0 — середина сегмента, 22.5° — середина зазора;\nпунктир — кромка сегмента (21.46°), правее '
             'неё — над зазором. Линии — изолинии с подписями, кружок — экстремум (его значение над панелью).',
             fontsize=9.5, color=MUTED, va='top')
    fig.savefig(a.o, dpi=110, facecolor='white')
    print('wrote', a.o)


if __name__ == '__main__':
    main()
