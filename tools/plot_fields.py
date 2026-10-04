#!/usr/bin/env python3
"""Поля этапа 1 из m.frd: карты ε_θ на развёртке внутренней поверхности
и деформированные сечения z = 0 с PEEQ.

    python3 tools/plot_fields.py maps     -o notes/figs/stage1_maps.png
    python3 tools/plot_fields.py sections -o notes/figs/stage1_sections.png

maps     — ε_θ(φ, z) внутренней поверхности (по хордам между соседними по φ
           узлами каждого ряда z) при ε_ном = 0.30 для четырёх случаев C3D8I,
           где локализация идёт в разных местах;
sections — сечение z = 0 в исходном и деформированном виде, цвет — PEEQ
           (узловая, как пишет решатель), H = 8 мм, μ = 0 / 0.05 / 0.2.
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_stage1 import read_deck          # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'stage1')
INK, MUTED = '#1f1f1e', '#6b6a63'
PHI_EDGE = 21.458


def frd_frame(path, want, x):
    """U и PE в кадре с ε_ном, ближайшей к x (время = ε_ном·R_i/u_r — снаружи)."""
    frames, t, kind, cur = {}, None, None, None
    with open(path) as f:
        for line in f:
            if line.startswith('  100CL'):
                t = float(line[12:24])
                kind = cur = None
            elif line.startswith(' -4'):
                kind = line.split()[1]
                cur = frames.setdefault(t, {}).setdefault(kind, {}) if kind in ('DISP', 'PE') else None
            elif cur is not None and line.startswith(' -1'):
                n = int(line[3:13])
                if n in want:
                    cur[n] = [float(line[13 + 12 * k:25 + 12 * k]) for k in range(3 if kind == 'DISP' else 1)]
            elif line.startswith(' -3'):
                cur = None
    return frames


def pick(frames, deck, x):
    ts = [t for t in frames if 'DISP' in frames[t]]
    t = min(ts, key=lambda t: abs(t * deck['ur'] / deck['ri'] - x))
    return t * deck['ur'] / deck['ri'], frames[t]


def surface_map(run, x):
    deck = read_deck(os.path.join(ROOT, run, 'm.inp'))
    nodes = deck['nodes']
    inner = set(deck['sets']['RING_INNER'])
    fr = frd_frame(os.path.join(ROOT, run, 'm.frd'), inner, x)
    en, f = pick(fr, deck, x)
    u = f['DISP']
    rows = {}
    for n in inner:
        rows.setdefault(round(nodes[n][2], 6), []).append(n)
    zs = sorted(rows)
    phis, grid = None, []
    for z in zs:
        ids = sorted(rows[z], key=lambda n: math.atan2(nodes[n][1], nodes[n][0]))
        ph, ee = [], []
        for p, q in zip(ids, ids[1:]):
            P, Q = nodes[p], nodes[q]
            l0 = math.dist(P, Q)
            l = math.dist([P[k] + u[p][k] for k in range(3)], [Q[k] + u[q][k] for k in range(3)])
            ph.append(math.degrees(0.5 * (math.atan2(P[1], P[0]) + math.atan2(Q[1], Q[0]))))
            ee.append(math.log(l / l0))
        phis = phis or ph
        grid.append(ee)
    return dict(deck=deck, eps_nom=en, phi=phis, z=zs, e=grid)


def fig_maps(out, x=0.30):
    cases = [('H5_mu0_C3D8I', 'H = 5 мм, μ = 0\nнад серединой сегмента'),
             ('H8_mu0.05_C3D8I', 'H = 8 мм, μ = 0.05\nу кромки, середина высоты'),
             ('H20_mu0.05_C3D8I', 'H = 20 мм, μ = 0.05\nу кромки, в 5–6 мм от торца'),
             ('H8_mu0.2_C3D8I', 'H = 8 мм, μ = 0.2\nу кромки и над зазором')]
    maps = [surface_map(r, x) for r, _ in cases]
    vmax = max(max(max(row) for row in m['e']) for m in maps)
    vmin = min(min(min(row) for row in m['e']) for m in maps)
    hmax = max(m['z'][-1] for m in maps)
    fig, axs = plt.subplots(1, len(cases), figsize=(4.0 * len(cases), 6.4), sharey=True)
    for ax, m, (run, title) in zip(axs, maps, cases):
        ri = m['deck']['ri']
        s = [ri * math.radians(p) for p in m['phi']]
        cs = ax.pcolormesh(s, m['z'], m['e'], cmap='Blues', vmin=vmin, vmax=vmax, shading='nearest')
        se = ri * math.radians(PHI_EDGE)
        ax.axvline(se, color=MUTED, lw=1, ls='--')
        ax.text(se - 0.04, m['z'][-1] + 0.15, 'кромка', fontsize=8, color=MUTED, ha='right')
        ax.axhline(m['z'][-1], color=INK, lw=1.2)
        # место наибольшей ε_θ
        i, j = max(((i, j) for i in range(len(m['z'])) for j in range(len(s))),
                   key=lambda ij: m['e'][ij[0]][ij[1]])
        ax.plot(s[j], m['z'][i], marker='o', ms=9, mfc='none', mec='#eb6834', mew=2)
        ax.text(s[j], m['z'][i] + 0.35, '%.2f' % m['e'][i][j], color='#eb6834', fontsize=10,
                ha='center', va='bottom', fontweight='bold')
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_xlabel('дуга по R_i, мм', color=INK, fontsize=9)
        ax.set_ylim(0, hmax + 0.6)
    axs[0].set_ylabel('z, мм от середины высоты (линия — свободный торец)', color=INK)
    cb = fig.colorbar(cs, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label('окружная деформация ε_θ (лог.), внутренняя поверхность')
    fig.suptitle('Карты ε_θ на внутренней поверхности при ε_ном ≈ %.2f (C3D8I, высоты в масштабе). '
                 'Дуга: 0 — середина сегмента, пунктир — кромка; кружок — наибольшая ε_θ'
                 % maps[0]['eps_nom'], fontsize=11.5, color=INK, x=0.01, ha='left', y=1.0)
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def section(run, x):
    deck = read_deck(os.path.join(ROOT, run, 'm.inp'))
    nodes = deck['nodes']
    ring = {n for e in deck['esets'].get('SEC_Z0', []) for n in deck['elems'][e]}
    fr = frd_frame(os.path.join(ROOT, run, 'm.frd'), ring, x)
    en, f = pick(fr, deck, x)
    quads = []
    for e in deck['esets']['SEC_Z0']:
        c = deck['elems'][e]
        if len(c) != 8:
            continue
        bot = c[:4]                       # грань z = 0 у C3D8I
        if not all(abs(nodes[n][2]) < 1e-9 for n in bot):
            continue
        quads.append(bot)
    return deck, en, f, quads


def fig_sections(out, x=0.30):
    cases = [('H8_mu0_C3D8I', 'μ = 0'), ('H8_mu0.05_C3D8I', 'μ = 0.05'), ('H8_mu0.2_C3D8I', 'μ = 0.2')]
    data = [section(r, x) for r, _ in cases]
    vmax = max(max(f['PE'][n][0] for q in qs for n in q) for _, _, f, qs in data)
    fig, axs = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 5.4))
    for ax, (deck, en, f, qs), (run, lab) in zip(axs, data, cases):
        nodes, u = deck['nodes'], f['DISP']
        ref = [[nodes[n][:2] for n in q] for q in qs]
        ax.add_collection(PolyCollection(ref, facecolor='none', edgecolor='#c9c8c0', linewidths=0.3))
        dfm = [[(nodes[n][0] + u[n][0], nodes[n][1] + u[n][1]) for n in q] for q in qs]
        val = [sum(f['PE'][n][0] for n in q) / 4 for q in qs]
        pc = PolyCollection(dfm, array=val, cmap='Blues', clim=(0, vmax), edgecolor='none')
        ax.add_collection(pc)
        # сегмент в деформированном положении (дуга R_i, сдвиг на u_r по x)
        ur = en * deck['ri']
        ph = [math.radians(PHI_EDGE * k / 60) for k in range(61)]
        ax.plot([deck['ri'] * math.cos(p) + ur for p in ph], [deck['ri'] * math.sin(p) for p in ph],
                color='#eb6834', lw=2)
        ax.set_aspect('equal')
        ax.set_xlim(5.0, 8.2)
        ax.set_ylim(-0.15, 3.15)
        pmax = max(range(len(val)), key=lambda i: val[i])
        cx = sum(p[0] for p in dfm[pmax]) / 4
        cy = sum(p[1] for p in dfm[pmax]) / 4
        ax.annotate('PEEQ max %.2f' % val[pmax], xy=(cx, cy), xytext=(cx - 1.35, min(cy + 0.55, 2.95)),
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle='->', color=INK, lw=0.8))
        ax.set_title('H = 8 мм, %s' % lab, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('x, мм', color=INK)
    axs[0].set_ylabel('y, мм', color=INK)
    cb = fig.colorbar(pc, ax=axs, shrink=0.85, pad=0.01)
    cb.set_label('эквивалентная пластическая деформация PEEQ')
    fig.suptitle('Сечение z = 0 при ε_ном ≈ %.2f: серый — исходное, цвет — деформированное, '
                 'оранжевый — поверхность сегмента' % data[0][1], fontsize=12, color=INK, x=0.01,
                 ha='left')
    fig.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('maps', 'sections'))
    ap.add_argument('-o', required=True)
    ap.add_argument('--eps', type=float, default=0.30)
    a = ap.parse_args()
    (fig_maps if a.what == 'maps' else fig_sections)(a.o, a.eps)
    print('wrote', a.o)


if __name__ == '__main__':
    main()
