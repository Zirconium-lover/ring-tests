#!/usr/bin/env python3
"""Окружная деформация по всей внутренней поверхности кольца (из m.frd).

    python3 tools/surfmap_stage1.py runs/stage1/H8_mu0.05_C3D8I

post_stage1.py видит только две линии (z = 0 и торец). Здесь по полю
перемещений U в m.frd строится ε_θ(φ, z) на внутренней поверхности:
узлы набора RING_INNER группируются по z, в каждом ряду ε_θ = ln(l/l0)
по хордам между соседними по φ узлами. Пишет RUN/post/surfmap.json:
по каждому кадру — ε_ном, и для каждого ряда z — максимум ε_θ по дуге и
его угол φ, а также ε_θ в середине зазора (последняя хорда).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_stage1 import read_deck   # noqa: E402


def frd_disp(path, want):
    """{время: {узел: (u1, u2, u3)}} для блоков DISP; только узлы из want."""
    out, t, cur = {}, None, None
    with open(path) as f:
        for line in f:
            if line.startswith('  100CL'):
                t = float(line[12:24])
                cur = None
            elif line.startswith(' -4  DISP'):
                cur = out.setdefault(t, {})
            elif line.startswith(' -4'):
                cur = None
            elif cur is not None and line.startswith(' -1'):
                n = int(line[3:13])
                if n in want:
                    cur[n] = (float(line[13:25]), float(line[25:37]), float(line[37:49]))
            elif line.startswith(' -3'):
                cur = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    a = ap.parse_args()
    deck = read_deck(os.path.join(a.run, 'm.inp'))
    nodes = deck['nodes']
    inner = set(deck['sets']['RING_INNER'])
    rows = {}
    for n in inner:
        rows.setdefault(round(nodes[n][2], 6), []).append(n)
    zs = sorted(rows)
    for z in zs:
        rows[z].sort(key=lambda n: math.atan2(nodes[n][1], nodes[n][0]))
    disp = frd_disp(os.path.join(a.run, 'm.frd'), inner)
    frames = []
    for t in sorted(disp):
        u = disp[t]
        if len(u) < len(inner):
            continue
        fr = dict(time=t, eps_nom=t * deck['ur'] / deck['ri'], z=[], emax=[], phimax=[], egap=[])
        for z in zs:
            ids = rows[z]
            if len(ids) < 3:
                continue
            best, bphi, last = -1e9, 0.0, 0.0
            for p, q in zip(ids, ids[1:]):
                P, Q = nodes[p], nodes[q]
                l0 = math.dist(P, Q)
                l = math.dist([P[k] + u[p][k] for k in range(3)], [Q[k] + u[q][k] for k in range(3)])
                e = math.log(l / l0)
                phi = math.degrees(0.5 * (math.atan2(P[1], P[0]) + math.atan2(Q[1], Q[0])))
                if e > best:
                    best, bphi = e, phi
                last = e
            fr['z'].append(z)
            fr['emax'].append(best)
            fr['phimax'].append(bphi)
            fr['egap'].append(last)
        frames.append(fr)
    os.makedirs(os.path.join(a.run, 'post'), exist_ok=True)
    out = dict(run=os.path.basename(os.path.normpath(a.run)), H=deck['H'], mu=deck['mu'],
               elem=deck['elem'], frames=frames)
    with open(os.path.join(a.run, 'post', 'surfmap.json'), 'w') as f:
        json.dump(out, f)
    last = frames[-1]
    i = max(range(len(last['emax'])), key=lambda k: last['emax'][k])
    print('%s: %d кадров; последний ε_ном = %.3f, max ε_θ = %.4f при φ = %.2f°, z = %.2f мм'
          % (out['run'], len(frames), last['eps_nom'], last['emax'][i], last['phimax'][i],
             last['z'][i]))


if __name__ == '__main__':
    main()
