#!/usr/bin/env python3
"""Этап 0: маленькая колода для проверки решателя (не модель испытания).

    python3 tools/mksmoke.py -o runs/smoke/contact/m.inp --mode contact
    python3 tools/mksmoke.py -o runs/smoke/nocontact/m.inp --mode disp

Сектор кольца 22.5° (от середины сегмента до середины зазора) × половина
высоты, сетка C3D4 (каждый шестигранник структурной сетки режется на 6
тетраэдров). Вопрос один: работают ли вместе контакт с трением и удаление
элементов в ccx-arch2.

mode=contact: жёсткий сегмент (C3D8, все узлы ведутся перемещением вдоль
              биссектрисы сегмента, ось x), контакт surface-to-surface с
              трением по внутренней поверхности кольца.
mode=disp:    сегмента нет, внутренние узлы кольца получают радиальное
              перемещение u_r напрямую (сравнение без контакта).

Плоскости симметрии: φ = 0 (y = 0) — u_y = 0; φ = 22.5° — уравнение
−sin α·u_x + cos α·u_y = 0; z = 0 — u_z = 0. Верхний торец свободен.
Повреждение нарочно раннее (низкая ε_f), чтобы удаления начались при
небольшой раздаче.
"""
import argparse
import math
import os

# узлы шестигранника (i,j,k) -> 6 тетраэдров с согласованной диагональю
# (разбиение Куна по диагонали 0-6), объёмы положительны при правой
# системе (r, φ, z)
HEX = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
KUHN = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
TET_FACES = {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)}
HEX_FACES = {1: (0, 1, 2, 3), 2: (4, 7, 6, 5), 3: (0, 4, 5, 1),
             4: (1, 5, 6, 2), 5: (2, 6, 7, 3), 6: (3, 7, 4, 0)}


def tet_volume(p):
    a = [p[1][i] - p[0][i] for i in range(3)]
    b = [p[2][i] - p[0][i] for i in range(3)]
    c = [p[3][i] - p[0][i] for i in range(3)]
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0


def build(a):
    nodes, elems = {}, []
    nid = [0]

    def node(x, y, z):
        nid[0] += 1
        nodes[nid[0]] = (x, y, z)
        return nid[0]

    alpha = math.radians(a.sector)
    # --- кольцо ---
    ring = {}
    for i in range(a.nr + 1):
        r = a.ri + (a.ro - a.ri) * i / a.nr
        for j in range(a.nphi + 1):
            ph = alpha * j / a.nphi
            for k in range(a.nz + 1):
                z = 0.5 * a.h * k / a.nz
                ring[i, j, k] = node(r * math.cos(ph), r * math.sin(ph), z)
    eid = 0
    ring_el, inner_faces = [], []
    for i in range(a.nr):
        for j in range(a.nphi):
            for k in range(a.nz):
                hx = [ring[i + di, j + dj, k + dk] for di, dj, dk in HEX]
                for t in KUHN:
                    conn = [hx[q] for q in t]
                    if tet_volume([nodes[n] for n in conn]) < 0:
                        conn[1], conn[2] = conn[2], conn[1]
                    eid += 1
                    elems.append((eid, 'C3D4', conn))
                    ring_el.append(eid)
                    if i == 0:
                        for f, fn in TET_FACES.items():
                            ff = [conn[q] for q in fn]
                            if all(abs(math.hypot(*nodes[n][:2]) - a.ri) < 1e-9
                                   for n in ff):
                                inner_faces.append((eid, f))
    sets = {
        'RING_INNER': [ring[0, j, k] for j in range(a.nphi + 1) for k in range(a.nz + 1)],
        'PHI0': [ring[i, 0, k] for i in range(a.nr + 1) for k in range(a.nz + 1)],
        'PHI1': [ring[i, a.nphi, k] for i in range(a.nr + 1) for k in range(a.nz + 1)],
        'ZBOT': [ring[i, j, 0] for i in range(a.nr + 1) for j in range(a.nphi + 1)],
    }
    # --- сегмент ---
    seg_el, seg_faces, seg_nodes = [], [], []
    if a.mode == 'contact':
        beta = math.radians(a.sector - a.gap / 2.0)
        seg = {}
        ns_r, ns_p, ns_z = 2, max(2, a.nphi // 2), max(2, a.nz // 2)
        zt = 0.5 * a.h + a.seg_over
        for i in range(ns_r + 1):
            r = a.rseg_in + (a.ri - a.clear - a.rseg_in) * i / ns_r
            for j in range(ns_p + 1):
                ph = beta * j / ns_p
                for k in range(ns_z + 1):
                    seg[i, j, k] = node(r * math.cos(ph), r * math.sin(ph), zt * k / ns_z)
        seg_nodes = list(seg.values())
        for i in range(ns_r):
            for j in range(ns_p):
                for k in range(ns_z):
                    conn = [seg[i + di, j + dj, k + dk] for di, dj, dk in HEX]
                    eid += 1
                    elems.append((eid, 'C3D8', conn))
                    seg_el.append(eid)
                    if i == ns_r - 1:
                        seg_faces.append((eid, 4))   # грань 2-6-7-3: r = max
    return nodes, elems, ring_el, seg_el, inner_faces, seg_faces, seg_nodes, sets


def fmt_list(ids, per=10):
    out = []
    for i in range(0, len(ids), per):
        out.append(', '.join(str(x) for x in ids[i:i + per]))
    return '\n'.join(out)


def write(a, path):
    nodes, elems, ring_el, seg_el, inner_faces, seg_faces, seg_nodes, sets = build(a)
    alpha = math.radians(a.sector)
    L = []
    L.append('** Этап 0: проверка контакта с удалением элементов (mode=%s)' % a.mode)
    L.append('** сгенерировано tools/mksmoke.py %s' % ' '.join(
        '--%s %s' % (k, v) for k, v in sorted(vars(a).items()) if k != 'o'))
    L.append('*Node')
    for n, (x, y, z) in nodes.items():
        L.append('%d, %.9f, %.9f, %.9f' % (n, x, y, z))
    L.append('*Element, Type=C3D4, Elset=RING')
    for e, t, c in elems:
        if t == 'C3D4':
            L.append('%d, %s' % (e, ', '.join(map(str, c))))
    if seg_el:
        L.append('*Element, Type=C3D8, Elset=SEGMENT')
        for e, t, c in elems:
            if t == 'C3D8':
                L.append('%d, %s' % (e, ', '.join(map(str, c))))
    for name, ids in sets.items():
        L.append('*Nset, Nset=%s' % name)
        L.append(fmt_list(sorted(set(ids))))
    if seg_nodes:
        L.append('*Nset, Nset=SEGNODES')
        L.append(fmt_list(sorted(seg_nodes)))
    L.append('*Material, Name=E635')
    L.append('*Elastic')
    L.append('%g, %g' % (a.E, a.nu))
    L.append('*Plastic')
    for ep in (0.0, 0.002, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        L.append('%.1f, %.4f' % (a.K * (a.e0 + ep) ** a.n, ep))
    L.append('*Damage Initiation, Criterion=Ductile, Evolution=Displacement, Npoints=2')
    L.append('1.0, %g, 0.0, %g, 1.5, %g' % (a.uf, a.epsf, a.epsf))
    L.append('*Solid Section, Elset=RING, Material=E635')
    if seg_el:
        L.append('*Material, Name=STEEL')
        L.append('*Elastic')
        L.append('210000., 0.3')
        L.append('*Solid Section, Elset=SEGMENT, Material=STEEL')
        L.append('*Surface, Name=SLAVE, Type=Element')
        for e, f in inner_faces:
            L.append('%d, S%d' % (e, f))
        L.append('*Surface, Name=SLAVENODES, Type=Node')
        L.append('RING_INNER')
        L.append('*Surface, Name=MASTER, Type=Element')
        for e, f in seg_faces:
            L.append('%d, S%d' % (e, f))
        L.append('*Surface Interaction, Name=SI')
        L.append('*Surface Behavior, Pressure-Overclosure=Linear')
        L.append('%g' % (a.kpen,))
        if a.mu > 0:
            L.append('*Friction')
            L.append('%g, %g' % (a.mu, a.kstick))
        if a.ctype == 'n2s':
            L.append('*Contact Pair, Interaction=SI, Type=Node to Surface')
            L.append('SLAVENODES, MASTER')
        else:
            L.append('*Contact Pair, Interaction=SI, Type=Surface to Surface')
            L.append('SLAVE, MASTER')
    # уравнения для φ = α: −sin α·u_x + cos α·u_y = 0
    phi1 = set(sets['PHI1'])
    if not seg_el:
        # в режиме disp у внутренних узлов u_x, u_y заданы явно
        phi1 -= set(sets['RING_INNER'])
    phi1 = sorted(phi1)
    L.append('*Equation')
    s, c = math.sin(alpha), math.cos(alpha)
    for n in phi1:
        L.append('2')
        L.append('%d, 2, %.12f, %d, 1, %.12f' % (n, c, n, -s))
    L.append('*Step, Nlgeom, Inc=%d' % a.maxinc)
    L.append('*Static')
    L.append('%g, 1., %g, %g' % (a.dt0, a.dtmin, a.dtmax))
    L.append('*Boundary')
    L.append('PHI0, 2, 2, 0.')
    L.append('ZBOT, 3, 3, 0.')
    if seg_el:
        L.append('SEGNODES, 1, 1, %g' % a.ur)
        L.append('SEGNODES, 2, 3, 0.')
    else:
        # радиальное перемещение внутренних узлов кольца
        for n in sorted(set(sets['RING_INNER'])):
            x, y, _ = nodes[n]
            r = math.hypot(x, y)
            L.append('%d, 1, 1, %.9f' % (n, a.ur * x / r))
            if n not in set(sets['PHI0']):
                L.append('%d, 2, 2, %.9f' % (n, a.ur * y / r))
    L.append('*Output, Frequency=%d' % a.outfreq)
    L.append('*Node File')
    L.append('U, RF')
    L.append('*El File')
    L.append('S, E, PEEQ, SDV')
    if seg_el:
        L.append('*Node Print, Nset=SEGNODES, Totals=Only')
        L.append('RF')
    else:
        L.append('*Node Print, Nset=RING_INNER, Totals=Only')
        L.append('RF')
    L.append('*End Step')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote %s: %d nodes, %d C3D4, %d C3D8, %d slave faces' % (
        path, len(nodes), len(ring_el), len(seg_el), len(inner_faces)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', required=True)
    ap.add_argument('--mode', choices=('contact', 'disp'), default='contact')
    ap.add_argument('--ri', type=float, default=5.5)
    ap.add_argument('--ro', type=float, default=6.3)
    ap.add_argument('--h', type=float, default=2.0, help='полная высота кольца, мм')
    ap.add_argument('--sector', type=float, default=22.5)
    ap.add_argument('--gap', type=float, default=2.0, help='угловой зазор между сегментами, град')
    ap.add_argument('--nr', type=int, default=4)
    ap.add_argument('--nphi', type=int, default=12)
    ap.add_argument('--nz', type=int, default=5)
    ap.add_argument('--rseg-in', dest='rseg_in', type=float, default=4.5)
    ap.add_argument('--clear', type=float, default=0.0, help='радиальный зазор сегмент–кольцо, мм')
    ap.add_argument('--seg-over', dest='seg_over', type=float, default=0.5,
                    help='на сколько сегмент выше кольца (над торцом), мм')
    ap.add_argument('--E', type=float, default=90500.)
    ap.add_argument('--nu', type=float, default=0.35)
    ap.add_argument('--K', type=float, default=653., help='Swift: σ = K (e0 + εp)^n')
    ap.add_argument('--e0', type=float, default=0.007)
    ap.add_argument('--n', type=float, default=0.082)
    ap.add_argument('--epsf', type=float, default=0.10, help='ε_f (нарочно низкая для проверки)')
    ap.add_argument('--uf', type=float, default=0.03, help='u_f, мм')
    ap.add_argument('--mu', type=float, default=0.1)
    ap.add_argument('--ctype', choices=('s2s', 'n2s'), default='s2s',
                    help='контакт surface-to-surface или node-to-surface')
    ap.add_argument('--kpen', type=float, default=1.0e6, help='жёсткость штрафа контакта, МПа/мм')
    ap.add_argument('--kstick', type=float, default=1.0e5, help='жёсткость прилипания, МПа/мм')
    ap.add_argument('--ur', type=float, default=0.6, help='радиальное перемещение сегмента, мм')
    ap.add_argument('--dt0', type=float, default=0.01)
    ap.add_argument('--dtmin', type=float, default=1e-7)
    ap.add_argument('--dtmax', type=float, default=0.02)
    ap.add_argument('--maxinc', type=int, default=20000)
    ap.add_argument('--outfreq', type=int, default=10)
    a = ap.parse_args()
    write(a, a.o)


if __name__ == '__main__':
    main()
