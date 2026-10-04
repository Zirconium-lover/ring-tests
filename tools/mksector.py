#!/usr/bin/env python3
"""Колода этапа 1: сектор кольца направляющего канала на сегментной оправке.

    python3 tools/mksector.py -o runs/stage1/decks/H5_mu0.05.inp --H 5 --mu 0.05

Область — 1/16 кольца: φ от 0 (середина сегмента) до 22.5° (середина
зазора при 8 сегментах) × половина высоты (z от 0 до H/2). Плоскости
симметрии: φ = 0 (u_y = 0), φ = 22.5° (уравнение −sin α·u_x + cos α·u_y = 0),
z = 0 (u_z = 0). Торец z = H/2 свободен.

Сегмент — жёсткий: слой C3D8, все узлы ведутся перемещением u_x = u_r(t)
вдоль биссектрисы сегмента (ось x), что эквивалентно ходу конуса
u_z = u_r / tan(θ/2). Наружная поверхность сегмента — цилиндр R_i − clear,
край сегмента при φ_e = 22.5° − (kerf/2)/R_i, по высоте сегмент выступает
за торец кольца на seg_over.

Контакт node-to-surface (ведомые — узлы внутренней поверхности кольца,
ведущая — наружная грань сегмента), трение μ. Surface-to-surface не
используется: в ccx-arch2 c75ad9b он падает при удалении элементов
(notes/stage0_solver_check.md).

Сетка кольца — структурная по (r, φ, z); --elem C3D4 режет каждый
шестигранник на тетраэдры (как на этапе разрушения), --elem C3D8I
оставляет шестигранники (контроль объёмного запирания C3D4). Разбиение
--tet kuhn — 6 тетраэдров вокруг диагонали; --tet x24 — 24 тетраэдра через
центры граней и центр шестигранника (каждая грань — 4 треугольника вокруг
своего центра, каждый треугольник с центром шестигранника даёт тетраэдр):
больше узлов на тетраэдр и нет выделенной диагонали, меньше запирание.

Материал — Э635, изотропная пластичность (кривая B, notes/literature_e635.md):
σ = K·(e0 + ε_p)^n, K = 653 МПа, n = 0.082, e0 подобран так, что
σ(ε_p = 0.002) = σ_0.2 = 459 МПа. E = 90.5 ГПа, ν = 0.35.
Повреждение по умолчанию выключено (этап 1).
"""
import argparse
import math
import os

HEX = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
KUHN = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
EPS_TABLE = (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15,
             0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)


def tet_volume(p):
    a = [p[1][i] - p[0][i] for i in range(3)]
    b = [p[2][i] - p[0][i] for i in range(3)]
    c = [p[3][i] - p[0][i] for i in range(3)]
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0


def swift_e0(K, n, s02):
    """e0 такое, что K (e0 + 0.002)^n = s02."""
    return (s02 / K) ** (1.0 / n) - 0.002


class Mesh:
    def __init__(self):
        self.nodes = {}
        self.elems = []          # (id, type, conn)

    def node(self, x, y, z):
        n = len(self.nodes) + 1
        self.nodes[n] = (x, y, z)
        return n


def build(a):
    m = Mesh()
    alpha = math.radians(a.sector)
    t = a.ro - a.ri
    nr = max(2, round(t / a.size))
    nphi = max(4, round(alpha * a.ri / a.size))
    nz = max(2, round(0.5 * a.H / a.size))
    a.nr, a.nphi, a.nz = nr, nphi, nz
    quad = a.elem == 'C3D20R'
    x24 = a.elem == 'C3D4' and a.tet == 'x24'
    f = 2 if quad or x24 else 1   # узлы на удвоенной сетке: середины рёбер / центры
    ring = {}
    for I in range(f * nr + 1):
        r = a.ri + t * I / (f * nr)
        for J in range(f * nphi + 1):
            ph = alpha * J / (f * nphi)
            for K in range(f * nz + 1):
                odd = (I % 2) + (J % 2) + (K % 2)
                if quad and odd > 1:
                    continue      # центры граней и элементов у C3D20R не нужны
                if x24 and odd == 1:
                    continue      # середины рёбер у x24 не нужны
                ring[I, J, K] = m.node(r * math.cos(ph), r * math.sin(ph),
                                       0.5 * a.H * K / (f * nz))
    # C3D20R: углы как у C3D8, затем середины рёбер 1-2, 2-3, 3-4, 4-1,
    # 5-6, 6-7, 7-8, 8-5, 1-5, 2-6, 3-7, 4-8 (в удвоенных индексах)
    Q20 = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
           (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
           (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)]
    eid = 0
    sec_z0, sec_ztop = [], []
    for i in range(nr):
        for j in range(nphi):
            for k in range(nz):
                first = eid + 1
                if quad:
                    eid += 1
                    m.elems.append((eid, 'C3D20R', [ring[2 * i + di, 2 * j + dj, 2 * k + dk]
                                                    for di, dj, dk in Q20]))
                elif x24:
                    B = ring[2 * i + 1, 2 * j + 1, 2 * k + 1]
                    for fc in HEX_FACES.values():
                        cs = [HEX[q] for q in fc]
                        F = ring[tuple(2 * (i, j, k)[d] + sum(c[d] for c in cs) // 2
                                       for d in range(3))]
                        cn = [ring[2 * i + 2 * c[0], 2 * j + 2 * c[1], 2 * k + 2 * c[2]]
                              for c in cs]
                        for q in range(4):
                            conn = [B, F, cn[q], cn[(q + 1) % 4]]
                            if tet_volume([m.nodes[n] for n in conn]) < 0:
                                conn[2], conn[3] = conn[3], conn[2]
                            eid += 1
                            m.elems.append((eid, 'C3D4', conn))
                else:
                    hx = [ring[i + di, j + dj, k + dk] for di, dj, dk in HEX]
                    if a.elem == 'C3D4':
                        for tt in KUHN:
                            conn = [hx[q] for q in tt]
                            if tet_volume([m.nodes[n] for n in conn]) < 0:
                                conn[1], conn[2] = conn[2], conn[1]
                            eid += 1
                            m.elems.append((eid, 'C3D4', conn))
                    else:
                        eid += 1
                        m.elems.append((eid, a.elem, hx))
                if k == 0:
                    sec_z0 += list(range(first, eid + 1))
                if k == nz - 1:
                    sec_ztop += list(range(first, eid + 1))
    nring_el = eid
    m.elsets = {'SEC_Z0': sec_z0, 'SEC_ZTOP': sec_ztop}
    NR, NP, NZ = f * nr, f * nphi, f * nz
    g = lambda I, J, K: ring.get((I, J, K))
    pick = lambda cond: [n for (I, J, K), n in ring.items() if cond(I, J, K)]
    sets = {
        'RING_INNER': pick(lambda I, J, K: I == 0),
        'RING_OUTER': pick(lambda I, J, K: I == NR),
        'PHI0': pick(lambda I, J, K: J == 0),
        'PHI1': pick(lambda I, J, K: J == NP),
        'ZBOT': pick(lambda I, J, K: K == 0),
        'ZTOP': pick(lambda I, J, K: K == NZ),
        # линии для постобработки: внутренняя и наружная поверхность
        # в середине высоты и на торце, по всей дуге
        'IN_Z0': pick(lambda I, J, K: I == 0 and K == 0),
        'OUT_Z0': pick(lambda I, J, K: I == NR and K == 0),
        'IN_ZTOP': pick(lambda I, J, K: I == 0 and K == NZ),
        'OUT_ZTOP': pick(lambda I, J, K: I == NR and K == NZ),
    }
    # --- жёсткий сегмент ---
    phe = alpha - (0.5 * a.kerf) / a.ri
    a.phi_edge_deg = math.degrees(phe)
    rs_out = a.ri - a.clear
    rs_in = rs_out - a.seg_t
    zt = 0.5 * a.H + a.seg_over
    ns_p = max(8, round(phe * rs_out / a.seg_size))
    ns_z = max(2, round(zt / max(a.seg_size, 0.25)))
    seg = {}
    for i in range(2):
        r = rs_in + (rs_out - rs_in) * i
        for j in range(ns_p + 1):
            ph = phe * j / ns_p
            for k in range(ns_z + 1):
                seg[i, j, k] = m.node(r * math.cos(ph), r * math.sin(ph), zt * k / ns_z)
    seg_faces = []
    for j in range(ns_p):
        for k in range(ns_z):
            conn = [seg[di, j + dj, k + dk] for di, dj, dk in HEX]
            eid += 1
            m.elems.append((eid, 'C3D8', conn))
            seg_faces.append((eid, 4))     # грань 2-6-7-3: r = max
    sets['SEGNODES'] = list(seg.values())
    return m, sets, seg_faces, nring_el


TET_FACES = {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)}
HEX_FACES = {1: (0, 1, 2, 3), 2: (4, 7, 6, 5), 3: (0, 4, 5, 1), 4: (1, 5, 6, 2),
             5: (2, 6, 7, 3), 6: (3, 7, 4, 0)}


def inner_faces(m, sets, a):
    """Грани элементов кольца, все угловые узлы которых на R_i."""
    inner = set(sets['RING_INNER'])
    out = []
    for e, t, c in m.elems:
        if t != a.elem:
            continue
        faces = TET_FACES if t == 'C3D4' else HEX_FACES
        for fid, fn in faces.items():
            if all(c[q] in inner for q in fn):
                out.append((e, fid))
    return out


def lines(ids, per=12):
    ids = sorted(set(ids))
    return [', '.join(str(x) for x in ids[i:i + per]) for i in range(0, len(ids), per)]


def write(a):
    m, sets, seg_faces, nring = build(a)
    alpha = math.radians(a.sector)
    e0 = swift_e0(a.K, a.n, a.s02)
    L = []
    L.append('** Этап 1: сектор кольца на сегментной оправке (tools/mksector.py)')
    L.append('** ' + ' '.join('%s=%s' % (k, v) for k, v in sorted(vars(a).items()) if k != 'o'))
    L.append('*Node')
    L += ['%d, %.9f, %.9f, %.9f' % (n, *xyz) for n, xyz in m.nodes.items()]
    L.append('*Element, Type=%s, Elset=RING' % a.elem)
    for e, t, c in m.elems:
        if t != a.elem:
            continue
        # не больше 16 значений в строке: у C3D20R хвост на строке продолжения
        if len(c) > 15:
            L.append('%d, %s,' % (e, ', '.join(map(str, c[:15]))))
            L.append(', '.join(map(str, c[15:])))
        else:
            L.append('%d, %s' % (e, ', '.join(map(str, c))))
    L.append('*Element, Type=C3D8, Elset=SEGMENT')
    L += ['%d, %s' % (e, ', '.join(map(str, c))) for e, t, c in m.elems
          if t == 'C3D8' and e > nring]
    for name, ids in sets.items():
        L.append('*Nset, Nset=%s' % name)
        L += lines(ids)
    for name, ids in m.elsets.items():
        L.append('*Elset, Elset=%s' % name)
        L += lines(ids)
    L.append('*Material, Name=E635')
    L.append('*Elastic')
    L.append('%g, %g' % (a.E, a.nu))
    L.append('*Plastic')
    L += ['%.2f, %.4f' % (a.K * (e0 + ep) ** a.n, ep) for ep in EPS_TABLE]
    if a.damage:
        L.append('*Damage Initiation, Criterion=Ductile, Evolution=Displacement, '
                 'Npoints=%d' % (len(a.epsf_eta) // 2))
        L.append('1.0, %g, %s' % (a.uf, ', '.join('%g' % v for v in a.epsf_eta)))
    L.append('*Solid Section, Elset=RING, Material=E635')
    L.append('*Material, Name=SEGMENT_STEEL')
    L.append('*Elastic')
    L.append('210000., 0.3')
    L.append('*Solid Section, Elset=SEGMENT, Material=SEGMENT_STEEL')
    L.append('*Surface, Name=RING_IN_NODES, Type=Node')
    L.append('RING_INNER')
    if a.ctype == 's2s':
        L.append('*Surface, Name=RING_IN_FACES, Type=Element')
        L += ['%d, S%d' % (e, f) for e, f in inner_faces(m, sets, a)]
    L.append('*Surface, Name=SEG_OUT, Type=Element')
    L += ['%d, S%d' % (e, f) for e, f in seg_faces]
    L.append('*Surface Interaction, Name=SEG_RING')
    L.append('*Surface Behavior, Pressure-Overclosure=Linear')
    L.append('%g' % a.kpen)
    if a.mu > 0:
        L.append('*Friction')
        L.append('%g, %g' % (a.mu, a.kstick))
    if a.ctype == 's2s':
        # только без удаления элементов (этап 0: s2s + удаление падает)
        L.append('*Contact Pair, Interaction=SEG_RING, Type=Surface to Surface')
        L.append('RING_IN_FACES, SEG_OUT')
    else:
        L.append('*Contact Pair, Interaction=SEG_RING, Type=Node to Surface')
        L.append('RING_IN_NODES, SEG_OUT')
    L.append('*Equation')
    s, c = math.sin(alpha), math.cos(alpha)
    for n in sorted(set(sets['PHI1'])):
        L.append('2')
        L.append('%d, 2, %.12f, %d, 1, %.12f' % (n, c, n, -s))
    L.append('*Step, Nlgeom, Inc=%d' % a.maxinc)
    L.append('*Static')
    L.append('%g, 1., %g, %g' % (a.dt0, a.dtmin, a.dtmax))
    L.append('*Boundary')
    L.append('PHI0, 2, 2, 0.')
    L.append('ZBOT, 3, 3, 0.')
    L.append('SEGNODES, 1, 1, %g' % a.ur)
    L.append('SEGNODES, 2, 3, 0.')
    L.append('*Node Print, Nset=SEGNODES, Totals=Only, Frequency=%d' % a.printfreq)
    L.append('RF')
    for name in ('IN_Z0', 'OUT_Z0', 'IN_ZTOP', 'OUT_ZTOP'):
        L.append('*Node Print, Nset=%s, Frequency=%d' % (name, a.printfreq))
        L.append('U')
    # слои элементов в середине высоты и у торца (вся толщина и дуга):
    # напряжения и PEEQ для трёхосности и σ_zz в опасной точке
    for name in ('SEC_Z0', 'SEC_ZTOP'):
        L.append('*El Print, Elset=%s, Frequency=%d' % (name, a.printfreq))
        L.append('S, PEEQ')
    # В CalculiX FREQUENCY у *NODE PRINT и *NODE FILE пишется в одну
    # переменную (jout(1)): частота общая для .dat и .frd, побеждает
    # последняя. Поэтому одна частота, а .frd облегчён до U и PEEQ.
    L.append('*Node File, Frequency=%d' % a.printfreq)
    L.append('U')
    L.append('*El File, Frequency=%d' % a.printfreq)
    L.append('PEEQ' + (', SDV' if a.damage else ''))
    L.append('*End Step')
    os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True)
    with open(a.o, 'w') as f:
        f.write('\n'.join(L) + '\n')
    nseg = len(m.elems) - nring
    print('wrote %s: H=%g mm, %d nodes, %d %s (nr=%d nphi=%d nz=%d), %d C3D8 segment, '
          'phi_edge=%.3f deg, e0=%.4f' % (a.o, a.H, len(m.nodes), nring, a.elem,
                                          a.nr, a.nphi, a.nz, nseg, a.phi_edge_deg, e0))
    return m, sets


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', required=True)
    ap.add_argument('--H', type=float, default=5.0, help='полная высота кольца, мм')
    ap.add_argument('--ri', type=float, default=5.5)
    ap.add_argument('--ro', type=float, default=6.3)
    ap.add_argument('--sector', type=float, default=22.5, help='180/число сегментов, град')
    ap.add_argument('--size', type=float, default=0.1, help='размер элемента кольца, мм')
    ap.add_argument('--elem', choices=('C3D4', 'C3D8I', 'C3D8', 'C3D20R'), default='C3D4')
    ap.add_argument('--tet', choices=('kuhn', 'x24'), default='kuhn',
                    help='разбиение шестигранника на C3D4')
    ap.add_argument('--kerf', type=float, default=0.2,
                    help='ширина пропила между сегментами по R_i, мм')
    ap.add_argument('--clear', type=float, default=0.0, help='радиальный зазор сегмент–кольцо, мм')
    ap.add_argument('--seg-t', dest='seg_t', type=float, default=0.5, help='толщина слоя сегмента, мм')
    ap.add_argument('--seg-over', dest='seg_over', type=float, default=1.0,
                    help='выступ сегмента за торец кольца, мм')
    ap.add_argument('--seg-size', dest='seg_size', type=float, default=0.05,
                    help='размер фасетки сегмента по дуге, мм')
    ap.add_argument('--E', type=float, default=90500.)
    ap.add_argument('--nu', type=float, default=0.35)
    ap.add_argument('--K', type=float, default=653.)
    ap.add_argument('--n', type=float, default=0.082)
    ap.add_argument('--s02', type=float, default=459.)
    ap.add_argument('--damage', action='store_true', help='включить вязкое повреждение')
    ap.add_argument('--uf', type=float, default=0.05)
    ap.add_argument('--epsf-eta', dest='epsf_eta', type=float, nargs='+',
                    default=[0.0, 0.9, 0.33, 0.8, 0.6, 0.55, 1.0, 0.35, 1.5, 0.25])
    ap.add_argument('--mu', type=float, default=0.05)
    ap.add_argument('--ctype', choices=('n2s', 's2s'), default='n2s',
                    help='контакт node-to-surface (по умолчанию) или surface-to-surface')
    ap.add_argument('--kpen', type=float, default=1.0e6, help='штраф контакта, МПа/мм')
    ap.add_argument('--kstick', type=float, default=1.0e5, help='жёсткость прилипания, МПа/мм')
    ap.add_argument('--ur', type=float, default=2.0, help='радиальное перемещение сегмента, мм')
    ap.add_argument('--dt0', type=float, default=0.005)
    ap.add_argument('--dtmin', type=float, default=1e-6)
    ap.add_argument('--dtmax', type=float, default=0.005)
    ap.add_argument('--maxinc', type=int, default=20000)
    ap.add_argument('--printfreq', type=int, default=5)
    return ap


if __name__ == '__main__':
    write(parser().parse_args())
