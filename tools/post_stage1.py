#!/usr/bin/env python3
"""Постобработка расчёта этапа 1 (колода tools/mksector.py).

    python3 tools/post_stage1.py runs/stage1/H5_mu0.05 [--plot]

Читает m.inp (координаты, наборы) и m.dat (печать каждые N инкрементов):
  - сила на полусегмент F_x (сумма RF узлов сегмента);
  - перемещения линий узлов IN_Z0, OUT_Z0, IN_ZTOP, OUT_ZTOP (по всей дуге);
  - S и PEEQ элементов слоёв SEC_Z0 (середина высоты) и SEC_ZTOP (у торца).

Пишет в RUN/post/:
  timeseries.csv — по моментам: u_r, номинальная ε = u_r/R_i, F_x, F_z (сила
                   на конусе), средняя и экстремальные окружные деформации
                   внутренней поверхности в середине и у торца, утонение,
                   опасная точка (max PEEQ в слое z = 0): φ, r, PEEQ, η,
                   σ_zz/σ_θθ, σ_rr/σ_θθ; то же у торца;
  profiles.json  — ε_θ(φ) и t/t0(φ) в каждый момент (для сравнения серий);
  summary.json   — итог: u_r и ε при максимуме силы, F_max, время счёта и т. п.

Окружная деформация — логарифмическая, по длине хорды между соседними узлами
линии: ε_θ = ln(l/l0). Толщина — расстояние между внутренним и наружным узлом
с тем же φ. Компоненты напряжения переводятся в цилиндрические по углу φ
центра элемента в исходном положении (поворот материала по φ — доли градуса).
"""
import argparse
import csv
import json
import math
import os
import re
import sys

TANH = math.tan(math.radians(10.0))     # полуугол конуса 20°
NSECT = 32                              # 16 секторов по дуге × 2 половины по высоте


def read_deck(path):
    nodes, elems, sets, esets = {}, {}, {}, {}
    cur = name = None
    ur = ri = None
    tet, size = 'kuhn', 0.1
    for line in open(path):
        s = line.strip()
        if s.startswith('**'):
            m = re.search(r'\bur=([0-9.eE+-]+)', s)
            if m:
                ur = float(m.group(1))
            m = re.search(r'\bri=([0-9.eE+-]+)', s)
            if m:
                ri = float(m.group(1))
            m = re.search(r'\bH=([0-9.eE+-]+)', s)
            if m:
                H = float(m.group(1))
            m = re.search(r'\bmu=([0-9.eE+-]+)', s)
            if m:
                mu = float(m.group(1))
            m = re.search(r'\belem=([A-Z0-9]+)', s)
            if m:
                et = m.group(1)
            m = re.search(r'\btet=(\w+)', s)
            if m:
                tet = m.group(1)
            m = re.search(r'\bsize=([0-9.eE+-]+)', s)
            if m:
                size = float(m.group(1))
            continue
        if s.startswith('*'):
            kw = s.lower()
            cur = None
            if kw.startswith('*node') and 'print' not in kw and 'file' not in kw:
                cur = 'node'
            elif kw.startswith('*element'):
                cur = 'elem'
            elif kw.startswith('*nset'):
                cur, name = 'nset', re.search(r'nset=(\w+)', kw).group(1).upper()
                sets[name] = []
            elif kw.startswith('*elset'):
                cur, name = 'elset', re.search(r'elset=(\w+)', kw).group(1).upper()
                esets[name] = []
            continue
        if not s:
            continue
        v = [x for x in s.split(',') if x.strip()]
        if cur == 'node':
            nodes[int(v[0])] = tuple(float(x) for x in v[1:4])
        elif cur == 'elem':
            elems[int(v[0])] = [int(x) for x in v[1:]]
        elif cur == 'nset':
            sets[name] += [int(x) for x in v]
        elif cur == 'elset':
            esets[name] += [int(x) for x in v]
    # метка сетки: тип элемента, разбиение на тетраэдры (если не kuhn), шаг (если не 0.1)
    label = et + ('-x24' if et == 'C3D4' and tet == 'x24' else '')
    label += '-h%g' % size if abs(size - 0.1) > 1e-9 else ''
    return dict(nodes=nodes, elems=elems, sets=sets, esets=esets, ur=ur, ri=ri,
                H=H, mu=mu, elem=label)


HDR = re.compile(r'^\s*(total force|displacements|stresses|equivalent plastic strain)'
                 r'.*?for set (\w+) and time\s+([0-9.eE+-]+)')


def read_dat(path):
    """{time: {(kind, set): rows}}; rows — списки чисел."""
    out, key = {}, None
    for line in open(path):
        m = HDR.match(line)
        if m:
            kind = {'total force': 'F', 'displacements': 'U', 'stresses': 'S',
                    'equivalent plastic strain': 'PE'}[m.group(1)]
            t = float(m.group(3))
            key = (kind, m.group(2).upper())
            out.setdefault(t, {})[key] = []
            cur = out[t][key]
            continue
        s = line.split()
        if key is None or not s:
            continue
        try:
            cur.append([float(x) for x in s])
        except ValueError:
            key = None
    return out


def cyl_angle(p):
    return math.atan2(p[1], p[0])


def line_order(deck, name):
    ids = sorted(set(deck['sets'][name]), key=lambda n: cyl_angle(deck['nodes'][n]))
    return ids


def hoop_profile(deck, order, disp):
    """φ середины хорд (град) и ε_θ = ln(l/l0) по хордам; средняя по линии."""
    P = [deck['nodes'][n] for n in order]
    p = [tuple(P[i][k] + disp[n][k] for k in range(3)) for i, n in enumerate(order)]
    phis, eps, L0, L = [], [], 0.0, 0.0
    for i in range(len(order) - 1):
        l0 = math.dist(P[i], P[i + 1])
        l = math.dist(p[i], p[i + 1])
        L0 += l0
        L += l
        phis.append(math.degrees(0.5 * (cyl_angle(P[i]) + cyl_angle(P[i + 1]))))
        eps.append(math.log(l / l0))
    return phis, eps, math.log(L / L0)


def thickness(deck, oin, oout, din, dout):
    phis, tr = [], []
    for a, b in zip(oin, oout):
        A, B = deck['nodes'][a], deck['nodes'][b]
        pa = [A[k] + din[a][k] for k in range(3)]
        pb = [B[k] + dout[b][k] for k in range(3)]
        phis.append(math.degrees(cyl_angle(A)))
        tr.append(math.dist(pa, pb) / math.dist(A, B))
    return phis, tr


def centroid(deck, e):
    c = deck['elems'][e]
    pts = [deck['nodes'][n] for n in c]
    return tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))


def stress_cyl(s, phi):
    sxx, syy, szz, sxy, sxz, syz = s
    c, sn = math.cos(phi), math.sin(phi)
    srr = c * c * sxx + sn * sn * syy + 2 * c * sn * sxy
    stt = sn * sn * sxx + c * c * syy - 2 * c * sn * sxy
    vm = math.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))
    sm = (sxx + syy + szz) / 3.0
    return srr, stt, szz, vm, sm


def hot_spot(deck, frame, eset, cents):
    """Опасная точка — точка интегрирования с максимальной PEEQ в слое.

    У C3D4 одна точка на элемент, у C3D8I — восемь; ключ (элемент, точка),
    угол φ берётся по центру элемента."""
    S = {(int(r[0]), int(r[1])): r[2:8] for r in frame.get(('S', eset), [])}
    PE = {(int(r[0]), int(r[1])): r[2] for r in frame.get(('PE', eset), [])}
    if not PE:
        return None
    key = max(PE, key=lambda k: PE[k])
    e = key[0]
    x, y, z = cents[e]
    phi = math.atan2(y, x)
    srr, stt, szz, vm, sm = stress_cyl(S[key], phi)
    return dict(elem=e, ip=key[1], phi=math.degrees(phi), r=math.hypot(x, y), peeq=PE[key],
                eta=sm / vm if vm > 0 else 0.0, szz_stt=szz / stt if stt else 0.0,
                srr_stt=srr / stt if stt else 0.0, vm=vm, stt=stt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    deck = read_deck(os.path.join(a.run, 'm.inp'))
    dat = read_dat(os.path.join(a.run, 'm.dat'))
    times = sorted(dat)
    if not times:
        sys.exit('no output in m.dat yet')
    ri, ur = deck['ri'], deck['ur']
    o = {k: line_order(deck, k) for k in ('IN_Z0', 'OUT_Z0', 'IN_ZTOP', 'OUT_ZTOP')}
    cents = {e: centroid(deck, e) for k in ('SEC_Z0', 'SEC_ZTOP') for e in deck['esets'][k]}
    rows, prof = [], {}
    for t in times:
        fr = dat[t]
        if ('F', 'SEGNODES') not in fr:
            continue
        fx = fr[('F', 'SEGNODES')][0][0]
        u = {k: {int(r[0]): r[1:4] for r in fr.get(('U', k), [])} for k in o}
        if not all(u[k] for k in o):
            continue
        row = dict(time=t, ur=t * ur, eps_nom=t * ur / ri, Fx=fx, Fz=NSECT * fx * TANH)
        pr = {}
        for k in o:
            ph, ep, em = hoop_profile(deck, o[k], u[k])
            row['epsm_' + k] = em
            row['epsmax_' + k] = max(ep)
            row['epsmin_' + k] = min(ep)
            row['phimax_' + k] = ph[ep.index(max(ep))]
            pr[k] = dict(phi=ph, eps=ep)
        for zl in ('Z0', 'ZTOP'):
            ph, tr = thickness(deck, o['IN_' + zl], o['OUT_' + zl], u['IN_' + zl], u['OUT_' + zl])
            row['tmin_' + zl] = min(tr)
            row['phitmin_' + zl] = ph[tr.index(min(tr))]
            pr['T_' + zl] = dict(phi=ph, t=tr)
        for eset, tag in (('SEC_Z0', 'z0'), ('SEC_ZTOP', 'ztop')):
            hs = hot_spot(deck, fr, eset, cents)
            if hs:
                for kk in ('phi', 'r', 'peeq', 'eta', 'szz_stt', 'srr_stt'):
                    row['hs_%s_%s' % (tag, kk)] = hs[kk]
        rows.append(row)
        prof['%.6f' % t] = pr
    os.makedirs(os.path.join(a.run, 'post'), exist_ok=True)
    with open(os.path.join(a.run, 'post', 'timeseries.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in rows[-1].keys()})
    with open(os.path.join(a.run, 'post', 'profiles.json'), 'w') as f:
        json.dump(prof, f)
    imax = max(range(len(rows)), key=lambda i: rows[i]['Fx'])
    prov = {}
    pp = os.path.join(a.run, 'provenance.txt')
    if os.path.exists(pp):
        for line in open(pp):
            if '=' in line and not line.startswith('CCX_'):
                k, v = line.strip().split('=', 1)
                prov[k] = v
    summ = dict(run=os.path.basename(os.path.normpath(a.run)), H=deck['H'], mu=deck['mu'],
                elem=deck['elem'], frames=len(rows), last_ur=rows[-1]['ur'],
                last_eps_nom=rows[-1]['eps_nom'],
                Fx_max=rows[imax]['Fx'], Fz_max=rows[imax]['Fz'],
                ur_at_Fmax=rows[imax]['ur'], eps_nom_at_Fmax=rows[imax]['eps_nom'],
                hs_at_Fmax={k[6:]: rows[imax][k] for k in rows[imax] if k.startswith('hs_z0_')},
                return_code=prov.get('return_code'), wall_seconds=prov.get('wall_seconds'))
    with open(os.path.join(a.run, 'post', 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
