"""Одна пластинка гидрида в матрице циркония: МКЭ (CalculiX), плоская деформация.

Четверть области [0, D]² с симметрией по x = 0 и y = 0; пластинка — прямоугольник
[0, a] × [0, h/2] (полудлина a, толщина h, нормаль по y). Несоответствие задаётся
ортотропным «тепловым» расширением пластинки (α_x = ε_t, α_y = ε_n, α_z = ε_t) при ΔT = 1,
нарастающим за шаг, — как собственная деформация в автомате. Плоская деформация —
один слой C3D20R с u_z = 0 во всех узлах. Матрица упругая или упругопластическая
(Мизес, изотропное упрочнение), пластинка упругая. Единицы: мкм, МПа.
"""
import os
import re
import subprocess
import numpy as np

CCX = os.environ.get("CCX", "/home/user/ccx-arch2/src/ccx_2.23_pardiso")
EPS_N, EPS_T = 0.0720, 0.0458


def mesh_quarter(D=40.0, a=2.5, h=0.6, s_min=0.04, s_win=0.15, win=12.0, s_max=2.5, shape="rect"):
    """Сетка четверти: 8-узловые четырёхугольники. Возвращает узлы (N×2), элементы (M×8),
    флаг пластинки для элементов."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("q")
    occ = gmsh.model.occ
    dom = occ.addRectangle(0, 0, 0, D, D)
    if shape == "rect":
        pl = occ.addRectangle(0, 0, 0, a, h / 2)
    else:                                   # линза: эллипс с полуосями a и h/2, четверть
        el = occ.addDisk(0, 0, 0, a, h / 2)
        box = occ.addRectangle(0, 0, 0, a * 1.1, h)
        pl = occ.intersect([(2, el)], [(2, box)])[0][0][1]
    out, _ = occ.fragment([(2, dom)], [(2, pl)])
    occ.synchronize()
    surfs = [s for d, s in gmsh.model.getEntities(2)]
    # какая поверхность — пластинка: по центру масс
    plate_tag = min(surfs, key=lambda s: np.hypot(*occ.getCenterOfMass(2, s)[:2]))
    plate_curves = [c for d, c in gmsh.model.getBoundary([(2, plate_tag)], oriented=False)]
    f = gmsh.model.mesh.field
    f.add("Distance", 1); f.setNumbers(1, "CurvesList", plate_curves); f.setNumber(1, "Sampling", 400)
    f.add("Threshold", 2); f.setNumber(2, "InField", 1)
    f.setNumber(2, "SizeMin", s_min); f.setNumber(2, "SizeMax", s_win)
    f.setNumber(2, "DistMin", 0.05); f.setNumber(2, "DistMax", 1.5); f.setNumber(2, "StopAtDistMax", 1)
    f.add("Box", 3); f.setNumber(3, "VIn", s_win); f.setNumber(3, "VOut", s_max)
    f.setNumber(3, "XMin", 0); f.setNumber(3, "XMax", win); f.setNumber(3, "YMin", 0); f.setNumber(3, "YMax", win)
    f.setNumber(3, "Thickness", 8.0)
    f.add("Min", 4); f.setNumbers(4, "FieldsList", [2, 3])
    f.add("MathEval", 5); f.setString(5, "F", "2*F4")
    f.setAsBackgroundMesh(5)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)        # только четырёхугольники
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    idx = {t: i for i, t in enumerate(tags)}
    nodes = xyz.reshape(-1, 3)[:, :2]
    E, flag = [], []
    for s in surfs:
        et, etags, enodes = gmsh.model.mesh.getElements(2, s)
        for typ, en in zip(et, enodes):
            name, dim, order, nn, *_ = gmsh.model.mesh.getElementProperties(typ)
            assert nn == 8, name
            conn = np.array([idx[t] for t in en]).reshape(-1, 8)
            E.append(conn); flag.append(np.full(len(conn), s == plate_tag))
    gmsh.finalize()
    E = np.concatenate(E); flag = np.concatenate(flag)
    # обход против часовой стрелки
    p = nodes[E[:, :4]]
    area2 = np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - np.roll(p[:, :, 0], -1, 1) * p[:, :, 1], 1)
    neg = area2 < 0
    E[neg] = E[neg][:, [0, 3, 2, 1, 7, 6, 5, 4]]
    # убрать неиспользуемые узлы
    used = np.unique(E)
    remap = -np.ones(len(nodes), int); remap[used] = np.arange(len(used))
    return nodes[used], remap[E], flag


def write_inp(path, nodes, quads, plate, t=0.1, E=90e3, nu=0.34, E_h=None, nu_h=None,
              sy=None, hard=((0.0, 0.0),), eps_n=EPS_N, eps_t=EPS_T, ninc=20):
    """Входной файл CalculiX. sy=None — упругая матрица; hard — пары (Δσ, ε_p) упрочнения от sy."""
    E_h = E if E_h is None else E_h
    nu_h = nu if nu_h is None else nu_h
    n2 = len(nodes)
    # узлы: слой z=0 (0..n2-1), слой z=t (n2..2n2-1), средний слой — только для угловых узлов
    corner = np.unique(quads[:, :4])
    mid_id = -np.ones(n2, int); mid_id[corner] = 2 * n2 + np.arange(len(corner))
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for k, (x, y) in enumerate(nodes):
            f.write(f"{k + 1},{x:.7g},{y:.7g},0\n")
        for k, (x, y) in enumerate(nodes):
            f.write(f"{n2 + k + 1},{x:.7g},{y:.7g},{t:.7g}\n")
        for k in corner:
            x, y = nodes[k]
            f.write(f"{mid_id[k] + 1},{x:.7g},{y:.7g},{t / 2:.7g}\n")
        for name, sel in (("EMAT", ~plate), ("EPL", plate)):
            f.write(f"*ELEMENT, TYPE=C3D20R, ELSET={name}\n")
            for e in np.nonzero(sel)[0]:
                q = quads[e]
                c = [q[0], q[1], q[2], q[3], q[0] + n2, q[1] + n2, q[2] + n2, q[3] + n2,
                     q[4], q[5], q[6], q[7], q[4] + n2, q[5] + n2, q[6] + n2, q[7] + n2,
                     mid_id[q[0]], mid_id[q[1]], mid_id[q[2]], mid_id[q[3]]]
                ids = [str(v + 1) for v in c]
                f.write(f"{e + 1}," + ",".join(ids[:15]) + ",\n" + ",".join(ids[15:]) + "\n")
        f.write("*ELSET, ELSET=EALL\nEMAT, EPL\n")
        xs = nodes[:, 0]; ys = nodes[:, 1]
        tol = 1e-6
        allnodes = lambda sel: [k + 1 for k in np.nonzero(sel)[0]] + [n2 + k + 1 for k in np.nonzero(sel)[0]] + \
            [mid_id[k] + 1 for k in np.nonzero(sel)[0] if mid_id[k] >= 0]
        for name, sel in (("SX", np.abs(xs) < tol), ("SY", np.abs(ys) < tol)):
            ids = allnodes(sel)
            f.write(f"*NSET, NSET={name}\n")
            for i in range(0, len(ids), 12):
                f.write(",".join(map(str, ids[i:i + 12])) + ",\n")
        f.write("*MATERIAL, NAME=ZR\n*ELASTIC\n%g, %g\n" % (E, nu))
        f.write("*EXPANSION\n0.\n")
        if sy is not None:
            f.write("*PLASTIC\n")
            for ds, ep in hard:
                f.write(f"{sy + ds:g}, {ep:g}\n")
        # ортотропное расширение в CalculiX работает только с ортотропной записью упругости
        lam = E_h * nu_h / ((1 + nu_h) * (1 - 2 * nu_h)); mu = E_h / (2 * (1 + nu_h)); d = lam + 2 * mu
        f.write("*MATERIAL, NAME=HYD\n*ELASTIC, TYPE=ORTHO\n%g,%g,%g,%g,%g,%g,%g,%g,\n%g,0.\n" % (d, lam, d, lam, lam, d, mu, mu, mu))
        f.write("*EXPANSION, TYPE=ORTHO\n%g, %g, %g\n" % (eps_t, eps_n, eps_t))
        f.write("*SOLID SECTION, ELSET=EMAT, MATERIAL=ZR\n*SOLID SECTION, ELSET=EPL, MATERIAL=HYD\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 0.\n")
        f.write("*BOUNDARY\nNALL, 3, 3\nSX, 1, 1\nSY, 2, 2\n")
        f.write("*STEP, INC=1000\n*STATIC\n%g, 1., 1e-6, %g\n" % (1.0 / ninc, 1.0 / ninc))
        f.write("*TEMPERATURE\nNALL, 1.\n")
        f.write("*EL PRINT, ELSET=EALL, FREQUENCY=10000\nS\n")
        if sy is not None:
            f.write("*EL PRINT, ELSET=EMAT, FREQUENCY=10000\nPEEQ\n")
        f.write("*END STEP\n")


def run_ccx(path, threads=4):
    d, base = os.path.split(path)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               CCX_NPROC_EQUATION_SOLVER=str(threads))
    r = subprocess.run([CCX, "-i", base[:-4]], cwd=d or ".", env=env, capture_output=True, text=True)
    if "Job finished" not in r.stdout and "Total CalculiX Time" not in r.stdout:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])
    return r.stdout


def read_dat(path, nelem):
    """Последний вывод напряжений (и PEEQ) из .dat: средние по точкам интегрирования."""
    txt = open(path).read()
    blocks = re.split(r"\n\s*(stresses|equivalent plastic strain)", txt)
    S = None; PE = np.zeros(nelem)
    for i in range(1, len(blocks), 2):
        kind, body = blocks[i], blocks[i + 1]
        rows = []
        for line in body.split("\n")[1:]:
            parts = line.split()
            if not parts:
                if rows:
                    break
                continue
            rows.append([float(v) for v in parts])
        a = np.array(rows)
        if kind == "stresses":
            S = np.zeros((nelem, 6)); cnt = np.zeros(nelem)
            e = a[:, 0].astype(int) - 1
            np.add.at(S, e, a[:, 2:8]); np.add.at(cnt, e, 1)
            S /= np.maximum(cnt, 1)[:, None]
        else:
            PE = np.zeros(nelem); cnt = np.zeros(nelem)
            e = a[:, 0].astype(int) - 1
            np.add.at(PE, e, a[:, 2]); np.add.at(cnt, e, 1)
            PE /= np.maximum(cnt, 1)
    return S, PE           # S: sxx, syy, szz, sxy, sxz, syz


def mirror(xy, S, PE, plate):
    """Четверть → вся плоскость (σ_xy меняет знак при отражении по одной оси)."""
    X, Y, SS, P, F = [], [], [], [], []
    for sx in (1, -1):
        for sy in (1, -1):
            s = S.copy(); s[:, 3] *= sx * sy
            X.append(sx * xy[:, 0]); Y.append(sy * xy[:, 1]); SS.append(s); P.append(PE); F.append(plate)
    return np.concatenate(X), np.concatenate(Y), np.concatenate(SS), np.concatenate(P), np.concatenate(F)


def solve(work, name, mesh, **kw):
    nodes, quads, plate = mesh
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, name + ".inp")
    write_inp(path, nodes, quads, plate, **kw)
    run_ccx(path)
    S, PE = read_dat(path[:-4] + ".dat", len(quads))
    cen = nodes[quads[:, :4]].mean(1)
    return dict(xy=cen, S=S, PE=PE, plate=plate)
