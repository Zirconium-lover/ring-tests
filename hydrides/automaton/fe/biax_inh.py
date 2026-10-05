"""Неоднородность гидрида и двухосность (проверка к Cinbiz et al. 2016): одна окружная пластинка
(нормаль по ND = y, полудлина 2.5 мкм, толщина 0.6 мкм) с модулями гидрида ≠ модулям матрицы, нагрузка —
окружное σ_x и осевое σ_z (обобщённая плоская деформация: u_z на слое задаёт ε_z). Сначала нагрузка,
затем несоответствие (гидрид выделяется под нагрузкой). Выход — поле Δg = (ε_n − ε_t)/ε_n·(σ_xx − σ_yy):
выгода радиальной новой пластинки над окружной в каждой точке.
python biax_inh.py папка"""
import os
import sys
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_fe import mesh_quarter, run_ccx, read_dat, mirror, EPS_N, EPS_T  # noqa: E402

E, NU, A_HALF, H, D = 90e3, 0.34, 2.5, 0.6, 40.0
# пороги Cinbiz: (σ_x, σ_z) — одноосное 155, «плоская деформация» 110 при 0.57, «почти равнодвухосное» 75 при 0.83
LOADS = {"none": (0, 0), "U155": (155, 0), "P110": (110, 0.57 * 110), "B75": (75, 0.83 * 75),
         "U110": (110, 0), "U75": (75, 0)}


def write_inp(path, nodes, quads, plate, t=0.1, E_h=E, nu_h=NU, sx=0.0, sz=0.0, sy=None, hard=((0.0, 0.0),), ninc=10):
    n2 = len(nodes)
    corner = np.unique(quads[:, :4])
    mid_id = -np.ones(n2, int); mid_id[corner] = 2 * n2 + np.arange(len(corner))
    ex = (sx - NU * sz) / E; ez = (sz - NU * sx) / E            # дальнее поле: σ_y = 0
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
        xs, ys = nodes[:, 0], nodes[:, 1]
        tol = 1e-6

        def nset(name, ids):
            f.write(f"*NSET, NSET={name}\n")
            for i in range(0, len(ids), 12):
                f.write(",".join(map(str, ids[i:i + 12])) + ",\n")
        layers = lambda sel: [k + 1 for k in np.nonzero(sel)[0]] + [n2 + k + 1 for k in np.nonzero(sel)[0]] + \
            [mid_id[k] + 1 for k in np.nonzero(sel)[0] if mid_id[k] >= 0]
        nset("SX", layers(np.abs(xs) < tol)); nset("SY", layers(np.abs(ys) < tol)); nset("RX", layers(np.abs(xs - D) < tol))
        nset("ZBOT", list(range(1, n2 + 1))); nset("ZTOP", list(range(n2 + 1, 2 * n2 + 1)))
        nset("ZMID", [mid_id[k] + 1 for k in corner])
        f.write("*MATERIAL, NAME=ZR\n*ELASTIC\n%g, %g\n*EXPANSION\n0.\n" % (E, NU))
        if sy is not None:
            f.write("*PLASTIC\n" + "".join(f"{sy + ds:g}, {ep:g}\n" for ds, ep in hard))
        lam = E_h * nu_h / ((1 + nu_h) * (1 - 2 * nu_h)); mu = E_h / (2 * (1 + nu_h)); d = lam + 2 * mu
        f.write("*MATERIAL, NAME=HYD\n*ELASTIC, TYPE=ORTHO\n%g,%g,%g,%g,%g,%g,%g,%g,\n%g,0.\n" % (d, lam, d, lam, lam, d, mu, mu, mu))
        f.write("*EXPANSION, TYPE=ORTHO\n%g, %g, %g\n" % (EPS_T, EPS_N, EPS_T))
        f.write("*SOLID SECTION, ELSET=EMAT, MATERIAL=ZR\n*SOLID SECTION, ELSET=EPL, MATERIAL=HYD\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 0.\n")
        f.write("*BOUNDARY\nSX, 1, 1\nSY, 2, 2\nZBOT, 3, 3\n")
        # шаг 1: нагрузка
        f.write("*STEP, INC=1000\n*STATIC\n0.5, 1., 1e-6, 0.5\n")
        f.write(f"*BOUNDARY\nRX, 1, 1, {ex * D:.9g}\nZTOP, 3, 3, {ez * t:.9g}\nZMID, 3, 3, {ez * t / 2:.9g}\n")
        f.write("*TEMPERATURE\nNALL, 0.\n*END STEP\n")
        # шаг 2: выделение гидрида под нагрузкой
        f.write("*STEP, INC=1000\n*STATIC\n%g, 1., 1e-6, %g\n" % (1.0 / ninc, 1.0 / ninc))
        f.write("*TEMPERATURE\nNALL, 1.\n")
        f.write("*EL PRINT, ELSET=EALL, FREQUENCY=10000\nS\n")
        if sy is not None:
            f.write("*EL PRINT, ELSET=EMAT, FREQUENCY=10000\nPEEQ\n")
        f.write("*END STEP\n")


def solve(work, name, mesh, **kw):
    nodes, quads, plate = mesh
    path = os.path.join(work, name + ".inp")
    write_inp(path, nodes, quads, plate, **kw)
    run_ccx(path, threads=1)
    S, PE = read_dat(path[:-4] + ".dat", len(quads))
    return dict(xy=nodes[quads[:, :4]].mean(1), S=S, PE=PE, plate=plate)


def dg_field(r, X, Y):
    x, y, S, PE, F = mirror(r["xy"], r["S"], r["PE"], r["plate"])
    pts = np.stack([x, y], 1)
    dg = (EPS_N - EPS_T) / EPS_N * (S[:, 0] - S[:, 1])
    return LinearNDInterpolator(pts, dg)(X, Y), LinearNDInterpolator(pts, F.astype(float))(X, Y) > 0.5


def job(args):
    work, name, kw = args
    out = os.path.join(work, name + ".npz")
    if os.path.exists(out):
        return name
    r = solve(work, name, MESH, **kw)
    g = np.arange(-8 + 0.05, 8, 0.1)
    X, Y = np.meshgrid(g, g)
    dg, pl = dg_field(r, X, Y)
    np.savez_compressed(out, dg=dg.astype(np.float32), plate=pl, x=g)
    return name


if __name__ == "__main__":
    from multiprocessing import Pool
    OUT = sys.argv[1]
    work = os.path.join(OUT, "ccx"); os.makedirs(work, exist_ok=True)
    MESH = mesh_quarter(D=D, a=A_HALF, h=H)
    print("сетка:", len(MESH[1]), "элементов", flush=True)
    cases = []
    for ratio, nu_h in ((1.0, NU), (0.9, 0.30), (1.2, 0.30), (1.4, 0.30)):
        for lk, (sx, sz) in LOADS.items():
            cases.append((work, f"el_r{ratio}_{lk}", dict(E_h=ratio * E, nu_h=nu_h, sx=sx, sz=sz)))
    for ratio, nu_h in ((1.0, NU), (1.2, 0.30)):
        for lk, (sx, sz) in LOADS.items():
            cases.append((work, f"pl_r{ratio}_{lk}", dict(E_h=ratio * E, nu_h=nu_h, sx=sx, sz=sz, sy=350.0,
                                                           hard=((0.0, 0.0), (20.0, 0.1)))))
    with Pool(4) as pool:
        for nm in pool.imap_unordered(job, cases):
            print(nm, flush=True)
    print("готово", len(cases))
