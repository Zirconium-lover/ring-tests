"""МКЭ-сверка Фурье-решателя для стопок (CalculiX): та же задача, что stack_runs.py, — периодическая ячейка
30 × 30 мкм, обобщённая плоская деформация, пластинки 5 × 0.6 мкм, Мизес (σ_y 350, упрочнение 200 МПа).

Оси сетки — оси пластинки: X — вдоль (t), Y — по нормали (n), Z — ось трубы. Окружная пластинка: σθ по X;
радиальная: σθ по Y (по нормали). Матрица изотропная, поэтому зеркальная укладка не меняет задачу.
• Периодичность — уравнениями u(правая) − u(левая) = u(CX), u(верх) − u(низ) = u(CY) для каждой пары узлов;
  CX, CY — управляющие узлы, силы на них = Σ·площадь грани (среднее напряжение задано, как в Фурье);
  поворот снят u_x(CY) = 0, сдвиговая сила 0.
• Обобщённая плоская деформация — один слой C3D20R: u_z = 0 снизу, u_z = u_z(CZ) сверху и u_z(CZ)/2 в середине,
  сила на CZ = Σ_zz·площадь.
• Шаги: 1 — нагрузка; 2 — несоответствие старых пластинок 0 → 1 (ортотропное «тепловое» расширение);
  3 — новой пластинки 0 → 1; ⟨σ⟩ по новой (и старым) пластинкам на каждом приращении → работа превращения
  W = ∫⟨σ⟩:ε* dt — та же величина, что в hill_fft.Plate.run.
• Новая пластинка во 2-м шаге уже упругий гидрид (без несоответствия): материал в CalculiX между шагами не
  сменить. В Фурье для сверки то же — stack_runs.py --fut-el.
python stack_fe.py папка случай[,случай...] [потоков]   (случай = deck_rad_U110, chain_circ_0, single_rad_P110…;
дополнительно _Eh0.9 — модуль гидрида 0.9 от матрицы, ν_h 0.32 (Puls 2012, 2.2.3: δ-гидрид 80–85 ГПа,
ν 0.32 по Yamanaka); _el — матрица упругая)"""
import os
import re
import sys
import json
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_fe import run_ccx, EPS_N, EPS_T  # noqa: E402
from hill_runs import LOADS  # noqa: E402

L, A_HALF, TH, T_LAYER = 30.0, 2.5, 0.6, 0.1
E, NU, SY, HARD = 90e3, 0.34, 350.0, 200.0
STEP = {"deck": (2.5, 1.2), "chain": (5.8, 0.6)}


def plates_of(conf):
    """Центры (t, n): k = 0 — новая, 1 и 2 — старые (как в stack_runs.py)."""
    if conf == "single":
        return [(0.0, 0.0)]
    st, sn = STEP[conf]
    return [(0.0, 0.0), (-st, -sn), (-2 * st, -2 * sn)]


def mesh_cell(centers, s_min=0.04, s_mid=0.12, s_max=1.0, d_mid=1.0, d_max=8.0):
    """Периодическая сетка ячейки [−L/2, L/2]²: 8-узловые четырёхугольники. Возвращает узлы, элементы, номер
    пластинки элемента (−1 — матрица)."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cell")
    occ = gmsh.model.occ
    dom = occ.addRectangle(-L / 2, -L / 2, 0, L, L)
    pls = [occ.addRectangle(t - A_HALF, n - TH / 2, 0, 2 * A_HALF, TH) for t, n in centers]
    occ.fragment([(2, dom)], [(2, p) for p in pls])
    occ.synchronize()
    surfs = [s for _, s in gmsh.model.getEntities(2)]
    owner = {}
    for s in surfs:                                     # пластинка — поверхность с рамкой её прямоугольника
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, s)
        k = [i for i, (t, n) in enumerate(centers)
             if abs(x0 - t + A_HALF) < 1e-6 and abs(x1 - t - A_HALF) < 1e-6 and abs(y0 - n + TH / 2) < 1e-6
             and abs(y1 - n - TH / 2) < 1e-6]
        owner[s] = k[0] if k else -1
    pcurves = sorted({c for s in surfs if owner[s] >= 0 for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)})
    side = {}
    for _, c in gmsh.model.getEntities(1):
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
        for key, ok in (("L", abs(x0 + L / 2) < 1e-6 and abs(x1 + L / 2) < 1e-6),
                        ("R", abs(x0 - L / 2) < 1e-6 and abs(x1 - L / 2) < 1e-6),
                        ("B", abs(y0 + L / 2) < 1e-6 and abs(y1 + L / 2) < 1e-6),
                        ("T", abs(y0 - L / 2) < 1e-6 and abs(y1 - L / 2) < 1e-6)):
            if ok:
                side[key] = c
    gmsh.model.mesh.setPeriodic(1, [side["R"]], [side["L"]], [1, 0, 0, L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    gmsh.model.mesh.setPeriodic(1, [side["T"]], [side["B"]], [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1])
    f = gmsh.model.mesh.field
    f.add("Distance", 1); f.setNumbers(1, "CurvesList", pcurves); f.setNumber(1, "Sampling", 600)
    f.add("Threshold", 2); f.setNumber(2, "InField", 1)
    f.setNumber(2, "SizeMin", s_min); f.setNumber(2, "SizeMax", s_mid)
    f.setNumber(2, "DistMin", 0.05); f.setNumber(2, "DistMax", d_mid); f.setNumber(2, "StopAtDistMax", 1)
    f.add("Threshold", 3); f.setNumber(3, "InField", 1)
    f.setNumber(3, "SizeMin", s_mid); f.setNumber(3, "SizeMax", s_max)
    f.setNumber(3, "DistMin", d_mid); f.setNumber(3, "DistMax", d_max)
    f.add("Min", 4); f.setNumbers(4, "FieldsList", [2, 3])
    f.add("MathEval", 5); f.setString(5, "F", "2*F4")         # подразбиение треугольников делит размер пополам
    f.setAsBackgroundMesh(5)
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Algorithm", 8), ("Mesh.RecombineAll", 1),
                 ("Mesh.RecombinationAlgorithm", 1), ("Mesh.SubdivisionAlgorithm", 1),
                 ("Mesh.SecondOrderIncomplete", 1)):
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    idx = {t: i for i, t in enumerate(tags)}
    nodes = xyz.reshape(-1, 3)[:, :2]
    Q, own = [], []
    for s in surfs:
        et, _, enodes = gmsh.model.mesh.getElements(2, s)
        for typ, en in zip(et, enodes):
            nn = gmsh.model.mesh.getElementProperties(typ)[3]
            assert nn == 8
            conn = np.array([idx[t] for t in en]).reshape(-1, 8)
            Q.append(conn); own.append(np.full(len(conn), owner[s]))
    gmsh.finalize()
    Q = np.concatenate(Q); own = np.concatenate(own)
    p = nodes[Q[:, :4]]
    a2 = np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - np.roll(p[:, :, 0], -1, 1) * p[:, :, 1], 1)
    Q[a2 < 0] = Q[a2 < 0][:, [0, 3, 2, 1, 7, 6, 5, 4]]
    used = np.unique(Q)
    remap = -np.ones(len(nodes), int); remap[used] = np.arange(len(used))
    return nodes[used], remap[Q], own


def quad_area(nodes, Q):
    p = nodes[Q[:, :4]]
    return 0.5 * np.abs(np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - np.roll(p[:, :, 0], -1, 1) * p[:, :, 1], 1))


def pairs(xs, ys, a, b, along):
    """Пары узлов на противоположных сторонах (a — зависимая, b — независимая), по координате along."""
    ia = np.nonzero(a)[0]; ib = np.nonzero(b)[0]
    ia = ia[np.argsort(along[ia])]; ib = ib[np.argsort(along[ib])]
    assert len(ia) == len(ib) and np.abs(along[ia] - along[ib]).max() < 1e-6, "сетка не периодична"
    return list(zip(ia, ib))


def write_inp(path, nodes, Q, own, Sig, E_h=E, nu_h=NU, ninc=10, plastic=True):
    n2 = len(nodes)
    corner = np.unique(Q[:, :4])
    mid = -np.ones(n2, int); mid[corner] = 2 * n2 + np.arange(len(corner))
    nn = 2 * n2 + len(corner)
    CX, CY, CZ = nn + 1, nn + 2, nn + 3                                   # номера управляющих узлов (с 1)
    layers = lambda k: [k + 1, n2 + k + 1] + ([mid[k] + 1] if mid[k] >= 0 else [])
    xs, ys = nodes[:, 0], nodes[:, 1]
    tol = 1e-6
    onL, onR, onB, onT = (np.abs(xs + L / 2) < tol, np.abs(xs - L / 2) < tol,
                          np.abs(ys + L / 2) < tol, np.abs(ys - L / 2) < tol)
    cor = onL | onR
    cor &= onB | onT
    A = int(np.nonzero(onL & onB)[0][0]); Bn = int(np.nonzero(onR & onB)[0][0])
    C = int(np.nonzero(onR & onT)[0][0]); D = int(np.nonzero(onL & onT)[0][0])
    pr_x = pairs(xs, ys, onR & ~cor, onL & ~cor, ys)
    pr_y = pairs(xs, ys, onT & ~cor, onB & ~cor, xs)
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for k, (x, y) in enumerate(nodes):
            f.write(f"{k + 1},{x:.9g},{y:.9g},0\n")
        for k, (x, y) in enumerate(nodes):
            f.write(f"{n2 + k + 1},{x:.9g},{y:.9g},{T_LAYER:.9g}\n")
        for k in corner:
            f.write(f"{mid[k] + 1},{nodes[k, 0]:.9g},{nodes[k, 1]:.9g},{T_LAYER / 2:.9g}\n")
        f.write(f"*NODE, NSET=NCTRL\n{CX},{L},0,0\n{CY},0,{L},0\n{CZ},0,0,{L}\n")
        sets = {"EMAT": own < 0, "ENEW": own == 0, "EOLD": own > 0}
        for name, sel in sets.items():
            if not sel.any():
                continue
            f.write(f"*ELEMENT, TYPE=C3D20R, ELSET={name}\n")
            for e in np.nonzero(sel)[0]:
                q = Q[e]
                c = [q[0], q[1], q[2], q[3], q[0] + n2, q[1] + n2, q[2] + n2, q[3] + n2,
                     q[4], q[5], q[6], q[7], q[4] + n2, q[5] + n2, q[6] + n2, q[7] + n2,
                     mid[q[0]], mid[q[1]], mid[q[2]], mid[q[3]]]
                ids = [str(v + 1) for v in c]
                f.write(f"{e + 1}," + ",".join(ids[:15]) + ",\n" + ",".join(ids[15:]) + "\n")
        f.write("*ELSET, ELSET=EALL\n" + ", ".join(k for k, s in sets.items() if s.any()) + "\n")

        def nset(name, ids):
            f.write(f"*NSET, NSET={name}\n")
            for i in range(0, len(ids), 12):
                f.write(",".join(map(str, ids[i:i + 12])) + ",\n")

        def nodes_of(sel):
            ks = np.unique(Q[sel])
            return sorted({v for k in ks for v in layers(k)})
        nset("NNEW", nodes_of(own == 0))
        if (own > 0).any():
            nset("NOLD", nodes_of(own > 0))
        nset("NBOT", list(range(1, n2 + 1)))
        nset("NFIX", layers(A))
        # периодичность в плоскости и обобщённая плоская деформация
        f.write("*EQUATION\n")
        for i, j in pr_x:
            for a, b in zip(layers(i), layers(j)):
                for d in (1, 2):
                    f.write(f"3\n{a},{d},1.,{b},{d},-1.,{CX},{d},-1.\n")
        for i, j in pr_y:
            for a, b in zip(layers(i), layers(j)):
                for d in (1, 2):
                    f.write(f"3\n{a},{d},1.,{b},{d},-1.,{CY},{d},-1.\n")
        for k, ctl in ((Bn, (CX,)), (D, (CY,)), (C, (CX, CY))):
            for a, b in zip(layers(k), layers(A)):
                for d in (1, 2):
                    f.write(f"{2 + len(ctl)}\n{a},{d},1.,{b},{d},-1.," + ",".join(f"{c},{d},-1." for c in ctl) + "\n")
        for k in range(n2):
            f.write(f"2\n{n2 + k + 1},3,1.,{CZ},3,-1.\n")
        for k in corner:
            f.write(f"2\n{mid[k] + 1},3,1.,{CZ},3,-0.5\n")
        f.write("*MATERIAL, NAME=ZR\n*ELASTIC\n%g, %g\n*EXPANSION\n0.\n" % (E, NU))
        if plastic:
            f.write("*PLASTIC\n%g, 0.\n%g, 1.\n" % (SY, SY + HARD))
        lam = E_h * nu_h / ((1 + nu_h) * (1 - 2 * nu_h)); mu = E_h / (2 * (1 + nu_h)); d = lam + 2 * mu
        f.write("*MATERIAL, NAME=HYD\n*ELASTIC, TYPE=ORTHO\n%g,%g,%g,%g,%g,%g,%g,%g,\n%g,0.\n" % (d, lam, d, lam, lam, d, mu, mu, mu))
        f.write("*EXPANSION, TYPE=ORTHO\n%g, %g, %g\n" % (EPS_T, EPS_N, EPS_T))
        f.write("*SOLID SECTION, ELSET=EMAT, MATERIAL=ZR\n*SOLID SECTION, ELSET=ENEW, MATERIAL=HYD\n")
        if (own > 0).any():
            f.write("*SOLID SECTION, ELSET=EOLD, MATERIAL=HYD\n")
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 0.\n")
        f.write(f"*BOUNDARY\nNBOT, 3, 3\nNFIX, 1, 2\n{CX}, 3, 3\n{CY}, 1, 1\n{CY}, 3, 3\n{CZ}, 1, 2\n")
        sxx, syy, szz = Sig
        f.write("*STEP, INC=1000\n*STATIC\n1., 1.\n")
        f.write(f"*CLOAD\n{CX}, 1, {sxx * L * T_LAYER:.9g}\n{CY}, 2, {syy * L * T_LAYER:.9g}\n{CZ}, 3, {szz * L * L:.9g}\n")
        # частота вывода в CalculiX одна на все запросы шага — поэтому поля целиком выводит отдельный пустой шаг 4
        f.write("*TEMPERATURE\nNALL, 0.\n*EL PRINT, ELSET=ENEW\nS\n" + ("*EL PRINT, ELSET=EOLD\nS\n" if (own > 0).any() else "")
                + "*NODE PRINT, NSET=NCTRL\nU\n*END STEP\n")
        if (own > 0).any():
            f.write("*STEP, INC=1000\n*STATIC\n%g, 1., 1e-6, %g\n" % (1.0 / ninc, 1.0 / ninc))
            f.write("*TEMPERATURE\nNOLD, 1.\n*EL PRINT, ELSET=EOLD\nS\n*EL PRINT, ELSET=ENEW\nS\n*END STEP\n")
        f.write("*STEP, INC=1000\n*STATIC\n%g, 1., 1e-6, %g\n" % (1.0 / ninc, 1.0 / ninc))
        f.write("*TEMPERATURE\nNNEW, 1.\n*EL PRINT, ELSET=ENEW\nS\n*END STEP\n")
        f.write("*STEP\n*STATIC\n1., 1.\n*EL PRINT, ELSET=EALL\nS\n" + ("*EL PRINT, ELSET=EMAT\nPEEQ\n" if plastic else ""))
        f.write("*NODE PRINT, NSET=NCTRL\nU\n*END STEP\n")


def read_blocks(dat, what, eset):
    """Все выводы величины what ('stresses' / 'equivalent plastic strain') для набора: [(время, строки)]."""
    pat = re.compile(r"\n\s*" + what + r" \([^)]*\)[^\n]*for set " + eset + r" and time\s+(\S+)")
    out = []
    for m in pat.finditer(dat):
        rows = []
        for line in dat[m.end():].split("\n")[1:]:
            p = line.split()
            if not p:
                if rows:
                    break
                continue
            rows.append([float(v) for v in p])
        out.append((float(m.group(1)), np.array(rows)))
    return out


def elem_mean(rows, nel, cols):
    cols = np.arange(rows.shape[1])[cols]
    S = np.zeros((nel, len(cols))); c = np.zeros(nel)
    e = rows[:, 0].astype(int) - 1
    np.add.at(S, e, rows[:, cols]); np.add.at(c, e, 1)
    return S / np.maximum(c, 1)[:, None], c > 0


def work(dat, eset, nel, area, t0, s_start, eps):
    """W = ∫⟨σ⟩:ε* dt по приращениям шага, начинающегося в момент t0 (время в .dat — полное)."""
    ts, ss = [0.0], [s_start]
    for tt, rows in read_blocks(dat, "stresses", eset):
        if tt <= t0 + 1e-9 or tt > t0 + 1.0 + 1e-9:
            continue
        S, has = elem_mean(rows, nel, slice(2, 8))
        ts.append(tt - t0); ss.append((S[has] * area[has, None]).sum(0) / area[has].sum())
    ts, ss = np.array(ts), np.array(ss)
    return float(np.trapezoid(ss @ eps, ts)), ss[-1]


def case_load(orient, lk):
    sth, sz = LOADS[lk]
    return (sth, 0.0, sz) if orient == "circ" else (0.0, sth, sz)      # оси сетки: X вдоль пластинки, Y по нормали


def run_case(work_dir, name, threads=1, mesh_kw=None):
    parts = name.split("_")
    conf, orient, lk = parts[:3]
    Eh = ([float(q[2:]) for q in parts[3:] if q.startswith("Eh")] or [1.0])[0]
    plastic = "el" not in parts[3:]
    fn = os.path.join(work_dir, name + ".json")
    if os.path.exists(fn):
        return json.load(open(fn))
    t0 = time.time()
    nodes, Q, own = mesh_cell(plates_of(conf), **(mesh_kw or {}))
    Sig = case_load(orient, lk)
    path = os.path.join(work_dir, name + ".inp")
    write_inp(path, nodes, Q, own, Sig, E_h=Eh * E, nu_h=NU if Eh == 1.0 else 0.32, plastic=plastic)
    dat_fn = path[:-4] + ".dat"
    if not (os.path.exists(dat_fn) and "for set EALL" in open(dat_fn).read()):
        run_ccx(path, threads=threads)
    dat = open(dat_fn).read()
    area = quad_area(nodes, Q)
    nel = len(Q)
    eps = np.array([EPS_T, EPS_N, EPS_T, 0, 0, 0])
    has_old = (own > 0).any()
    a_old = np.where(own > 0, area, 0.0); a_new = np.where(own == 0, area, 0.0)
    W_old = None
    if has_old:
        pre = [r for tt, r in read_blocks(dat, "stresses", "EOLD") if tt <= 1.0 + 1e-9][-1]
        S0, h0 = elem_mean(pre, nel, slice(2, 8))
        W_old, _ = work(dat, "EOLD", nel, a_old, 1.0, (S0[h0] * a_old[h0, None]).sum(0) / a_old[h0].sum(), eps)
    # ⟨σ⟩ новой пластинки перед её выделением — последний вывод предыдущего шага
    t_new = 2.0 if has_old else 1.0
    pre = [r for tt, r in read_blocks(dat, "stresses", "ENEW") if tt <= t_new + 1e-9][-1]
    S0, h0 = elem_mean(pre, nel, slice(2, 8))
    s_new0 = (S0[h0] * a_new[h0, None]).sum(0) / a_new[h0].sum()
    W, s_fin = work(dat, "ENEW", nel, a_new, t_new, s_new0, eps)
    # поля конца расчёта (шаг 4): σ по элементам, PEEQ
    allS = read_blocks(dat, "stresses", "EALL")[-1][1]
    S_el, _ = elem_mean(allS, nel, slice(2, 8))
    pe = read_blocks(dat, "equivalent plastic strain", "EMAT")
    PE = elem_mean(pe[-1][1], nel, slice(2, 3))[0][:, 0] if pe else np.zeros(nel)
    cen = nodes[Q[:, :4]].mean(1)
    np.savez_compressed(os.path.join(work_dir, name + ".npz"), xy=cen.astype(np.float32), area=area.astype(np.float32),
                        S=S_el.astype(np.float32), PE=PE.astype(np.float32), own=own.astype(np.int8))
    res = dict(conf=conf, orient=orient, load=lk, Eh=Eh, plastic=plastic, s_theta=LOADS[lk][0], s_z=LOADS[lk][1], W=W,
               g=W / EPS_N, W_old=W_old, s_new0=s_new0.tolist(), s_plate=s_fin.tolist(),
               pl_area=float(area[PE > 1e-4].sum()), n_elem=int(nel), time_s=time.time() - t0)
    json.dump(res, open(fn, "w"))
    for ext in (".frd", ".12d", ".cvg"):
        if os.path.exists(path[:-4] + ext):
            os.remove(path[:-4] + ext)
    return res


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    thr = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for nm in sys.argv[2].split(","):
        r = run_case(out, nm, thr)
        print(nm, "g", round(r["g"], 1), "W_old", r["W_old"] and round(r["W_old"], 2), "элементов", r["n_elem"],
              "время", round(r["time_s"]), flush=True)
