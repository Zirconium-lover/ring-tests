"""Напряжения несовместности между зёрнами циркония в сечении r–θ (МКЭ, CalculiX).

Зёрна — те же ячейки Вороного, что в автомате (make_grains); ось c каждого зерна лежит
в плоскости сечения и перпендикулярна следу пластинки гидрида (ψ = ±χ0 + разброс).
Гексагональный кристалл: упругость и расширение трансверсально-изотропны вокруг оси c.
Регулярная сетка C3D8 в один слой, обобщённая плоская деформация (верх по z сдвинут
на заданную осевую деформацию). Два линейных случая:
  thermal — остывание на ΔT (по умолчанию −100 К) от безнапряжённого состояния;
  mech    — окружное (вдоль TD = x) растяжение σ (по умолчанию 100 МПа).
Края свободные (для mech — нагружены), анализ — во внутренней области.
Единицы: мкм, МПа, К."""
import os
import re
import subprocess
import numpy as np

CCX = os.environ.get("CCX", "/home/user/ccx-arch2/src/ccx_2.23_pardiso")
# Zr, монокристалл (Fisher & Renken, комнатная температура), МПа; расширение — средние 20–400 °C
C11, C12, C13, C33, C44 = 143.4e3, 72.8e3, 65.3e3, 164.8e3, 32.0e3
ALPHA_A, ALPHA_C = 5.5e-6, 10.5e-6


def write_inp(path, grains, gpsi, dx, case="thermal", dT=-100.0, sigma=100.0, t=None):
    ny, nx = grains.shape
    t = dx if t is None else t
    NX, NY = nx + 1, ny + 1
    nid = lambda ix, iy, k: 1 + k * NX * NY + iy * NX + ix          # iy = 0 — низ (y = 0)
    with open(path, "w") as f:
        f.write("*NODE, NSET=NALL\n")
        for k in (0, 1):
            for iy in range(NY):
                for ix in range(NX):
                    f.write(f"{nid(ix, iy, k)},{ix * dx:.6g},{iy * dx:.6g},{k * t:.6g}\n")
        # элементы: строка i (0 — верх), столбец j
        for gid in np.unique(grains):
            f.write(f"*ELEMENT, TYPE=C3D8, ELSET=G{gid}\n")
            ii, jj = np.nonzero(grains == gid)
            for i, j in zip(ii, jj):
                iy = ny - 1 - i
                c = [nid(j, iy, 0), nid(j + 1, iy, 0), nid(j + 1, iy + 1, 0), nid(j, iy + 1, 0),
                     nid(j, iy, 1), nid(j + 1, iy, 1), nid(j + 1, iy + 1, 1), nid(j, iy + 1, 1)]
                f.write(f"{1 + i * nx + j}," + ",".join(map(str, c)) + "\n")
        f.write("*ELSET, ELSET=EALL\n")
        ids = [f"G{g}" for g in np.unique(grains)]
        for k in range(0, len(ids), 10):
            f.write(",".join(ids[k:k + 10]) + ",\n")
        c66 = 0.5 * (C11 - C12)
        f.write("*MATERIAL, NAME=ZR\n*ELASTIC, TYPE=ORTHO\n")
        f.write(f"{C11:g},{C12:g},{C11:g},{C13:g},{C13:g},{C33:g},{c66:g},{C44:g},\n{C44:g},0.\n")
        f.write(f"*EXPANSION, TYPE=ORTHO\n{ALPHA_A:g},{ALPHA_A:g},{ALPHA_C:g}\n")
        for gid in np.unique(grains):
            th = gpsi[gid] + np.pi / 2                         # ось c ⟂ следу пластинки
            a = (-np.sin(th), np.cos(th), 0.0)
            f.write(f"*ORIENTATION, NAME=O{gid}, SYSTEM=RECTANGULAR\n{a[0]:.8f},{a[1]:.8f},0,0,0,1\n")
            f.write(f"*SOLID SECTION, ELSET=G{gid}, MATERIAL=ZR, ORIENTATION=O{gid}\n")
        bot = [nid(ix, iy, 0) for iy in range(NY) for ix in range(NX)]
        top = [nid(ix, iy, 1) for iy in range(NY) for ix in range(NX)]

        def nset(name, ids):
            f.write(f"*NSET, NSET={name}\n")
            for k in range(0, len(ids), 16):
                f.write(",".join(map(str, ids[k:k + 16])) + ",\n")
        nset("BOT", bot); nset("TOP", top)
        nset("C00", [nid(0, 0, 0), nid(0, 0, 1)])
        nset("C10", [nid(nx, 0, 0), nid(nx, 0, 1)])
        left = [nid(0, iy, k) for k in (0, 1) for iy in range(NY)]
        nset("LEFT", left)
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 0.\n")
        f.write("*STEP\n*STATIC\n")
        f.write("*BOUNDARY\nBOT, 3, 3, 0.\n")
        if case == "thermal":
            f.write(f"TOP, 3, 3, {ALPHA_A * dT * t:.6g}\nC00, 1, 2, 0.\nC10, 2, 2, 0.\n")
            f.write(f"*TEMPERATURE\nNALL, {dT:g}\n")
        else:
            ezz = -0.34 * sigma / 95e3
            f.write(f"TOP, 3, 3, {ezz * t:.6g}\nLEFT, 1, 1, 0.\nC00, 2, 2, 0.\n")
            f.write("*CLOAD\n")
            fz = sigma * dx * t / 4.0
            for iy in range(NY):
                w = 1.0 if 0 < iy < NY - 1 else 0.5
                for k in (0, 1):
                    f.write(f"{nid(nx, iy, k)}, 1, {2 * fz * w:.6g}\n")
        f.write("*EL PRINT, ELSET=EALL, GLOBAL=YES\nS\n*END STEP\n")


def run(path, threads=4):
    d, base = os.path.split(path)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    r = subprocess.run([CCX, "-i", base[:-4]], cwd=d or ".", env=env, capture_output=True, text=True)
    if "Total CalculiX Time" not in r.stdout:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])


def read_stress(path, shape):
    """Средние по точкам интегрирования напряжения → массивы (ny, nx): σxx, σyy, σxy, σzz (оси x вправо, y вверх)."""
    txt = open(path).read()
    body = txt.split("stresses")[-1].split("\n", 1)[1]
    rows = np.loadtxt([ln for ln in body.split("\n") if ln.strip()][:])
    e = rows[:, 0].astype(int) - 1
    S = np.zeros((shape[0] * shape[1], 6)); cnt = np.zeros(shape[0] * shape[1])
    np.add.at(S, e, rows[:, 2:8]); np.add.at(cnt, e, 1)
    S /= np.maximum(cnt, 1)[:, None]
    S = S.reshape(shape + (6,))
    return dict(sxx=S[..., 0], syy=S[..., 1], szz=S[..., 2], sxy=S[..., 3])


def solve(work, name, grains, gpsi, dx, **kw):
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, name + ".inp")
    write_inp(path, grains, gpsi, dx, **kw)
    run(path)
    return read_stress(path[:-4] + ".dat", grains.shape)
