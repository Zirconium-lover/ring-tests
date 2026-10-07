"""Проверка шага диффузии автомата (Фурье) против точного решения и CalculiX (теплопроводность).

Шаг автомата: c ← IFFT[FFT(c)·exp(−D k² dt)] — точное решение уравнения Фика в периодической ячейке
для поля, заданного на сетке. Свежая пластинка — «яма» −fr·C_гидр в своих клетках (водород пластинки
взят из раствора), дальше она расплывается диффузией.

A. Фик: яма пластинки (5 × 0.6 мкм) в ячейке 96 мкм, шаг 0.4 мкм. Эталон — точное решение для
   кусочно-постоянного начального поля (сумма erf по клеткам и периодическим образам), усреднённое
   по клеткам. Те же случаи в CalculiX: DC3D8, узлы в центрах клеток, периодические уравнения,
   неявная схема с мелким шагом.
B. Диффузия с напряжениями (stress_diff): в автомате u = c·e^(−φ) диффундирует с постоянным D,
   c = u·e^φ, масса перенормируется. Эталон — конечные объёмы со схемой Шарфеттера–Гуммеля
   (точный поток при линейном φ на грани), неявно, мелкими шагами. φ = V_H σ_h / RT от пластинки.

Запуск: python fe/diff_check.py [--no-ccx] [--b-only]
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ca_hydride import Elastic, plate_eigen  # noqa: E402
from plate_fe import CCX  # noqa: E402

C_HYD = 15663.0
D0, QD = 7.9e-7, 44.4e3
WORK = os.environ.get("DIFF_WORK", "/tmp/diff_check")


def Dum2(TC):
    return D0 * np.exp(-QD / (8.314 * (TC + 273.15))) * 1e12


def setup(N=240, dx=0.4, psi=0.0, half=2.5, h=0.6):
    L = N * dx
    ys, xs, fr, comps = plate_eigen((N, N), dx, np.array([L / 2, L / 2]), psi, half, h)
    d = np.zeros((N, N))
    d[np.ix_(ys, xs)] = fr * C_HYD
    return L, d, (ys, xs, fr, comps)


def k2grid(N, dx):
    ky = 2 * np.pi * np.fft.fftfreq(N, dx)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(N, dx)[None, :]
    return kx ** 2 + ky ** 2


def fft_step(c, D, t, dx):
    return np.fft.irfft2(np.fft.rfft2(c) * np.exp(-D * k2grid(c.shape[0], dx) * t), s=c.shape)


def _ierf(u):
    return u * erf(u) + np.exp(-u * u) / np.sqrt(np.pi)


def exact_cellavg(d, dx, D, t, nimg=4):
    """Точное решение для кусочно-постоянной ямы d (ppm), усреднённое по клеткам."""
    N = d.shape[0]
    L = N * dx
    s = np.sqrt(4 * D * t)
    edges = np.arange(N + 1) * dx

    def F(a0, a1):
        # среднее по клетке [e_i, e_{i+1}] от ½[erf((a1−x)/s) − erf((a0−x)/s)], с образами
        out = np.zeros(N)
        for m in range(-nimg, nimg + 1):
            for a, sg in ((a1 + m * L, 1.0), (a0 + m * L, -1.0)):
                # ∫ erf((a−x)/s) dx от e_i до e_{i+1} = s·[I((a−e_i)/s) − I((a−e_{i+1})/s)]
                I = _ierf((a - edges) / s)
                out += sg * 0.5 * s * (I[:-1] - I[1:]) / dx
        return out

    res = np.zeros((N, N))
    for iy, ix in zip(*np.nonzero(d)):
        fy = F(iy * dx, (iy + 1) * dx)
        fx = F(ix * dx, (ix + 1) * dx)
        res += d[iy, ix] * np.outer(fy, fx)
    return res


def ccx_heat(d, dx, D, t, ninc, name):
    """Яма d как начальная «температура» (со знаком минус), DC3D8, узлы в центрах клеток."""
    N = d.shape[0]
    os.makedirs(WORK, exist_ok=True)
    # узлы: i, j = 0..N (N+1 узлов, правый/верхний ряд — копия левого/нижнего)
    M = N + 1
    nid = lambda i, j, k: 1 + i + M * j + M * M * k          # i — x, j — y, k — слой z  # noqa: E731
    lines = ["*NODE, NSET=NALL"]
    for k in range(2):
        for j in range(M):
            for i in range(M):
                lines.append(f"{nid(i, j, k)}, {(i + 0.5) * dx:.6f}, {(j + 0.5) * dx:.6f}, {k * dx:.6f}")
    lines.append("*ELEMENT, TYPE=DC3D8, ELSET=EALL")
    e = 1
    for j in range(N):
        for i in range(N):
            n = [nid(i, j, 0), nid(i + 1, j, 0), nid(i + 1, j + 1, 0), nid(i, j + 1, 0),
                 nid(i, j, 1), nid(i + 1, j, 1), nid(i + 1, j + 1, 1), nid(i, j + 1, 1)]
            lines.append(f"{e}, " + ", ".join(map(str, n)))
            e += 1
    eqs = []
    for k in range(2):
        for j in range(N):                       # правый ряд → левый
            eqs.append((nid(N, j, k), nid(0, j, k)))
        for i in range(N):                       # верхний ряд → нижний
            eqs.append((nid(i, N, k), nid(i, 0, k)))
        eqs.append((nid(N, N, k), nid(0, 0, k)))
    lines.append("*EQUATION")
    for a, b in eqs:
        lines += ["2", f"{a}, 11, 1., {b}, 11, -1."]
    lines += ["*MATERIAL, NAME=M", "*CONDUCTIVITY", f"{D:.8g}", "*SPECIFIC HEAT", "1.", "*DENSITY", "1.",
              "*SOLID SECTION, ELSET=EALL, MATERIAL=M", "*INITIAL CONDITIONS, TYPE=TEMPERATURE"]
    for k in range(2):
        for j in range(M):
            for i in range(M):
                lines.append(f"{nid(i, j, k)}, {-d[j % N, i % N]:.8g}")
    lines += ["*STEP, INC=1000000", "*HEAT TRANSFER, DIRECT", f"{t / ninc:.8g}, {t:.8g}",
              f"*NODE PRINT, NSET=NALL, FREQUENCY={ninc}", "NT", "*END STEP"]
    path = os.path.join(WORK, name + ".inp")
    open(path, "w").write("\n".join(lines) + "\n")
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", CCX_NPROC_EQUATION_SOLVER="1")
    r = subprocess.run([CCX, "-i", name], cwd=WORK, env=env, capture_output=True, text=True)
    if "Job finished" not in r.stdout and "Total CalculiX Time" not in r.stdout:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-1000:])
    txt = open(os.path.join(WORK, name + ".dat")).read()
    blk = txt.split("temperatures")[-1].splitlines()
    T = {}
    for ln in blk:
        p = ln.split()
        if len(p) == 2:
            try:
                T[int(p[0])] = float(p[1])
            except ValueError:
                pass
    out = np.array([[T[nid(i, j, 0)] for i in range(N)] for j in range(N)])
    return -out, time.time() - t0


def sg_matrix(phi, D, dx):
    """Конечные объёмы, поток Шарфеттера–Гуммеля: dc/dt = A c (масса сохраняется точно)."""
    N = phi.shape[0]
    idx = np.arange(N * N).reshape(N, N)
    rows, cols, vals = [], [], []
    for sh in ((0, 1), (1, 0)):
        if sh == (0, 1):
            nb = np.roll(idx, -1, axis=1); pn = np.roll(phi, -1, axis=1)
        else:
            nb = np.roll(idx, -1, axis=0); pn = np.roll(phi, -1, axis=0)
        dphi = pn - phi
        with np.errstate(invalid="ignore", divide="ignore"):
            G = np.where(np.abs(dphi) > 1e-12, dphi / (np.exp(-phi) - np.exp(-pn)), np.exp(phi))
        G = D / dx ** 2 * G
        a, b = idx.ravel(), nb.ravel()
        g = G.ravel()
        ea, eb = np.exp(-phi).ravel(), np.exp(-pn).ravel()
        # поток из a в b: g·(u_a − u_b), u = c·e^(−φ)
        rows += [a, a, b, b]
        cols += [a, b, b, a]
        vals += [-g * ea, g * eb, -g * eb, g * ea]
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(N * N, N * N))


def implicit(A, c, t, nsub):
    dt = t / nsub
    lu = spla.splu((sp.identity(A.shape[0], format="csc") - dt * A).tocsc())
    x = c.ravel().copy()
    for _ in range(nsub):
        x = lu.solve(x)
    return x.reshape(c.shape)


def slotboom_fft(c, phi, D, t, dx):
    mass = c.sum()
    u = fft_step(c * np.exp(-phi), D, t, dx)
    out = u * np.exp(phi)
    return out * mass / out.sum()


def part_a(use_ccx=True):
    dx = 0.4
    cases = [  # T, dt: обычный шаг 1 °C/мин; укороченный ×0.05; быстрое охлаждение + укороченный
        dict(T=250, t=30.0, d="0deg"), dict(T=250, t=1.5, d="0deg"),
        dict(T=150, t=0.1, d="0deg"), dict(T=150, t=0.02, d="0deg"), dict(T=150, t=0.1, d="30deg"),
    ]
    out = []
    for cs in cases:
        D = Dum2(cs["T"])
        N = 240 if cs["t"] > 10 else 120             # ячейка 96 или 48 мкм: больше диффузионной длины
        _, dd, _ = setup(N=N, dx=dx, psi=0.0 if cs["d"] == "0deg" else np.radians(30))
        ex = exact_cellavg(dd, dx, D, cs["t"])
        ff = fft_step(dd, D, cs["t"], dx)
        far = ~ndimage.binary_dilation(dd > 0, iterations=int(round(1.0 / dx)))   # дальше 1 мкм от пластинки
        mtx = dd == 0
        r = dict(cs, D=D, Ldiff=float(np.sqrt(2 * D * cs["t"])), peak=float(ex.max()),
                 fft_maxerr=float(np.abs(ff - ex).max()), fft_rms=float(np.sqrt(((ff - ex) ** 2).mean())),
                 fft_far=float(np.abs(ff - ex)[far].max()),
                 # «перелёт»: концентрация в матрице выше фона (яма с обратным знаком) — ложное пересыщение
                 ex_over=float(max(0.0, -ex[mtx].min())), fft_over=float(max(0.0, -ff[mtx].min())),
                 mass_err=float(abs(ff.sum() - dd.sum()) / dd.sum()))
        if use_ccx:
            for ninc in ((50,) if N > 120 else (50, 400)):
                fe, sec = ccx_heat(dd, dx, D, cs["t"], ninc, f"heat_{cs['T']}_{cs['t']}_{cs['d']}_{ninc}")
                r[f"ccx{ninc}_maxerr"] = float(np.abs(fe - ex).max())
                r[f"ccx{ninc}_rms"] = float(np.sqrt(((fe - ex) ** 2).mean()))
                r[f"ccx{ninc}_far"] = float(np.abs(fe - ex)[far].max())
                r[f"ccx{ninc}_over"] = float(max(0.0, -fe[mtx].min()))
                r[f"ccx{ninc}_sec"] = round(sec, 1)
                if ninc == 400:
                    m = N // 2
                    r["prof"] = dict(ex=ex[m, m - 20:m + 20].tolist(), ff=ff[m, m - 20:m + 20].tolist(),
                                     fe=fe[m, m - 20:m + 20].tolist())
        out.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "prof"}, flush=True)
    return out


def part_b():
    dx, N = 0.4, 240
    L, d, (ys, xs, fr, comps) = setup(N=N, dx=dx, psi=0.0)
    el = Elastic((N, N), dx, 90e3, 0.34, free_z=True)
    Ein = [np.zeros((N, N)) for _ in range(4)]
    for a, v in zip(Ein, comps):
        a[np.ix_(ys, xs)] += fr * v
    S = el.stress(*Ein)
    sh_raw = (S[0] + S[1] + S[3]) / 3.0
    out = []
    for TC, t in ((250, 30.0), (250, 1.5), (150, 1.5), (150, 0.1)):
        D = Dum2(TC)
        for cap in (300.0, 1e9):
            sh = np.clip(sh_raw, -cap, cap)
            phi = 1.7e-6 * 1e6 * sh / (8.314 * (TC + 273.15))
            c0 = 100.0 - d                                   # яма свежей пластинки в растворе 100 ppm
            A = sg_matrix(phi, D, dx)
            A0 = sg_matrix(np.zeros_like(phi), D, dx)
            nsub = 400
            ref = implicit(A, c0, t, nsub)
            ref0 = implicit(A0, c0, t, nsub)
            slb = slotboom_fft(c0, phi, D, t, dx)
            # исправление в автомате: в клетках гидрида (счётная яма) φ = 0
            phim = np.where(d > 0, 0.0, phi)
            refm = implicit(sg_matrix(phim, D, dx), c0, t, nsub)
            slbm = slotboom_fft(c0, phim, D, t, dx)
            fick = fft_step(c0, D, t, dx)
            # равновесие: c ∝ e^φ при той же массе
            eq = np.exp(phi) * c0.sum() / np.exp(phi).sum()
            mtx = ~ndimage.binary_dilation(d > 0, iterations=int(round(1.0 / dx)))   # дальше 1 мкм от пластинки
            r = dict(T=TC, t=t, cap=cap, Ldiff=float(np.sqrt(2 * D * t)), sh_min=float(sh.min()), sh_max=float(sh.max()),
                     stress_effect=float(np.abs(ref - ref0)[mtx].max()),       # насколько напряжение вообще меняет c
                     slotboom_err=float(np.abs(slb - ref)[mtx].max()),
                     masked_effect=float(np.abs(refm - ref0)[mtx].max()),
                     masked_err=float(np.abs(slbm - refm)[mtx].max()),
                     fick_vs_fv=float(np.abs(fick - ref0)[mtx].max()),
                     ref_vs_eq=float(np.abs(ref - eq)[mtx].max()),
                     slb_vs_eq=float(np.abs(slb - eq)[mtx].max()))
            if cap == 300.0 and t == 1.5:
                r["prof"] = dict(ref=ref[120, 100:160].tolist(), slb=slb[120, 100:160].tolist(),
                                 ref0=ref0[120, 100:160].tolist(), eq=eq[120, 100:160].tolist(),
                                 refm=refm[120, 100:160].tolist(), slbm=slbm[120, 100:160].tolist())
            out.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "prof"}, flush=True)
    return out


if __name__ == "__main__":
    use_ccx = "--no-ccx" not in sys.argv
    dst = os.path.join(HERE, "diff_check.json")
    A = json.load(open(dst))["A"] if "--b-only" in sys.argv else part_a(use_ccx)
    B = part_b()
    json.dump(dict(A=A, B=B), open(dst, "w"), indent=1)
    print("→", dst)
