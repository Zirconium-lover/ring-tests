"""Кинетический движок в 3D: то же, что ca_kinetic.py (время, охлаждение, водород с диффузией, одновременное
зарождение, скорость зарождения по теории зарождения относительно TSSP), но в кубе с трёхмерными зёрнами.

Оси: x — TD (окружность), y — ND (радиус), z — L (ось трубы); массивы [ix, iy, iz], куб периодический.
• Зёрна и ось c — как в ca3d.make_grains3 (наклон к TD ±χ0 с разбросом, к оси трубы — разброс χ_sL).
• Пластинка — диск толщиной h с нормалью n = ось c (базис), растёт в своей плоскости до границы зерна,
  соседа или R_max (ca3d.grow_disc).
• Сдвиг температуры выпадения под нагрузкой — по нормальному к пластинке напряжению (Vizcaíno 2014):
      dT = app_dT·(σθ·n_x² + σz·n_z²) + bias_dT·n_y² (+ разброс по зёрнам),
  в движущую силу — через наклон TSSP: n_H·Q/T. Осевое σz действует на пластинки с осевой составляющей
  нормали; окружная и радиальная (нормаль по θ) пластинки обе содержат ось трубы и от σz через энергию
  несоответствия не различаются.
• Поле соседей — 3D-микроупругость (ca3d.stress_of_field) раз в шаг на все зародыши шага, g = σ:ε*/ε_n с
  потолком ±sigma_cap.
Сечения как шлифы (section_measures): поперечное r–θ (⊥ оси трубы, как у Lepine и Son), продольное r–z
(⊥ TD) и плоскость поверхности θ–z (⊥ ND, «вид сверху» на лист, как у Cinbiz для двухосных образцов)."""
from dataclasses import dataclass
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi
from ca3d import P3, make_grains3, Elastic3, stress_of_field, disc_fraction, grow_disc, eps_star, plate_chord, \
    simon_w, EPS_N, EPS_T, WORKERS
from ca_kinetic import N_H_MPa_per_K, C_HYD
from thermo import TSS, c_line


@dataclass
class K3(P3):
    size_um: float = 48.0
    dx: float = 0.5
    sigma_cap: float = 180.0
    cap_local_only: bool = False
    H_ppm: float = 178.0
    T_max: float = 415.0
    T_end: float = 100.0
    rate: float = 3.0
    lines: str = "E635"
    B: float = 80.0
    Delta0: float = 15.0
    nu_c: float = 1e-6
    lam_max: float = 6.0
    dT_max: float = 0.5
    D0: float = 7.9e-7
    QD: float = 44.4e3
    t_grow: float = 10.0
    app_dT: float = 0.08
    bias_dT: float = 12.4
    dT_s: float = 0.0
    R_min: float = 0.6                      # диск меньше — зародыш не растёт (в 2D L_min = 1.2 мкм)


def run3d_kinetic(p: K3, verbose=False):
    L = TSS[p.lines] if isinstance(p.lines, str) else p.lines
    rng = np.random.default_rng(p.seed)
    grains, cax = make_grains3(p, rng)
    n = grains.shape[0]
    el = Elastic3(n, p.dx, p.E, p.nu)
    e_new = [np.asarray(a, np.float32)[grains] for a in eps_star(cax)]  # ε* возможной пластинки в клетке
    dTg = p.app_dT * (p.sigma_app * cax[:, 0] ** 2 + p.sigma_axial * p.sigma_app * cax[:, 2] ** 2) \
        + p.bias_dT * cax[:, 1] ** 2
    if p.dT_s > 0:
        dTg = dTg + np.random.default_rng([p.seed, 11]).normal(0.0, p.dT_s, len(cax))
    dT_map = dTg[grains].astype(np.float32)
    Q_over_R = np.log(10.0) * 1000.0 * L["TSSP"][1]
    S = [np.zeros((n, n, n), np.float32) for _ in range(6)]
    hyd = np.zeros((n, n, n), np.float32)
    occ = np.zeros((n, n, n), bool)
    tried = np.zeros((n, n, n), bool)
    blocked_geo = np.zeros((n, n, n), bool)     # диск не помещается (зерно, соседи) — навсегда: занятых только больше
    kk2 = (el.kx ** 2 + el.ky ** 2 + el.kz ** 2).astype(np.float32)        # для диффузии (в нуле — 0)
    plates = []
    st = ndi.generate_binary_structure(3, 1)

    def add_batch(batch):
        E6 = [np.zeros((n, n, n), np.float32) for _ in range(6)]
        for ax, fr, nv in batch:
            sel = np.ix_(*ax)
            hyd[sel] = np.clip(hyd[sel] + fr, 0, 1)
            for a, v in zip(E6, eps_star(nv)):
                a[sel] += (fr * v).astype(np.float32)
        for a, d in zip(S, stress_of_field(el, E6)):
            a += d
        occ[:] = ndi.binary_dilation(hyd > 0.2, structure=st, iterations=1)

    c0 = float(min(p.H_ppm, c_line(p.T_max, L["TSSD"])))
    c = np.full((n, n, n), c0, np.float32)
    T, t = p.T_max, 0.0
    q = p.rate / 60.0
    hist = dict(T=[], c_mean=[], n=[])
    w2 = (1, 1, 1, 2, 2, 2)
    while T > p.T_end:
        Tk = T + 273.15
        g = sum(w2[i] * e_new[i] * S[i] for i in range(6)) / EPS_N
        np.clip(g, -p.sigma_cap, p.sigma_cap, out=g)
        lnTSSP = np.log(c_line(T, L["TSSP"]))
        Delta = p.Delta0 + N_H_MPa_per_K * Tk * (np.log(np.clip(c, 1e-6, None)) - lnTSSP) + EPS_N * g \
            + N_H_MPa_per_K * Q_over_R / Tk * dT_map
        del g
        ok = (Delta > 0) & ~occ & ~tried & ~blocked_geo
        D = p.D0 * np.exp(-p.QD / (8.314 * Tk)) * 1e12
        rad_um = np.sqrt(2 * D * p.t_grow)
        c_eq = c_line(T, L["TSSD"])
        dt = p.dT_max / q
        lam_tot, n_new = 0.0, 0
        if ok.any():
            lnr = np.full((n, n, n), -np.inf, np.float32)
            lnr[ok] = np.log(p.nu_c) - p.B * ((p.Delta0 / Delta[ok]) ** 2 - 1.0)
            m = float(lnr[ok].max())
            w = np.exp((lnr - m).astype(np.float64)).ravel()
            del lnr
            lam_tot = float(np.exp(m) * w.sum() * dt)
            if lam_tot > p.lam_max:
                k_dt = max(p.lam_max / lam_tot, 0.05)
                dt *= k_dt
                lam_tot = min(lam_tot * k_dt, 4 * p.lam_max)
            n_new = rng.poisson(lam_tot) if lam_tot > 0 else 0
        del Delta
        if n_new:
            idx = rng.choice(w.size, size=n_new, p=w / w.sum())
            batch = []
            occ_b = occ.copy()
            spent = 0.0
            Rw = int(np.clip(2 * rad_um, 2.0, 0.5 * p.size_um) / p.dx)
            # запас водорода над TSSD в окне (2Rw+1)³ вокруг каждой клетки — один раз за шаг (с обеднёнными
            # клетками); внутри шага расход зародышей вычитается из запаса их окрестности приближённо
            wsz = min(2 * Rw + 1, n)
            avail_f = ndi.uniform_filter(c - np.float32(c_eq), size=wsz, mode="wrap") * float(wsz ** 3)
            for k in rng.permutation(idx):
                ci = np.unravel_index(k, (n, n, n))
                if occ_b[ci]:
                    continue
                nv = cax[grains[ci]]
                cc, R = grow_disc(ci, nv, grains, occ_b, p)
                if R < p.R_min:                          # мешают зерно и пластинки (и этого шага) — навсегда
                    blocked_geo[ci] = True
                    continue
                avail = max(0.0, float(avail_f[ci]) - spent)                     # ppm·клетка
                R = min(R, np.sqrt(avail * p.dx ** 3 / C_HYD / (np.pi * p.h_um)))
                if R < p.R_min:
                    tried[ci] = True
                    continue
                ax, fr = disc_fraction(n, p.dx, cc, nv, R, p.h_um)
                sel = np.ix_(*ax)
                occ_b[sel] |= fr > 0.2
                c[sel] -= (fr * C_HYD).astype(np.float32)
                spent += float(fr.sum()) * C_HYD                                 # грубо: весь расход шага — всем
                batch.append((ax, fr, nv))
                plates.append(dict(c=cc, n=nv.copy(), R=float(R), grain=int(grains[ci]), T=float(T), t=float(t)))
            if batch:
                add_batch(batch)
                tried[:] = False
        c = sfft.irfftn(sfft.rfftn(c, workers=WORKERS) * np.exp(-D * kk2 * dt), s=(n,) * 3,
                        workers=WORKERS).astype(np.float32)
        t += dt; T -= q * dt
        hist["T"].append(T); hist["c_mean"].append(float(c.mean())); hist["n"].append(len(plates))
        if verbose and n_new:
            print(f"T {T:6.1f}  c̄ {c.mean():6.1f}  пластинок {len(plates)}", flush=True)
    return dict(params=p, plates=plates, hyd=hyd, grains=grains, cax=cax, c=c,
                hist={k: np.array(v) for k, v in hist.items()})


# ------------------------------------------------------------------ сечения как шлифы
_PERM = {"rt": (0, 1, 2), "rz": (1, 2, 0), "surf": (0, 2, 1)}   # новые (x', y', z'): z' — нормаль сечения


def section_measures(res, n_sec=16):
    """Следы дисков в сечениях. rt — ⊥ оси трубы (след к TD; RHF по весам Simon/PROPHET 0 / 0.5 / 1 при
    40° / 65°), rz — ⊥ TD (след к оси трубы, те же веса), surf — ⊥ ND (плоскость поверхности): доля длины
    следов пластинок круче 45° к поверхности (|n_ND| < cos 45°) — «выходящие из плоскости», как видит
    шлиф плоскости листа у Cinbiz (лежащие в плоскости там не видны)."""
    p = res["params"]
    Lc = p.size_um
    zs = (np.arange(n_sec) + 0.5) / n_sec * Lc
    out = {}
    for key, pm in _PERM.items():
        pm = list(pm)
        num = den = 0.0
        for q in res["plates"]:
            nv = q["n"][pm]; cc = q["c"][pm]
            u = np.cross(nv, [0, 0, 1.0] if abs(nv[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u)
            for z in zs:
                tr = plate_chord(cc, nv, u, q["R"], q["R"], z, Lc)
                if tr is None:
                    continue
                d, half = tr[1], tr[2]
                if key == "rt":         # (x' = TD, y' = ND): угол следа к TD
                    wgt = simon_w(np.degrees(np.arccos(min(1.0, abs(d[0])))))
                elif key == "rz":       # (x' = ND, y' = L): угол следа к оси трубы
                    wgt = simon_w(np.degrees(np.arccos(min(1.0, abs(d[1])))))
                else:                   # поверхность: пластинка круче 45° к ней
                    wgt = 1.0 if abs(q["n"][1]) < np.cos(np.radians(45)) else 0.0
                num += 2 * half * wgt; den += 2 * half
        out["RHF_" + key] = num / den if den else np.nan
    V = np.array([np.pi * q["R"] ** 2 for q in res["plates"]]) if res["plates"] else np.zeros(0)
    if len(V):
        N = np.array([q["n"] for q in res["plates"]])
        for name, i in (("TD", 0), ("ND", 1), ("L", 2)):
            out["frac_n" + name] = float((V * (N[:, i] ** 2 > 0.5)).sum() / V.sum())
    return out
