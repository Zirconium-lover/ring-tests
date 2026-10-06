"""Кинетический вариант автомата: время, охлаждение, одновременное зарождение, диффузия водорода.

Отличие от ca_hydride.run (пластинки строго по одной, каждая видит все прежние выросшими):
  • время идёт шагами, температура падает со скоростью охлаждения q;
  • растворённый водород — поле c(x), диффундирует с D(T) (Kearns: D = 7.9·10⁻⁷ exp(−44.4 кДж/RT) м²/с);
  • за шаг зародыши возникают сразу во многих местах (пуассоновское число, вероятность ∝ скорости);
    зародыши одного шага друг друга не видят — поле напряжений пересчитывается раз в шаг;
  • зародыш сразу вырастает в пластинку (рост по диффузии — секунды) и забирает свой водород
    из ячеек пластинки; дальше обеднение расходится диффузией.

Скорость зарождения в клетке (1/с):
    r = ν_c · exp(−B·[(Δ0/Δ)² − 1]),   Δ = Δ0 + n_H·R·T·ln(c/TSSP(T)) + ε_n·g   (МПа, Δ > 0)
n_H·R·T·ln(c/TSSP) — избыток химической движущей силы над линией выделения TSSP (n_H = 1.01·10⁵ моль H/м³
гидрида); ε_n·g = σ:ε* — работа поля над несоответствием новой пластинки (соседи с потолком,
приложенное напряжение, добавка g_extra). На линии TSSP без напряжений Δ = Δ0, r = ν_c.
B — барьер в kT на линии TSSP (классическая теория зарождения: ΔG* ∝ 1/Δ²).
Избирательность по g вблизи линии: d ln r/dg ≈ 2·B·ε_n/Δ0.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
import ca_hydride as ca
from ca_hydride import Params, Elastic, make_grains, plate_eigen, eigen_components_for, grow
from thermo import TSS, c_line, H_IN_HYDRIDE_PPM, VOL_HYD_PER_ZR

N_H_MPa_per_K = 1.01e5 * 8.314 / 1e6          # n_H·R, МПа/К
C_HYD = H_IN_HYDRIDE_PPM / VOL_HYD_PER_ZR       # водород в единице площади гидрида, ppm металла


@dataclass
class KParams(Params):
    H_ppm: float = 180.0
    T_max: float = 415.0            # °C
    T_end: float = 100.0            # °C — ниже почти весь водород уже выпал
    rate: float = 3.0               # скорость охлаждения, °C/мин
    lines: str = "E635"             # линии растворимости (thermo.TSS)
    B: float = 40.0                 # барьер на линии TSSP, kT
    Delta0: float = 30.0            # избыток движущей силы на линии TSSP, МПа
    nu_c: float = 1e-6              # скорость зарождения в клетке на линии TSSP, 1/с
    lam_max: float = 6.0            # не больше стольких зародышей (в среднем) за шаг
    dT_max: float = 0.5             # шаг по температуре, °C
    D0: float = 7.9e-7              # м²/с
    QD: float = 44.4e3              # Дж/моль
    t_grow: float = 10.0            # время роста пластинки, с: водород собирается с радиуса √(2·D·t_grow)
    stress_diff: bool = False       # диффузия водорода и в растянутые области: c ∝ exp(V_H·σ_h/RT) в равновесии
    V_H: float = 1.7e-6             # парциальный мольный объём водорода в Zr, м³/моль
    sh_cap: float = 300.0           # ограничение гидростатического напряжения от соседей, МПа


def run_kinetic(p: KParams, verbose=False, callback=None):
    ca.EPS_N, ca.EPS_T = p.eps_n, p.eps_t
    EPS_N, EPS_T = p.eps_n, p.eps_t
    L = TSS[p.lines] if isinstance(p.lines, str) else p.lines
    rng = np.random.default_rng(p.seed)
    grains, gpsi = make_grains(p, rng)
    ny, nx = grains.shape
    el = Elastic((ny, nx), p.dx, p.E, p.nu)
    psi_map = gpsi[grains]
    e11n, e22n, e12n = eigen_components_for(psi_map)
    e33n = EPS_T
    yfrac = (np.arange(ny) + 0.5) / ny
    sapp = (p.sigma_app + p.sigma_app_grad * (0.5 - yfrac))[:, None] * np.ones((1, nx))
    e_mean = 0.5 * (EPS_N + EPS_T)
    g_app = sapp * (e11n - e_mean) / EPS_N
    if p.app_dg is not None:            # вклад нагрузки по МКЭ (fe/hill_table.json), как в ca_hydride
        g_app = np.interp(sapp, *p.app_dg) * (np.sin(psi_map) ** 2 - 0.5)
    lnc_gorsky = p.v_h_over_rt * sapp / 3.0 if p.v_h_over_rt else 0.0
    S11 = np.zeros((ny, nx)); S22 = np.zeros_like(S11); S12 = np.zeros_like(S11); S33 = np.zeros_like(S11)
    occ = np.zeros((ny, nx), bool)
    hyd = np.zeros((ny, nx))
    tried = np.zeros((ny, nx), bool)
    plates = []
    ky = 2 * np.pi * np.fft.fftfreq(ny, p.dx)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(nx, p.dx)[None, :]
    k2 = kx ** 2 + ky ** 2

    def add_batch(batch, deplete=True):
        """Пластинки одного шага: одно решение упругой задачи на всех, водород — из ячеек пластинок."""
        Ein = [np.zeros((ny, nx)) for _ in range(4)]
        sink = np.zeros((ny, nx))
        for c, psi, half in batch:
            ys, xs, fr, comps = plate_eigen((ny, nx), p.dx, c, psi, half, p.h_um)
            sel = np.ix_(ys, xs)
            new = np.clip(hyd[sel] + fr, 0, 1) - hyd[sel]
            hyd[sel] += new
            sink[sel] += new
            for a, v in zip(Ein, comps):
                a[sel] += fr * v
        d11, d22, d12, d33 = el.stress(*Ein)
        S11[:] += d11; S22[:] += d22; S12[:] += d12; S33[:] += d33
        occ[:] = ndi.binary_dilation(hyd > 0.2, iterations=1)
        return sink * C_HYD if deplete else 0.0

    # начальное состояние: растворилось до TSSD(T_max), остальное — нерастворившиеся пластинки
    c0 = float(min(p.H_ppm, c_line(p.T_max, L["TSSD"])))
    c = np.full((ny, nx), c0)
    if p.init_plates is not None and len(p.init_plates):
        init = [(np.array([q[0], q[1]]), q[2], q[3]) for q in p.init_plates]
        add_batch(init, deplete=False)
        for (cc, psi, half) in init:
            iy, ix = int(cc[0] / p.dx) % ny, int(cc[1] / p.dx) % nx
            plates.append(dict(c=cc, psi=float(psi), half=float(half), grain=int(grains[iy, ix]), T=None, t=None, init=True))
    T, t = p.T_max, 0.0
    q = p.rate / 60.0                                   # °C/с
    hist = dict(t=[], T=[], c_mean=[], n=[], frac=[])
    while T > p.T_end:
        Tk = T + 273.15
        g = (e11n * S11 + e22n * S22 + 2 * e12n * S12 + e33n * S33) / EPS_N * p.kappa
        if p.cap_local_only:
            mtx = hyd < 0.2
            Sm = [float(a[mtx].mean()) for a in (S11, S22, S12, S33)]
            g_mean = (e11n * Sm[0] + e22n * Sm[1] + 2 * e12n * Sm[2] + e33n * Sm[3]) / EPS_N * p.kappa
            g = np.clip(g - g_mean, -p.sigma_cap, p.sigma_cap) + g_mean + g_app
        else:
            g = np.clip(g, -p.sigma_cap, p.sigma_cap) + g_app
        if p.g_extra is not None:
            g = g + p.g_extra
        lnc = np.log(np.clip(c, 1e-6, None)) + lnc_gorsky
        Delta = p.Delta0 + N_H_MPa_per_K * Tk * (lnc - np.log(c_line(T, L["TSSP"]))) + EPS_N * g
        ok = (Delta > 0) & ~occ & ~tried
        lnr = np.full((ny, nx), -np.inf)
        lnr[ok] = np.log(p.nu_c) - p.B * ((p.Delta0 / Delta[ok]) ** 2 - 1.0)
        D = p.D0 * np.exp(-p.QD / (8.314 * Tk)) * 1e12   # мкм²/с
        rad_um = np.sqrt(2 * D * p.t_grow)
        c_eq = c_line(T, L["TSSD"])
        dt = p.dT_max / q
        lam_tot = 0.0
        if ok.any():
            m = lnr[ok].max()
            w = np.exp(lnr - m)                         # относительные веса
            lam_tot = float(np.exp(m) * w.sum() * dt)
            if lam_tot > p.lam_max:                     # всплеск зарождения — шаг короче (не короче 1/20 обычного)
                k_dt = max(p.lam_max / lam_tot, 0.05)
                dt *= k_dt
                lam_tot = min(lam_tot * k_dt, 4 * p.lam_max)
        n_new = rng.poisson(lam_tot) if lam_tot > 0 else 0
        if n_new:
            pr = (w / w.sum()).ravel()
            idx = rng.choice(pr.size, size=n_new, p=pr)
            batch = []
            occ_b = occ.copy()
            for k in rng.permutation(idx):
                ci = np.unravel_index(k, (ny, nx))
                if occ_b[ci]:
                    continue
                psi = psi_map[ci]
                cc, half = grow(ci, psi, grains, occ_b, p)
                # водорода сверх TSSD в окрестности радиусом ~ диффузионной длины роста хватает не на всё
                R = int(np.clip(2 * rad_um, 2.0, 60.0) / p.dx)
                yy = np.arange(ci[0] - R, ci[0] + R + 1) % ny; xx = np.arange(ci[1] - R, ci[1] + R + 1) % nx
                avail = float(np.clip(c[np.ix_(yy, xx)] - c_eq, 0, None).sum())       # ppm·клетка
                half = min(half, avail / C_HYD * p.dx ** 2 / (2 * p.h_um))
                if 2 * half < p.L_min:
                    tried[ci] = True
                    continue
                batch.append((cc, psi, half))
                ys, xs, fr, _ = plate_eigen((ny, nx), p.dx, cc, psi, half, p.h_um)
                occ_b[np.ix_(ys, xs)] |= fr > 0.2       # зародыши шага не перекрываются
                c[np.ix_(ys, xs)] -= fr * C_HYD          # водород пластинки — сразу, дальше разойдётся диффузией
                plates.append(dict(c=cc, psi=float(psi), half=float(half), grain=int(grains[ci]), T=float(T), t=float(t), init=False))
            if batch:
                add_batch(batch, deplete=False)
                tried[:] = False
        # диффузия водорода за шаг; с напряжениями — по химическому потенциалу RT ln c − V_H σ_h
        # (переменные Слотбома: u = c·e^(−φ) диффундирует, c = u·e^φ; масса сохраняется перенормировкой)
        if p.stress_diff:
            sh = np.clip((S11 + S22 + S33) / 3.0, -p.sh_cap, p.sh_cap)
            phi = p.V_H * 1e6 * sh / (8.314 * Tk)
            mass = c.sum()
            u = np.fft.irfft2(np.fft.rfft2(c * np.exp(-phi)) * np.exp(-D * k2 * dt), s=(ny, nx))
            c = u * np.exp(phi)
            c *= mass / c.sum()
        else:
            c = np.fft.irfft2(np.fft.rfft2(c) * np.exp(-D * k2 * dt), s=(ny, nx))
        t += dt; T -= q * dt
        hist["t"].append(t); hist["T"].append(T); hist["c_mean"].append(float(c.mean()))
        hist["n"].append(len(plates)); hist["frac"].append(float(hyd.mean()))
        if callback is not None:
            callback(plates, lnr, hyd, c, T)
        if verbose and n_new:
            print(f"T {T:6.1f}  c̄ {c.mean():6.1f} (TSSP {c_line(T, L['TSSP']):6.1f})  пластинок {len(plates)}")
    return dict(params=p, plates=plates, hyd=hyd, grains=grains, gpsi=gpsi, S=(S11, S22, S12, S33),
                c=c, hist={k: np.array(v) for k, v in hist.items()})
