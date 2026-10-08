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
import scipy.fft as sfft
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
    rate2: float = 0.0              # °C/мин ниже T_rate2 (двухступенчатое охлаждение Desquines 2014: 5 → 0.4); 0 — нет
    T_rate2: float = 0.0            # °C
    lines: str = "E635"             # линии растворимости (thermo.TSS)
    B: float = 40.0                 # барьер на линии TSSP, kT
    Delta0: float = 30.0            # избыток движущей силы на линии TSSP, МПа
    nu_c: float = 1e-6              # скорость зарождения в клетке на линии TSSP, 1/с
    nuc_um: float = 0.115           # мкм: поле для зарождения усредняется по размеру зародыша (гаусс, σ = nuc_um;
                                    # 0.115 = 0.4/√12 — как ячейка 0.4 мкм); 0 — по клетке. С occ_um модель сходится
                                    # по сетке (0.4 → 0.2 мкм: Fn45 и доля межзёренных в пределах разброса)
    occ_um: float = 0.4             # мкм: запретная зона вокруг пластинок для новых зародышей; 0 — одна клетка
    nu_dx: float = 0.4              # мкм: ν_c задана для клетки такого размера; на другой сетке скорость
                                    # пересчитывается на площадь (тело зерна, ∝ dx²) и на длину (грань, ∝ dx),
                                    # чтобы соотношение мест «грань/тело» не зависело от сетки (0 — по клетке)
    lam_max: float = 6.0            # не больше стольких зародышей (в среднем) за шаг
    dT_max: float = 0.5             # шаг по температуре, °C
    D0: float = 7.9e-7              # м²/с
    QD: float = 44.4e3              # Дж/моль
    t_grow: float = 10.0            # время роста пластинки, с: водород собирается с радиуса √(2·D·t_grow)
    stress_diff: bool = False       # диффузия водорода и в растянутые области: c ∝ exp(V_H·σ_h/RT) в равновесии
    V_H: float = 1.7e-6             # парциальный мольный объём водорода в Zr, м³/моль
    sh_cap: float = 300.0           # ограничение гидростатического напряжения от соседей, МПа
    app_dT: float = 0.0             # °C/МПа: сдвиг температуры выпадения на 1 МПа нормального к пластинке
                                    # напряжения (Vizcaíno и др. 2014: 0.08 ± 0.02 в зёрнах с осью c вдоль нагрузки);
                                    # > 0 — заменяет упругий вклад нагрузки (g_app) этим измеренным
    halo_cap: float = 0.0           # предел суммарной пластической деформации ореолов в клетке (эквивалентная, по
                                    # Мизесу); 0 — без предела. Ореолы соседей складываются линейно, а пластическая
                                    # зона скопления — не сумма зон: в рое из сотен пластинок сумма уходит за десятки %
    halo_smax: float = 0.0          # ореолы — из таблицы при нагрузке не выше этой (МПа); 0 — без предела
    app_center: bool = False        # сдвиг от нагрузки ∝ (sin²ψ − ½): радиальным фора, окружным штраф, в среднем
                                    # выделение не ускоряется (Lacroix 2021: под 200 МПа TSSP не выше, K_N тот же)
    bias_dT: float = 0.0            # °C: фора выпадения в зёрнах с осью c по радиусу (∝ cos²ψ) — плотность
                                    # дислокаций зависит от ориентации зерна (Vizcaíno: 5 °C между семействами зёрен)
    # межзёренный канал: пластинка вдоль грани зерна, след грани не дальше gb_tol от базисного следа одного
    # из двух соседей (соотношение ориентаций с одним соседом: Qin и др. 2011, Son и др. 2026)
    dT_s: float = 0.0               # °C: разброс температуры выпадения по зёрнам (неоднородность мест зарождения:
                                    # дислокации, остаточные напряжения), нормальный, своё значение у каждого зерна
    gb: bool = False
    gb_dT: float = 0.0              # °C: фора зарождения на границе против тела зерна (ниже барьер)
    gb_dT_s: float = 0.0            # °C: разброс форы по граням (энергия границы зависит от разориентации)
    gb_tol: float = 15.0            # град: допуск между следом грани и базисным следом соседа
    gb_L_max: float = 0.0           # наибольшая длина межзёренной пластинки (0 — как L_max)
    # рост во времени: зародыш длиной L_min, кончики идут со скоростью v = k_tip·D/h·(c − TSSD)/C_гидрида
    # (k_tip = 1 — предел по диффузии к кончику радиусом h/2; рост медленнее — k_tip < 1, как в HNGD)
    grow_kin: bool = False
    k_tip: float = 0.05
    halo: bool = False              # пластические ореолы пластинок (halo.py, таблица из fe/halo_runs.py) в упругом
                                    # Фурье вместо обрезки поля соседей; sigma_cap тогда — только предохранитель
    halo_step: float = 0.1          # мкм: ореол растущей пластинки обновляется, когда полудлина изменилась на столько
    halo_tab: str = ""              # путь к таблице ореолов (пусто — data_halo/halo_tab.npz)
    sigma_prof: tuple = ()          # профиль окружного напряжения по толщине: ((доля от наружной поверхности, σ МПа), ...);
                                    # вместо sigma_app + sigma_app_grad (кольца на пальцах, Плясов 2023, рис. 8)
    sigma_T0: float = 0.0           # °C: напряжение задано при этой температуре и при охлаждении спадает как
                                    # (T + 273)/(T0 + 273) — давление газа в запаянной трубе (Плясов 2023); 0 — постоянное
    tssp_ref: bool = False          # измеренная TSSP — начало выпадения в самых выгодных местах (грань в зерне
                                    # окружного семейства): все сдвиги отсчитываются вниз на bias_dT + gb_dT
    cross_tol: float = 0.0          # град: упёршись в границу, кончик продолжается в соседнем зерне (или на
                                    # соседней грани), если след там отличается не больше (Fang 2017, Son 2026); 0 — нет


def run_kinetic(p: KParams, verbose=False, callback=None):
    ca.EPS_N, ca.EPS_T = p.eps_n, p.eps_t
    EPS_N, EPS_T = p.eps_n, p.eps_t
    L = TSS[p.lines] if isinstance(p.lines, str) else p.lines
    rng = np.random.default_rng(p.seed)
    if p.gb:
        grains, gpsi, gbd = make_grains(p, rng, gb=True)
    else:
        grains, gpsi = make_grains(p, rng)
    ny, nx = grains.shape
    el = Elastic((ny, nx), p.dx, p.E, p.nu, free_z=p.free_z)
    if p.halo:
        from halo import HaloTable
        kw_h = dict(path=p.halo_tab) if p.halo_tab else {}
        hs = (lambda s_: min(s_, p.halo_smax) if p.halo_smax > 0 else s_)
        ht = HaloTable(**kw_h); ht.set_load(hs(p.sigma_app))
        ht0 = HaloTable(**kw_h); ht0.set_load(0.0)            # нерастворившиеся пластинки прошлого цикла — без нагрузки
    halos = {}                                          # номер пластинки → (строки, столбцы, компоненты, полудлина)
    psi_map = gpsi[grains]
    e11n, e22n, e12n = eigen_components_for(psi_map)
    e33n = EPS_T
    yfrac = (np.arange(ny) + 0.5) / ny
    if p.sigma_prof:
        yp, sp = np.array(p.sigma_prof, float).T
        sapp = np.interp(yfrac, yp, sp)[:, None] * np.ones((1, nx))
    else:
        sapp = (p.sigma_app + p.sigma_app_grad * (0.5 - yfrac))[:, None] * np.ones((1, nx))
    e_mean = 0.5 * (EPS_N + EPS_T)
    g_app = sapp * (e11n - e_mean) / EPS_N
    if p.app_dg is not None:            # вклад нагрузки по МКЭ (fe/hill_table.json), как в ca_hydride
        g_app = np.interp(sapp, *p.app_dg) * (np.sin(psi_map) ** 2 - 0.5)
    lnc_gorsky = p.v_h_over_rt * sapp / 3.0 if p.v_h_over_rt else 0.0
    # измеренный сдвиг температуры выпадения: нормальное напряжение на пластинке σ_nn = σθ·sin²ψ и фора зёрен
    # с осью c по радиусу (cos²ψ), в градусах; в движущую силу (МПа) — через наклон линии TSSP: n_H·Q/T
    dT_ref = (p.bias_dT + (p.gb_dT if p.gb else 0.0)) if p.tssp_ref else 0.0
    dT_app = p.app_dT * sapp * (np.sin(psi_map) ** 2 - (0.5 if p.app_center else 0.0))   # масштабируется при sigma_T0
    dT_map = dT_app + p.bias_dT * np.cos(psi_map) ** 2 - dT_ref
    dTg = np.zeros(len(gpsi))
    if p.dT_s > 0:                      # свой генератор: зёрна и остальные случайные числа не меняются
        dTg = np.random.default_rng([p.seed, 11]).normal(0.0, p.dT_s, len(gpsi))
        dT_map = dT_map + dTg[grains]
    Q_over_R = np.log(10.0) * 1000.0 * L["TSSP"][1]
    if p.app_dT > 0:
        g_app = np.zeros_like(g_app)
    ngb = 0
    if p.gb:
        # грани, след которых близок (gb_tol) к базисному следу хотя бы одного из двух соседей
        ang = lambda a, b: np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2)
        d1, d2 = ang(gbd["psi"], gpsi[gbd["pair"][:, 0]]), ang(gbd["psi"], gpsi[gbd["pair"][:, 1]])
        keep = np.minimum(d1, d2) <= np.radians(p.gb_tol)
        gcell, gpair, gpsi_f = gbd["cells"][keep], gbd["pair"][keep], gbd["psi"][keep]
        gmatch = np.where(d1[keep] <= d2[keep], gpair[:, 0], gpair[:, 1])     # зерно с соотношением ориентаций
        uniq, gpid = np.unique(gpair, axis=0, return_inverse=True)
        gpid = gpid.ravel()
        pid_map = np.full(ny * nx, -1); pid_map[gcell] = gpid; pid_map = pid_map.reshape(ny, nx)
        ngb = len(gcell)
        dTs = np.random.default_rng([p.seed, 7]).normal(0.0, p.gb_dT_s, len(uniq))[gpid] if p.gb_dT_s > 0 else 0.0
        sg = sapp.ravel()[gcell]
        # разброс по зёрнам — и на грани: от зерна, с которым у гидрида соотношение ориентаций
        dT_gb_app = p.app_dT * sg * (np.sin(gpsi_f) ** 2 - (0.5 if p.app_center else 0.0))
        dT_gb = dT_gb_app + p.bias_dT * np.cos(gpsi_f) ** 2 + p.gb_dT + dTs + dTg[gmatch] - dT_ref
        ge11, ge22, ge12 = eigen_components_for(gpsi_f)
        if p.app_dT > 0:
            g_app_gb = np.zeros(ngb)
        elif p.app_dg is not None:
            g_app_gb = np.interp(sg, *p.app_dg) * (np.sin(gpsi_f) ** 2 - 0.5)
        else:
            g_app_gb = sg * (ge11 - e_mean) / EPS_N
        tried_gb = np.zeros(ngb, bool)
        noroom_gb = np.zeros(ngb, bool)
        L_gb = p.gb_L_max if p.gb_L_max > 0 else p.L_max

        def grow_gb(k, occ_b):
            """Пластинка по следу грани от клетки k, пока рядом (3×3) клетки той же грани и свободно."""
            iy0, ix0 = divmod(int(gcell[k]), nx)
            psi = gpsi_f[k]; pid = gpid[k]
            ty, tx = -np.sin(psi), np.cos(psi)
            step = p.dx * 0.5
            ext = []
            for sgn in (1, -1):
                d = 0.0
                while d < L_gb / 2:
                    d2 = d + step
                    yy = (iy0 + 0.5) * p.dx + sgn * d2 * ty
                    xx = (ix0 + 0.5) * p.dx + sgn * d2 * tx
                    iy, ix = int(np.floor(yy / p.dx)), int(np.floor(xx / p.dx))
                    if occ_b[iy % ny, ix % nx]:
                        break
                    win = pid_map[np.arange(iy - 1, iy + 2)[:, None] % ny, np.arange(ix - 1, ix + 2)[None, :] % nx]
                    if not (win == pid).any():
                        break
                    d = d2
                ext.append(d)
            a, b = ext
            shift = 0.5 * (a - b)
            return np.array([(iy0 + 0.5) * p.dx + shift * ty, (ix0 + 0.5) * p.dx + shift * tx]), 0.5 * (a + b)
    S11 = np.zeros((ny, nx)); S22 = np.zeros_like(S11); S12 = np.zeros_like(S11); S33 = np.zeros_like(S11)
    occ = np.zeros((ny, nx), bool)
    hyd = np.zeros((ny, nx))
    tried = np.zeros((ny, nx), bool)
    noroom = np.zeros((ny, nx), bool)          # пластинка L_min не помещается; занятость только растёт — навсегда
    need_H = p.L_min * C_HYD * p.h_um / p.dx ** 2   # ppm·клетка сверх TSSD рядом — на пластинку L_min
    plates = []
    owner = np.full((ny, nx), -1, np.int32)     # номер пластинки в клетке (рост во времени)
    tips = []                                   # растущие пластинки: концы A (−t), B (+t), активность концов
    ang = lambda a, b: np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2)

    def tvec(psi):
        return np.array([-np.sin(psi), np.cos(psi)])

    def claim(cc, psi, half, pid):
        ys, xs, fr, _ = plate_eigen((ny, nx), p.dx, cc, psi, half, p.h_um)
        sel = np.ix_(ys, xs)
        o = owner[sel]
        o[(fr > 0.2) & (o < 0)] = pid
        owner[sel] = o
        return ys, xs, fr

    def new_tip(pid, A, B, psi, kind, gid, act=(True, True)):
        tips.append(dict(pid=pid, A=np.array(A, float), B=np.array(B, float), psi=float(psi), kind=kind, gid=int(gid),
                         act=list(act), acc=[0.0, 0.0]))

    def blocked(iy, ix, pid):
        w = owner[np.arange(iy - 1, iy + 2)[:, None] % ny, np.arange(ix - 1, ix + 2)[None, :] % nx]
        return bool(((w >= 0) & (w != pid)).any())

    def try_cross(P, u, psi, e):
        """Кончик упёрся в границу (или конец грани): продолжение в соседнем зерне или на соседней грани
        со следом, отличающимся не больше cross_tol; новая пластинка нулевой длины с одним активным концом."""
        Q = P + 1.5 * p.dx * u
        iy, ix = int(np.floor(Q[0] / p.dx)) % ny, int(np.floor(Q[1] / p.dx)) % nx
        best = None
        gq = grains[iy, ix]
        if not (e["kind"] == "intra" and gq == e["gid"]):
            d = ang(gpsi[gq], psi)
            if d <= np.radians(p.cross_tol):
                best = (d, "intra", gq, gpsi[gq])
        if ngb:
            win = pid_map[np.arange(iy - 1, iy + 2)[:, None] % ny, np.arange(ix - 1, ix + 2)[None, :] % nx]
            for f_id in np.unique(win[win >= 0]):
                if e["kind"] == "gb" and f_id == e["gid"]:
                    continue
                k = np.flatnonzero(gpid == f_id)[0]
                d = ang(gpsi_f[k], psi)
                if d <= np.radians(p.cross_tol) and (best is None or d < best[0]):
                    best = (d, "gb", f_id, gpsi_f[k])
        if best is None or blocked(int(np.floor(Q[0] / p.dx)), int(np.floor(Q[1] / p.dx)), e["pid"]):
            return
        _, kind, gid, psi2 = best
        t2 = tvec(psi2)
        fwd = 1 if np.dot(t2, u) >= 0 else 0
        pid = len(plates)                               # начинается за границей, в 1.5 клетки от кончика
        plates.append(dict(c=Q.copy(), psi=float(psi2), half=0.0, T=float(T), t=float(t), init=False, cont=True,
                           grain=int(gid if kind == "intra" else gmatch[np.flatnonzero(gpid == gid)[0]]), kind=kind))
        new_tip(pid, Q, Q, psi2, kind, gid, act=(fwd == 0, fwd == 1))

    def grow_tips(dt, D, c_g):
        """Шаг роста всех активных кончиков; водород — из клеток прироста; общий прирост не больше половины
        пересыщения над TSSD на поле."""
        v0 = p.k_tip * D / p.h_um / C_HYD                # мкм/с на 1 ppm пересыщения
        req = []
        for e in tips:
            for end in (0, 1):
                if not e["act"][end]:
                    continue
                P = e["B"] if end else e["A"]
                iy, ix = int(np.floor(P[0] / p.dx)), int(np.floor(P[1] / p.dx))
                sup = float(c[np.arange(iy - 1, iy + 2)[:, None] % ny, np.arange(ix - 1, ix + 2)[None, :] % nx].mean()) - c_g
                if sup > 0:
                    req.append((e, end, v0 * sup * dt))
        if not req:
            return []
        need = sum(r[2] for r in req) * p.h_um * C_HYD / p.dx ** 2       # ppm·клетка
        have = float(np.clip(c - c_g, 0, None).sum())
        fscale = min(1.0, 0.5 * have / need) if need > 0 else 1.0
        segs = []
        step = 0.5 * p.dx
        for e, end, dl in req:
            e["acc"][end] += dl * fscale
            if e["acc"][end] < step:
                continue
            P = (e["B"] if end else e["A"]).copy()
            sgn = 1.0 if end else -1.0
            tv = tvec(e["psi"]); u = sgn * tv
            length = float(np.linalg.norm(e["B"] - e["A"]))
            Lcap = (L_gb if e["kind"] == "gb" else p.L_max)
            d, stop = 0.0, None
            while d + step <= e["acc"][end]:
                if length + d + step > Lcap:
                    stop = "len"; break
                Q = P + (d + step) * u
                iy, ix = int(np.floor(Q[0] / p.dx)), int(np.floor(Q[1] / p.dx))
                if blocked(iy, ix, e["pid"]):
                    stop = "occ"; break
                if e["kind"] == "gb":
                    win = pid_map[np.arange(iy - 1, iy + 2)[:, None] % ny, np.arange(ix - 1, ix + 2)[None, :] % nx]
                    if not (win == e["gid"]).any():
                        stop = "gb"; break
                elif grains[iy % ny, ix % nx] != e["gid"]:
                    stop = "gb"; break
                d += step
            e["acc"][end] = 0.0 if stop else e["acc"][end] - d
            if d > 0:
                P2 = P + d * u
                segs.append((0.5 * (P + P2), e["psi"], 0.5 * d))
                ys, xs, fr = claim(0.5 * (P + P2), e["psi"], 0.5 * d, e["pid"])
                c[np.ix_(ys, xs)] -= fr * C_HYD
                if end:
                    e["B"] = P2
                else:
                    e["A"] = P2
                q_ = plates[e["pid"]]
                q_["c"] = 0.5 * (e["A"] + e["B"]); q_["half"] = 0.5 * float(np.linalg.norm(e["B"] - e["A"]))
                P = P2
            if stop:
                e["act"][end] = False
                if stop == "gb" and p.cross_tol > 0:
                    try_cross(P, u, e["psi"], e)
        return segs
    ky = 2 * np.pi * np.fft.fftfreq(ny, p.dx)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(nx, p.dx)[None, :]
    k2 = kx ** 2 + ky ** 2

    halo_tabs = {}

    def halo_local(c):
        """Ореол по местному напряжению (профиль по толщине): таблица на ближайшем уровне через 25 МПа;
        при сжатии — как без нагрузки."""
        s_loc = float(sapp[int(c[0] / p.dx) % ny, 0])
        key = int(round(max(s_loc, 0.0) / 25.0) * 25)
        if p.halo_smax > 0:
            key = min(key, int(p.halo_smax))
        if key not in halo_tabs:
            from halo import HaloTable
            halo_tabs[key] = HaloTable(**(dict(path=p.halo_tab) if p.halo_tab else {}))
            halo_tabs[key].set_load(float(key))
        return halo_tabs[key]

    def halo_update(pids, table=None):
        """Ореолы пластинок pids (новых или подросших на halo_step): прежний вычитается, новый — по текущей
        длине; возвращает добавки собственной деформации для add_batch."""
        out = []
        for pid in pids:
            q_ = plates[pid]
            old = halos.get(pid)
            if q_["half"] < 0.25 or (old is not None and abs(q_["half"] - old[3]) < p.halo_step):
                continue
            if old is not None:
                out.append((old[0], old[1], old[2], -1.0))
            tab = table or (halo_local(q_["c"]) if p.sigma_prof else ht)
            ys, xs, comps = tab.patch((ny, nx), p.dx, q_["c"], q_["psi"], q_["half"])
            halos[pid] = (ys, xs, comps, q_["half"])
            out.append((ys, xs, comps, 1.0))
        return out

    Epend = [np.zeros((ny, nx)) for _ in range(4)]     # собственная деформация, ещё не учтённая в S
    if p.halo_cap > 0:                                  # сумма ореолов как есть и учтённая (с пределом)
        Hraw = [np.zeros((ny, nx)) for _ in range(4)]
        Heff = [np.zeros((ny, nx)) for _ in range(4)]
    pend = [False]
    gcache = {}                                         # поле зарождения от S — до следующего решения

    def add_batch(batch, deplete=True, extra=None, solve=True):
        """Пластинки одного шага: водород — из ячеек пластинок, собственная деформация копится;
        extra — добавки собственной деформации (ореолы): (строки, столбцы, компоненты, знак).
        solve=False — без решения упругой задачи: на шаг одно решение (solve_pending), задача линейная."""
        for ys, xs, comps, sgn in (extra or ()):
            sel = np.ix_(ys, xs)
            if p.halo_cap > 0:                          # учтённое = сумма, сжатая до предела (e11, e22, e12, e33)
                for a, v in zip(Hraw, comps):
                    a[sel] += sgn * v
                r = [a[sel] for a in Hraw]
                eq = np.sqrt(2.0 / 3.0 * (r[0] ** 2 + r[1] ** 2 + r[3] ** 2 + 2.0 * r[2] ** 2))
                k = np.minimum(1.0, p.halo_cap / np.maximum(eq, 1e-30))
                for a, h, rr in zip(Epend, Heff, r):
                    new = rr * k
                    a[sel] += new - h[sel]
                    h[sel] = new
            else:
                for a, v in zip(Epend, comps):
                    a[sel] += sgn * v
        sink = np.zeros((ny, nx)) if deplete else None
        for c, psi, half in batch:
            ys, xs, fr, comps = plate_eigen((ny, nx), p.dx, c, psi, half, p.h_um)
            sel = np.ix_(ys, xs)
            new = np.clip(hyd[sel] + fr, 0, 1) - hyd[sel]
            hyd[sel] += new
            if deplete:
                sink[sel] += new
            for a, v in zip(Epend, comps):
                a[sel] += fr * v
        pend[0] = True
        if solve:
            solve_pending()
        return sink * C_HYD if deplete else 0.0

    def solve_pending():
        """Одно решение упругой задачи на всё накопленное; занятость и поле зарождения — заново."""
        if not pend[0]:
            return
        d11, d22, d12, d33 = el.stress(*Epend)
        S11[:] += d11; S22[:] += d22; S12[:] += d12; S33[:] += d33
        for a in Epend:
            a.fill(0.0)
        occ[:] = ndi.binary_dilation(hyd > 0.2, iterations=max(1, int(round(p.occ_um / p.dx))) if p.occ_um > 0 else 1)
        pend[0] = False
        gcache.clear()

    def nuc_field():
        """Вклад напряжений в движущую силу зарождения: тело зерна (g) и грани (g_gb); меняется только
        после решения упругой задачи — считается один раз на решение, а не на каждом шаге."""
        if p.nuc_um > 0:                                # поле, усреднённое по зародышу
            Sn = [ndi.gaussian_filter(a, p.nuc_um / p.dx, mode="wrap") for a in (S11, S22, S12, S33)]
        else:
            Sn = (S11, S22, S12, S33)
        g = (e11n * Sn[0] + e22n * Sn[1] + 2 * e12n * Sn[2] + e33n * Sn[3]) / EPS_N * p.kappa
        if p.cap_local_only:
            mtx = hyd < 0.2
            Sm = [float(a[mtx].mean()) for a in (S11, S22, S12, S33)]
            g_mean = (e11n * Sm[0] + e22n * Sm[1] + 2 * e12n * Sm[2] + e33n * Sm[3]) / EPS_N * p.kappa
            g = np.clip(g - g_mean, -p.sigma_cap, p.sigma_cap) + g_mean + g_app
        else:
            g = np.clip(g, -p.sigma_cap, p.sigma_cap) + g_app
        if p.g_extra is not None:
            g = g + p.g_extra
        g_gb = None
        if ngb:
            # межзёренные зародыши: та же формула, но ε* — для пластинки по следу грани
            Sf = [a.ravel()[gcell] for a in Sn]
            g_gb = (ge11 * Sf[0] + ge22 * Sf[1] + 2 * ge12 * Sf[2] + e33n * Sf[3]) / EPS_N * p.kappa
            if p.cap_local_only:
                gm = (ge11 * Sm[0] + ge22 * Sm[1] + 2 * ge12 * Sm[2] + e33n * Sm[3]) / EPS_N * p.kappa
                g_gb = np.clip(g_gb - gm, -p.sigma_cap, p.sigma_cap) + gm + g_app_gb
            else:
                g_gb = np.clip(g_gb, -p.sigma_cap, p.sigma_cap) + g_app_gb
            if p.g_extra is not None:
                g_gb = g_gb + np.asarray(p.g_extra).ravel()[gcell]
        return g, g_gb

    # начальное состояние: растворилось до TSSD(T_max), остальное — нерастворившиеся пластинки
    c0 = float(min(p.H_ppm, c_line(p.T_max, L["TSSD"])))
    c = np.full((ny, nx), c0)
    if p.init_plates is not None and len(p.init_plates):
        init = [(np.array([q[0], q[1]]), q[2], q[3]) for q in p.init_plates]
        add_batch(init, deplete=False)
        for (cc, psi, half) in init:
            iy, ix = int(cc[0] / p.dx) % ny, int(cc[1] / p.dx) % nx
            plates.append(dict(c=cc, psi=float(psi), half=float(half), grain=int(grains[iy, ix]), T=None, t=None, init=True,
                               kind="init"))
            if p.grow_kin:
                claim(cc, psi, half, len(plates) - 1)
                new_tip(len(plates) - 1, cc - half * tvec(psi), cc + half * tvec(psi), psi, "intra", grains[iy, ix])
        if p.halo:
            add_batch([], deplete=False, extra=halo_update(range(len(plates)), ht0))
    T, t = p.T_max, 0.0
    q = p.rate / 60.0                                   # °C/с
    fac_halo = 1.0
    ln_cell = 2 * np.log(p.dx / p.nu_dx) if p.nu_dx > 0 else 0.0      # тело: места ∝ площади клетки
    ln_gbcell = np.log(p.dx / p.nu_dx) if p.nu_dx > 0 else 0.0        # грань: места ∝ длине грани в клетке
    hist = dict(t=[], T=[], c_mean=[], n=[], frac=[])
    while T > p.T_end:
        Tk = T + 273.15
        # напряжение газа в запаянной трубе спадает с абсолютной температурой
        fac = min(1.0, Tk / (p.sigma_T0 + 273.15)) if p.sigma_T0 > 0 else 1.0
        if p.halo and p.sigma_T0 > 0 and abs(fac - fac_halo) > 0.01:
            ht.set_load(hs(p.sigma_app * fac)); fac_halo = fac
        if not gcache:
            gcache["g"], gcache["g_gb"] = nuc_field()
            gcache["Eg"] = EPS_N * gcache["g"]
        g, g_gb = gcache["g"], gcache["g_gb"]
        # Δ = Δ0 + n_H·R·T·ln(c/TSSP) + ε_n·g + n_H·Q/T·dT — на месте, в том же порядке действий, что и формула
        # (результат побитно тот же, без полудюжины временных полей)
        lnc = np.maximum(c, 1e-6)
        np.log(lnc, out=lnc)
        lnc += lnc_gorsky
        Delta = lnc - np.log(c_line(T, L["TSSP"]))
        Delta *= N_H_MPa_per_K * Tk
        Delta += p.Delta0
        Delta += gcache["Eg"]
        Delta += N_H_MPa_per_K * Q_over_R / Tk * (dT_map if fac == 1.0 else dT_map + (fac - 1.0) * dT_app)
        D = p.D0 * np.exp(-p.QD / (8.314 * Tk)) * 1e12   # мкм²/с
        rad_um = np.sqrt(2 * D * p.t_grow)
        c_eq = c_line(T, L["TSSD"])
        # водорода сверх TSSD в окрестности зародыша (та же, что в проверке ниже) должно хватать на пластинку
        # L_min — иначе место не разыгрывается: отказ был бы заведомым, а темп всплеска — ложным
        Rb = int(np.clip(2 * rad_um, 2.0, 60.0) / p.dx); nb = 2 * Rb + 1
        h_ok = ndi.uniform_filter(c, size=nb, mode="wrap") * (nb * nb) - c_eq * (nb * nb) >= need_H * (1 - 1e-6)
        ok = (Delta > 0) & ~occ & ~tried & ~noroom & h_ok
        # ln скорости по всем клеткам сразу, не-места — −∞ (без выборки по маске)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lnr = np.divide(p.Delta0, Delta)
            np.square(lnr, out=lnr)
            lnr -= 1.0
            lnr *= p.B
            np.subtract(np.log(p.nu_c) + ln_cell, lnr, out=lnr)
        np.copyto(lnr, -np.inf, where=~ok)
        if ngb:
            Delta_gb = p.Delta0 + N_H_MPa_per_K * Tk * (lnc.ravel()[gcell] - np.log(c_line(T, L["TSSP"]))) \
                + EPS_N * g_gb + N_H_MPa_per_K * Q_over_R / Tk * (dT_gb + (fac - 1.0) * dT_gb_app)
            ok_gb = (Delta_gb > 0) & ~occ.ravel()[gcell] & ~tried_gb & ~noroom_gb & h_ok.ravel()[gcell]
            lnr_gb = np.full(ngb, -np.inf)
            lnr_gb[ok_gb] = np.log(p.nu_c) + ln_gbcell - p.B * ((p.Delta0 / Delta_gb[ok_gb]) ** 2 - 1.0)
        else:
            ok_gb = np.zeros(0, bool); lnr_gb = np.zeros(0)
        q = (p.rate2 if (p.rate2 > 0 and T <= p.T_rate2) else p.rate) / 60.0
        dt = p.dT_max / q
        lam_tot = 0.0
        if ok.any() or ok_gb.any():
            m = max(lnr.max() if ok.any() else -np.inf, lnr_gb[ok_gb].max() if ok_gb.any() else -np.inf)
            w = np.subtract(lnr, m)                     # относительные веса
            np.exp(w, out=w)
            if ngb:
                w = np.concatenate([w.ravel(), np.exp(lnr_gb - m)])
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
            n_before = len(plates)
            for k in rng.permutation(idx):
                kgb = k - ny * nx                       # ≥ 0 — межзёренный зародыш номер kgb
                ci = np.unravel_index(k if kgb < 0 else gcell[kgb], (ny, nx))
                if occ_b[ci]:
                    continue
                if kgb < 0:
                    psi = psi_map[ci]
                    cc, half = grow(ci, psi, grains, occ_b, p)
                else:
                    psi = gpsi_f[kgb]
                    cc, half = grow_gb(kgb, occ_b)
                if 2 * half < p.L_min:                  # не помещается (зерно, грань, соседи) — больше не разыгрывать
                    if kgb < 0:
                        noroom[ci] = True
                    else:
                        noroom_gb[kgb] = True
                    continue
                # водорода сверх TSSD в окрестности радиусом ~ диффузионной длины роста хватает не на всё
                R = int(np.clip(2 * rad_um, 2.0, 60.0) / p.dx)
                yy = np.arange(ci[0] - R, ci[0] + R + 1) % ny; xx = np.arange(ci[1] - R, ci[1] + R + 1) % nx
                # чистый избыток над TSSD (с уже обеднёнными клетками — иначе зародыши одного шага берут водород дважды)
                avail = max(0.0, float((c[np.ix_(yy, xx)] - c_eq).sum()))             # ppm·клетка
                half = min(half, avail / C_HYD * p.dx ** 2 / (2 * p.h_um))
                if 2 * half < p.L_min:
                    if kgb < 0:
                        tried[ci] = True
                    else:
                        tried_gb[kgb] = True
                    continue
                if p.grow_kin:                          # зародыш L_min у места зарождения, дальше растёт
                    tv = tvec(psi)
                    h0 = min(0.5 * p.L_min, half)
                    s0 = float(np.dot((np.array(ci) + 0.5) * p.dx - cc, tv))
                    cc = cc + float(np.clip(s0, -half + h0, half - h0)) * tv
                    half = h0
                batch.append((cc, psi, half))
                ys, xs, fr, _ = plate_eigen((ny, nx), p.dx, cc, psi, half, p.h_um)
                occ_b[np.ix_(ys, xs)] |= fr > 0.2       # зародыши шага не перекрываются
                c[np.ix_(ys, xs)] -= fr * C_HYD          # водород пластинки — сразу, дальше разойдётся диффузией
                plates.append(dict(c=cc, psi=float(psi), half=float(half), T=float(T), t=float(t), init=False,
                                   grain=int(grains[ci] if kgb < 0 else gmatch[kgb]), kind="intra" if kgb < 0 else "gb"))
                if p.grow_kin:
                    pid = len(plates) - 1
                    claim(cc, psi, half, pid)
                    new_tip(pid, cc - half * tv, cc + half * tv, psi, "intra" if kgb < 0 else "gb",
                            grains[ci] if kgb < 0 else gpid[kgb])
            if batch:
                add_batch(batch, deplete=False, extra=halo_update(range(n_before, len(plates))) if p.halo else None,
                          solve=False)
                tried[:] = False
                if ngb:
                    tried_gb[:] = False
        if p.grow_kin and tips:                         # рост кончиков за шаг (и зародышей этого шага)
            segs = grow_tips(dt, D, c_eq)
            if segs:
                add_batch(segs, deplete=False, extra=halo_update(sorted({e["pid"] for e in tips})) if p.halo else None,
                          solve=False)
        solve_pending()                                 # кончики не читают S и occ — решение одно на шаг
        # диффузия водорода за шаг; с напряжениями — по химическому потенциалу RT ln c − V_H σ_h
        # (переменные Слотбома: u = c·e^(−φ) диффундирует, c = u·e^φ; масса сохраняется перенормировкой)
        if p.stress_diff:
            sh = np.clip((S11 + S22 + S33) / 3.0, -p.sh_cap, p.sh_cap)
            phi = p.V_H * 1e6 * sh / (8.314 * Tk)
            # в клетках гидрида c — счётная «яма» (водород пластинки), а не раствор: её не взвешиваем
            # внутренним σ_h гидрида, иначе обеднение вокруг пластинки искажается (fe/diff_check.py)
            phi[hyd > 0] = 0.0
            mass = c.sum()
            u = np.fft.irfft2(sfft.rfft2(c * np.exp(-phi)) * np.exp(-D * k2 * dt), s=(ny, nx))
            c = u * np.exp(phi)
            c *= mass / c.sum()
        else:
            c = np.fft.irfft2(sfft.rfft2(c) * np.exp(-D * k2 * dt), s=(ny, nx))      # прямое scipy: быстрее, побитно то же
        t += dt; T -= q * dt
        hist["t"].append(t); hist["T"].append(T); hist["c_mean"].append(float(c.mean()))
        hist["n"].append(len(plates)); hist["frac"].append(float(hyd.mean()))
        if callback is not None:
            callback(plates, lnr, hyd, c, T)
        if verbose and n_new:
            print(f"T {T:6.1f}  c̄ {c.mean():6.1f} (TSSP {c_line(T, L['TSSP']):6.1f})  пластинок {len(plates)}")
    if p.grow_kin:
        plates = [q_ for q_ in plates if q_["init"] or q_["half"] >= 0.5 * p.dx]
    return dict(params=p, plates=plates, hyd=hyd, grains=grains, gpsi=gpsi, S=(S11, S22, S12, S33),
                c=c, hist={k: np.array(v) for k, v in hist.items()})
