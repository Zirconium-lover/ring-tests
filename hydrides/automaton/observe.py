"""Единый оператор съёмки модели (calib_plan.md, ревью Р2): все меры — с одного «оптического» снимка.

Снимок: поле гидрида → 0.5 мкм/пикс, размытие 1 мкм, порог, утолщение травлением etch_um (rhc_model.optical).
Меры:
  Fn_lab — оператором лаборатории МИФИ (скелет снимка, отрезки между ветвлениями, доля длины отрезков ближе 45°
        к радиусу; ВКР МИФИ 2023, разд. 2.2) — им получены Fn труб, они же точки рис. 6 Плясова; гистограмма по углу;
  F_l — доля длины связных объектов длиннее L_min, главная ось которых ближе 45° к радиусу (прежняя мера),
        по всему полю, по третям толщины и профилем (12 слоёв);
  RHCP, RHCF — по той же маске (connectivity.rhc_mask) и через конвейер снимков с пикселем um_rhc (как рис. 3–4);
  M_* — морфология (morph.py): длина макрогидрида и отрезка скелета, плотность длины, шаг по радиусу;
  площадь, число пластинок, средняя длина, доля межзёренных.
Поверхности стенки (walls) отрезаются. F_l0 — прежняя мера (без травления) для сравнения.
"""
import numpy as np
from scipy import ndimage as ndi

import rhc_model as RM
from connectivity import rhc_mask

ETCH_UM = 2.2          # утолщение травлением, мкм: этап 1 — доля площади структуры до опыта 11.5 % (снимки колец 11–12 %),
                       # полная модель, 16 полей (bench_report.py etch); прежнее 3.0 — по старой модели


def fl_objects(mask, um, L_min=5.0, n_prof=12):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    L, rad, yc = [], [], []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        if len(ys) < 3:
            continue
        P = np.stack([ys * um, xs * um], 1).astype(float); P -= P.mean(0)
        _, v = np.linalg.eigh(P.T @ P)
        ax = v[:, -1]
        pr = P @ ax
        L.append(pr.max() - pr.min() + um)
        rad.append(abs(ax[0]) >= np.cos(np.radians(45)))
        yc.append((ys.mean() + sl[0].start) / mask.shape[0])
    L, rad, yc = np.array(L), np.array(rad, bool), np.array(yc)
    sel = L >= L_min if len(L) else np.zeros(0, bool)

    def frac(b):
        return float((L[b] * rad[b]).sum() / L[b].sum()) if b.any() else np.nan
    out = dict(F_l=frac(sel), n_obj=int(sel.sum()), L_obj_mean=float(L[sel].mean()) if sel.any() else np.nan)
    for i, name in enumerate(("out", "mid", "in")):
        out[f"F_l_{name}"] = frac(sel & (yc >= i / 3) & (yc < (i + 1) / 3))
    out["F_l_prof"] = [frac(sel & (yc >= i / n_prof) & (yc < (i + 1) / n_prof)) for i in range(n_prof)]
    return out


def fn_skeleton(mask, um, um_px=1.0, min_px=2, bins=18):
    """Fn как в программе лаборатории (ВКР МИФИ 2023 по этим трубам, разд. 2.2): бинарный снимок → скелет (Чжан–Суэн) →
    разбиение в точках ветвления → каждый отрезок — прямая (главная ось) → Fn = доля длины отрезков, отклонённых от
    окружного направления больше чем на 45°. Строки маски — радиус, столбцы — окружность (снимок модели уже в
    полярных осях трубы). um_px — пиксель снимка лаборатории (×100, ~1 мкм). → (Fn, гистограмма длины по углу
    к окружному направлению 0–90°, число отрезков)."""
    from skimage.morphology import skeletonize
    k = um_px / um
    m = ndi.zoom(mask.astype(float), 1.0 / k, order=1) > 0.5 if k != 1 else mask.astype(bool)
    sk = skeletonize(m)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    branch = sk & (nb >= 3)
    seg = sk & ~ndi.binary_dilation(branch, structure=np.ones((3, 3)))
    lab, n = ndi.label(seg, structure=np.ones((3, 3)))
    hist = np.zeros(bins); Lr = Lt = 0.0; nseg = 0
    for k_, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == k_)
        if len(ys) < min_px:
            continue
        P = np.stack([ys, xs], 1).astype(float); P -= P.mean(0)
        _, v = np.linalg.eigh(P.T @ P)
        ax = v[:, -1]
        pr = P @ ax
        L = (pr.max() - pr.min() + 1.0) * um_px
        ang = np.degrees(np.arctan2(abs(ax[0]), abs(ax[1])))      # от окружного (столбцы) направления, 0–90°
        hist[min(int(ang / (90.0 / bins)), bins - 1)] += L
        Lt += L; Lr += L * (ang > 45.0); nseg += 1
    return (Lr / Lt if Lt > 0 else np.nan), (hist / max(Lt, 1e-12)).tolist(), nseg


def plate_stats(plates, size_um, gap=2.0, side_um=3.0):
    """Расположение пластинок модели (без съёмки): контакты — пары пластинок, сблизившиеся ближе gap (по точкам
    вдоль пластинок, поле периодическое). side_frac — доля контактов «бок о бок» (смещение центров больше по нормали
    пластинки, чем вдоль неё), остальное — «торец в торец» (продолжение линии); stack_n — сколько соседей в среднем
    лежит сбоку от пластинки (перекрываются по длине, ближе side_um по нормали) — толщина пачки."""
    from scipy.spatial import cKDTree
    P = [(np.array(q["c"], float), float(q["psi"]), float(q["half"])) for q in plates if q["half"] > 0]
    if len(P) < 2:
        return dict(side_frac=np.nan, stack_n=np.nan)
    box = np.array(size_um, float)
    C = np.array([c for c, _, _ in P]) % box
    T = np.array([(-np.sin(psi), np.cos(psi)) for _, psi, _ in P])      # (y, x): строки вниз — как plate_eigen
    N = np.stack([T[:, 1], -T[:, 0]], 1)
    Hh = np.array([h for _, _, h in P])
    pts, own = [], []
    for k, (c, t, h) in enumerate(zip(C, T, Hh)):
        s = np.linspace(-h, h, max(2, int(2 * h) + 1))
        pts.append((c + s[:, None] * t) % box); own.append(np.full(len(s), k))
    pts = np.concatenate(pts); own = np.concatenate(own)
    tree = cKDTree(pts, boxsize=box)
    pairs = {(min(own[a], own[b]), max(own[a], own[b])) for a, b in tree.query_pairs(gap) if own[a] != own[b]}
    side = 0
    for i, j in pairs:
        d = C[j] - C[i]; d -= box * np.round(d / box)
        side += abs(d @ N[i]) > abs(d @ T[i])
    stack = np.zeros(len(P))
    ct = cKDTree(C, boxsize=box)
    for i, nb in enumerate(ct.query_ball_point(C, 2 * Hh.max() + side_um)):
        for j in nb:
            if j == i:
                continue
            d = C[j] - C[i]; d -= box * np.round(d / box)
            stack[i] += abs(d @ T[i]) < Hh[i] and abs(d @ N[i]) < side_um
    return dict(side_frac=float(side / len(pairs)) if pairs else np.nan, stack_n=float(stack.mean()), n_contacts=len(pairs))


def observe(res, etch_um=ETCH_UM, um_rhc=3.5, L_min=5.0, spec=True):
    out = observe_image(res["hyd"], res["params"].dx, int(res.get("walls", 0)), etch_um, um_rhc, L_min, spec)
    new = [q for q in res["plates"] if not q.get("init")]
    L = np.array([2 * q["half"] for q in new])
    gbk = np.array([q.get("kind") == "gb" for q in new], bool)
    radp = np.array([abs(np.sin(q["psi"])) >= np.sin(np.radians(45)) for q in new], bool)
    out.update(n_plates=len(new), L_plate_mean=float(L.mean()) if len(L) else np.nan,
               GB_frac=float(L[gbk].sum() / L.sum()) if len(L) else np.nan,
               F_plates=float(L[radp].sum() / L.sum()) if len(L) else np.nan,
               T_first=float(new[0]["T"]) if new else np.nan,
               p_plast_mean=float(np.mean(res["p_plast"])) if "p_plast" in res else np.nan)
    p = res["params"]
    out.update({"P_" + k: v for k, v in plate_stats(res["plates"], p.size_um).items()})
    return out


def observe_image(hyd, dx, nw=0, etch_um=ETCH_UM, um_rhc=3.5, L_min=5.0, spec=True):
    """Меры по снимку (всё, что зависит от оператора съёмки) — и для пересъёмки сохранённых полей."""
    hyd = np.asarray(hyd, float)
    if nw:
        hyd = hyd[nw: hyd.shape[0] - nw]
    m = RM.optical(hyd, dx, etch_um)
    out = dict(area=float(m.mean()))
    out.update(fl_objects(m, RM.UM, L_min))
    m0 = RM.optical(hyd, dx, 0.0)
    # Fn оператором лаборатории (скелет, отрезки между ветвлениями): без утолщения — основной (структура до опыта
    # 0.10–0.13 при опыте 0.09–0.12), с утолщением 1 мкм и ETCH_UM — для полосы неопределённости оператора
    fn, fh, ns = fn_skeleton(m0, RM.UM)
    out.update(Fn_lab=fn, Fn_hist=fh, n_seg=ns, Fn_lab_e1=fn_skeleton(RM.optical(hyd, dx, 1.0), RM.UM)[0],
               Fn_lab_e=fn_skeleton(m, RM.UM)[0])
    f0 = fl_objects(m0, RM.UM, L_min)
    out.update(F_l0=f0["F_l"], F_l0_out=f0["F_l_out"], F_l0_mid=f0["F_l_mid"], F_l0_in=f0["F_l_in"])
    r = rhc_mask(m, RM.UM, periodic=not nw)
    out.update(RHCP_05=r["RHCP"], RHCF_05=r["RHCF"])
    if spec:
        s = RM.rhc_spec(hyd, dx, etch_um, um_rhc)
        out.update(RHCP=s["RHCP"], RHCF=s["RHCF"], RHCP_rad=s["RHCP_rad"], RHCF_rad=s["RHCF_rad"])
        from morph import morph_model                  # морфология как на снимке 3.5 мкм/пикс (calib_plan.md, п. 8)
        out.update({"M_" + k: v for k, v in morph_model(hyd, dx, etch_um, um_rhc).items()})
    return out
