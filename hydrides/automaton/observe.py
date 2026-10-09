"""Единый оператор съёмки модели (calib_plan.md, ревью Р2): все меры — с одного «оптического» снимка.

Снимок: поле гидрида → 0.5 мкм/пикс, размытие 1 мкм, порог, утолщение травлением etch_um (rhc_model.optical).
Меры:
  F_l — доля длины связных объектов длиннее L_min, главная ось которых ближе 45° к радиусу (Плясов, SIAMS),
        по всему полю, по третям толщины и профилем (12 слоёв);
  RHCP, RHCF — по той же маске (connectivity.rhc_mask) и через конвейер снимков с пикселем um_rhc (как рис. 3–4);
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
    f0 = fl_objects(m0, RM.UM, L_min)
    out.update(F_l0=f0["F_l"], F_l0_out=f0["F_l_out"], F_l0_mid=f0["F_l_mid"], F_l0_in=f0["F_l_in"])
    r = rhc_mask(m, RM.UM, periodic=not nw)
    out.update(RHCP_05=r["RHCP"], RHCF_05=r["RHCF"])
    if spec:
        s = RM.rhc_spec(hyd, dx, etch_um, um_rhc)
        out.update(RHCP=s["RHCP"], RHCF=s["RHCF"], RHCP_rad=s["RHCP_rad"], RHCF_rad=s["RHCF_rad"])
    return out
