"""Инструмент: гидридная структура в сечении r–θ по карточке материала и истории охлаждения.

    from tool import Material, Texture, History, Model, simulate, MATERIALS
    r = simulate(MATERIALS["E635"], Texture(chi0=38.5), History(H_ppm=200, T_max=430, sigma=150))
    r["metrics"]["RHF_image"], r["thermo"]["T_start"], [q["T"] for q in r["plates"]]

Что зависит от температуры выпадения (её даёт thermo.Cooling по линиям Плясова):
  • Model.beta_mode = "1/T"  — избирательность β ∝ 1/T (от калибровочной T_ref);
  • Model.cap_mode = "sy"    — эффективный потолок ∝ σ_y(T) материала (от калибровочного σ_y,ref);
  • нерастворившиеся при T_max гидриды остаются с прошлого состояния (окружные, без напряжения).
По умолчанию β и потолок постоянны — как в калибровке по Lepine (Zry-4, 178 ppm, 415 °C).
"""
import os
import sys
import json
import warnings
from dataclasses import dataclass, field, asdict
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_hydride import Params, run  # noqa: E402
from ca_analysis import packet_metrics, image_rhf, interdistance  # noqa: E402
from thermo import Cooling, area_fraction  # noqa: E402


@dataclass
class Material:
    name: str
    tss: str = "E635"                       # линии растворимости (thermo.TSS)
    sy_T: tuple = ()                        # предел текучести: ((T °C, σ_y МПа), ...)
    E: float = 90e3
    nu: float = 0.34

    def sy(self, T):
        if not self.sy_T:
            return np.nan
        t, s = np.array(self.sy_T, float).T
        return float(np.interp(T, t, s))


MATERIALS = {
    # σ_y: Э635 рекр. — 448 МПа при 20 °C, 226 при 380 °C (notes/literature_e635.md);
    # Zry-4 отожж. для снятия напряжений — ≈ 350 МПа при 350 °C (оценка по литературе), 20 °C — ≈ 520
    "E635": Material("Э635 (рекристаллизованный)", "E635", ((20, 448), (380, 226))),
    "Zry4_SR": Material("Zry-4 (снятие напряжений)", "E635", ((20, 520), (350, 350))),
}


@dataclass
class Texture:
    chi0: float = 30.0                      # наклон базисных полюсов от радиуса, °
    chi_s: float = 26.0                     # разброс, °
    chi0_profile: tuple = ()                # по толщине: (наружный, средний, внутренний)
    grain_um: tuple = (2.5, 4.5)            # размер зерна по ND и TD, мкм


@dataclass
class History:
    H_ppm: float = 180.0
    T_max: float = 415.0                    # °C
    sigma: float = 0.0                      # окружное напряжение при охлаждении, МПа
    sigma_grad: float = 0.0                 # изменение по толщине (изгиб), МПа
    T_end: float = 20.0


@dataclass
class Model:
    beta: float = 0.12                      # 1/МПа при T_ref
    sigma_cap: float = 90.0                 # МПа при T_ref (эффективный, см. fe/README.md)
    capture_um: float = 35.0
    beta_mode: str = "const"                # "const" | "1/T"
    cap_mode: str = "const"                 # "const" | "sy"
    T_ref: float = 300.0                    # °C — средняя температура выпадения в калибровке
    cal_material: str = "Zry4_SR"           # материал калибровки (для cap_mode = "sy")


def _plates_arr(plates):
    return np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in plates]) if plates else np.zeros((0, 4))


def simulate(mat: Material, tex: Texture, hist: History, model: Model = Model(), size_um=(240.0, 240.0),
             seed=1, dx=0.4, metrics=True, cache=None):
    th = Cooling(hist.H_ppm, hist.T_max, hist.T_end, mat.tss)
    base = dict(size_um=tuple(size_um), dx=dx, grain_um=tuple(tex.grain_um), chi0=tex.chi0, chi_s=tex.chi_s,
                chi0_profile=tuple(tex.chi0_profile), beta=model.beta, sigma_cap=model.sigma_cap,
                capture_um=model.capture_um, E=mat.E, nu=mat.nu, seed=seed)
    # 1. нерастворившиеся гидриды: исходная (окружная, без напряжения) структура при полном водороде,
    #    из неё остаются первые выпавшие пластинки — общей площадью frac_left
    init = None
    if th.frac_left > 1e-5:
        key = None
        if cache:
            key = os.path.join(cache, f"pre_{mat.tss}_chi{tex.chi0}_{'-'.join(map(str, tex.chi0_profile))}_H{hist.H_ppm:g}_s{seed}.npz")
        if key and os.path.exists(key):
            P0 = np.load(key)["plates"]
        else:
            r0 = run(Params(**base, frac=float(area_fraction(hist.H_ppm)), sigma_app=0.0))
            P0 = _plates_arr(r0["plates"])
            if key:
                np.savez_compressed(key, plates=P0)
        area = np.cumsum(2 * P0[:, 3] * 0.6) / (size_um[0] * size_um[1])
        init = P0[: max(1, int(np.searchsorted(area, th.frac_left)))]
    # 2. охлаждение под напряжением: температура каждой пластинки — по ходу выпадения
    cal = MATERIALS[model.cal_material]
    Tk_ref = model.T_ref + 273.15
    sy_ref = cal.sy(model.T_ref)

    def schedule(progress):
        T = th.T(progress)
        st = dict(T=T)
        if model.beta_mode == "1/T":
            st["beta"] = model.beta * Tk_ref / (T + 273.15)
        if model.cap_mode == "sy":
            st["sigma_cap"] = model.sigma_cap * mat.sy(T) / sy_ref
        return st

    p = Params(**base, frac=th.frac_new, sigma_app=hist.sigma, sigma_app_grad=hist.sigma_grad,
               init_plates=init, schedule=schedule)
    r = run(p)
    out = dict(thermo=th.summary(), plates=r["plates"], hyd=r["hyd"], grains=r["grains"],
               config=dict(material=asdict(mat), texture=asdict(tex), history=asdict(hist), model=asdict(model),
                           size_um=list(size_um), seed=seed, dx=dx))
    if metrics:
        m, C, Tb = packet_metrics(r)
        m["RHF_image"] = image_rhf(r)
        new = [q for q in r["plates"] if not q.get("init")]
        if new:
            Tn = np.array([q["T"] for q in new])
            dev = np.degrees(np.abs(np.arctan(np.tan([q["psi"] for q in new]))))
            L = np.array([2 * q["half"] for q in new])
            m["T_first"], m["T_median"] = float(Tn[0]), float(np.median(Tn))
            # доля радиальных среди выпавших в первой и последней трети
            k = len(new) // 3
            m["radial_first_third"] = float((L[:k] * (dev[:k] >= 65)).sum() / L[:k].sum()) if k else np.nan
            m["radial_last_third"] = float((L[-k:] * (dev[-k:] >= 65)).sum() / L[-k:].sum()) if k else np.nan
        m["n_init"] = 0 if init is None else len(init)
        out["metrics"] = m
    return out


def save(r, path):
    """Лёгкое сохранение: метрики и конфигурация — json, пластинки — npz."""
    js = dict(metrics=r.get("metrics"), thermo=r["thermo"], config=r["config"])
    json.dump(js, open(path + ".json", "w"), ensure_ascii=False, indent=1, default=float)
    np.savez_compressed(path + ".npz", plates=_plates_arr(r["plates"]),
                        T=np.array([np.nan if q["T"] is None else q["T"] for q in r["plates"]]),
                        init=np.array([bool(q.get("init")) for q in r["plates"]]))


if __name__ == "__main__":
    import time
    t = time.time()
    r = simulate(MATERIALS["Zry4_SR"], Texture(chi0=30), History(H_ppm=178, T_max=415, sigma=200), size_um=(120, 120))
    print(r["thermo"])
    print({k: r["metrics"][k] for k in ("RHF_image", "T_first", "T_median", "radial_first_third", "radial_last_third", "n_init")},
          f"{time.time() - t:.0f} с")
