"""Связность гидридов на поле автомата: метрики, которые в литературе связывают с охрупчиванием.

Оси поля: строки — радиальное направление (ND, толщина стенки), столбцы — окружное (TD), по TD поле
периодично. Верх и низ поля считаются поверхностями стенки: путь трещины идёт сверху вниз.

Метрики:
- RHLD — плотность длины радиальных гидридов, мм/мм² (пластинки со следом под 45–135°; Aliev,
  Kolesnik 2023 и ссылки там);
- HCC — доля толщины, перекрытая проекциями гидридов на радиальное направление в полосе шириной
  band_um вдоль TD (объединение проекций, а не сумма), среднее и максимум по полосам; HCC_rad — то же
  только по радиальным пластинкам (след под 45–135°);
- RHCF — наибольшая радиальная протяжённость связного кластера гидридов, доля толщины (по смыслу —
  как у Billone: самая длинная радиальная цепочка на снимке в долях толщины стенки);
- RHCP — «радиальный непрерывный путь»: трещина идёт сверху вниз по пути наименьшей цены,
  клетка матрицы стоит 1, клетка гидрида — eps; RHCP = 1 − цена/толщина (0 — гидридов на пути нет,
  ~1 — сквозная цепочка; по смыслу как у Simon и др. 2021 и Kim и др. 2022, алгоритм Дейкстры);
- распределение радиальных протяжённостей кластеров (для статистики по многим прогонам).
Связными считаются гидриды ближе gap_um (разрешение снимка).
python connectivity.py папка_с_npz [size_um dx]   — сводка по сохранённым расчётам (пластинки → поле)."""
import os
import sys
import glob
import json
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_hydride import plate_eigen  # noqa: E402


def field_from_plates(plates, shape, dx, h_um=0.6):
    """Поле доли гидрида по таблице пластинок [(cy, cx, ψ, полудлина)], мкм."""
    hyd = np.zeros(shape)
    for cy, cx, psi, half in plates:
        ys, xs, fr, _ = plate_eigen(shape, dx, np.array([cy, cx]), psi, half, h_um)
        sel = np.ix_(ys, xs)
        hyd[sel] = np.clip(hyd[sel] + fr, 0, 1)
    return hyd


def mask_of(hyd, dx, level=0.2, gap_um=1.0):
    m = hyd > level
    r = int(round(gap_um / 2 / dx))
    if r > 0:
        st = ndi.generate_binary_structure(2, 1)
        m_link = ndi.binary_dilation(m, st, iterations=r)
    else:
        m_link = m
    return m, m_link


def rhld(plates, area_um2):
    """мм/мм²: суммарная длина пластинок со следом под 45–135° к TD на единицу площади."""
    if len(plates) == 0:
        return 0.0
    p = np.asarray(plates)
    rad = np.abs(np.sin(p[:, 2])) >= np.sin(np.radians(45))
    return float((2 * p[rad, 3]).sum() / area_um2 * 1e3)


def hcc(mask, dx, band_um=100.0, step_um=10.0):
    """Объединение проекций гидрида на радиальное направление в полосах ширины band_um (периодично по TD)."""
    ny, nx = mask.shape
    w = max(1, int(round(band_um / dx))); st = max(1, int(round(step_um / dx)))
    col_rows = mask  # (ny, nx)
    ext = np.concatenate([col_rows, col_rows[:, :w]], axis=1)
    cs = np.concatenate([np.zeros((ny, 1)), np.cumsum(ext, axis=1)], axis=1)
    vals = []
    for x0 in range(0, nx, st):
        covered = (cs[:, x0 + w] - cs[:, x0]) > 0
        vals.append(covered.mean())
    vals = np.array(vals)
    return float(vals.mean()), float(vals.max())


def cluster_extents(mask_link, dx):
    """Радиальные протяжённости (мкм) связных кластеров; периодичность по TD учитывается удвоением поля."""
    ny, nx = mask_link.shape
    tiled = np.concatenate([mask_link, mask_link], axis=1)
    lab, n = ndi.label(tiled, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(0)
    sl = ndi.find_objects(lab)
    ext = []
    for i, s in enumerate(sl):
        if s is None:
            continue
        if s[1].start >= nx:                       # копия во второй половине — дубль
            continue
        ext.append((s[0].stop - s[0].start) * dx)
    return np.array(ext)


def rhcp(mask, dx, eps=0.05, coarsen_um=0.8):
    """Путь трещины сверху вниз наименьшей цены (Дейкстра); RHCP = 1 − цена / толщина."""
    k = max(1, int(round(coarsen_um / dx)))
    ny, nx = mask.shape
    m = mask[: ny // k * k, : nx // k * k].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3)) > 0.25
    cost = np.where(m, eps, 1.0)
    cost3 = np.concatenate([cost, cost, cost], axis=1)            # периодичность по TD
    H, W = cost3.shape
    mcp = MCP_Geometric(cost3, fully_connected=True)
    starts = [(0, x) for x in range(nx // k, 2 * (nx // k))]
    costs, _ = mcp.find_costs(starts)
    c_min = float(costs[-1, :].min()) + 0.5 * (cost3[0, nx // k:2 * (nx // k)].min())
    return float(max(0.0, 1.0 - c_min / H))


def metrics_of(plates, shape, dx, h_um=0.6, band_um=100.0, gap_um=1.0, hyd=None):
    if hyd is None:
        hyd = field_from_plates(plates, shape, dx, h_um)
    m, ml = mask_of(hyd, dx, gap_um=gap_um)
    H = shape[0] * dx
    ext = cluster_extents(ml, dx)
    hm, hx = hcc(m, dx, band_um=band_um)
    P = np.asarray(plates) if len(plates) else np.zeros((0, 4))
    rad = P[np.abs(np.sin(P[:, 2])) >= np.sin(np.radians(45))] if len(P) else P
    mr, _ = mask_of(field_from_plates(rad, shape, dx, h_um), dx, gap_um=gap_um)
    hrm, hrx = hcc(mr, dx, band_um=band_um)
    return dict(RHLD=rhld(plates, shape[0] * shape[1] * dx * dx), HCC_mean=hm, HCC_max=hx,
                HCC_rad_mean=hrm, HCC_rad_max=hrx,
                RHCF=float(ext.max() / H) if len(ext) else 0.0,
                L_cluster_p90=float(np.percentile(ext, 90)) if len(ext) else 0.0,
                L_cluster_max=float(ext.max()) if len(ext) else 0.0,
                RHCP=rhcp(m, dx), extents=ext.tolist())


if __name__ == "__main__":
    d = sys.argv[1]
    size = float(sys.argv[2]) if len(sys.argv) > 2 else 240.0
    dx = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    n = int(round(size / dx))
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(d + "/*.npz")):
        jf = f[:-4] + ".json"
        if not os.path.exists(jf):
            continue
        meta = json.load(open(jf))
        P = np.load(f)["plates"]
        m = metrics_of(P, (n, n), dx)
        key = tuple((k, meta.get(k)) for k in ("bias_dT", "sigma_cap", "beta", "app") if k in meta)
        for k, v in m.items():
            if k != "extents":
                out[key][(int(meta["sigma_app"]), k)].append(v)
    for key, by in out.items():
        print(dict(key))
        sig = sorted({s for s, _ in by})
        for k in ("RHLD", "HCC_mean", "HCC_rad_mean", "HCC_rad_max", "RHCF", "L_cluster_max", "RHCP"):
            print(f"  {k:14s}" + " ".join(f"{s}:{np.mean(by[(s, k)]):.3g}" for s in sig))
