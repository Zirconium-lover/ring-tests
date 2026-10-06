"""Сводка calib_kin.py: те же ошибки, что в calib_beta_report.py (Lepine — RHF по изображению, Cinbiz —
доля длины под 45–135° против логистики), группировка по параметрам кинетики. python calib_kin_report.py папка"""
import sys
import json
import glob
from collections import defaultdict
import numpy as np
from calib_beta_report import LEP, cinbiz

KEYS = ("bias_dT", "app_dT", "B", "Delta0", "sigma_cap")


def table(d):
    g = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        g[tuple(m.get(k) for k in KEYS)][int(m["sigma_app"])].append(m)
    rows = []
    for k, by in g.items():
        s = sorted(by)
        img = {q: np.mean([m["RHF_image"] for m in by[q]]) for q in s}
        fn = {q: np.nanmean([m["Fn45_image"] for m in by[q]]) for q in s}
        pl = {q: np.mean([m["RHF45_plates"] for m in by[q]]) for q in s}
        tf = {q: np.nanmean([m["T_first"] for m in by[q]]) for q in s}
        eL = np.sqrt(np.mean([(img[q] - LEP[q]) ** 2 for q in LEP if q in img])) if all(q in img for q in LEP) else np.nan
        eC = np.sqrt(np.nanmean([(fn[q] - cinbiz(q)) ** 2 for q in s]))
        rows.append((np.hypot(eL, eC) / np.sqrt(2), eL, eC, k, s, img, fn, pl, tf))
    rows.sort(key=lambda r: r[0])
    return rows


if __name__ == "__main__":
    for tot, eL, eC, k, s, img, fn, pl, tf in table(sys.argv[1]):
        print(" ".join(f"{a} {b}" for a, b in zip(KEYS, k)) + f" | ошибка {tot:.3f} (Lepine {eL:.3f}, Cinbiz {eC:.3f})")
        print("    изобр:    " + " ".join(f"{q}:{img[q]:.2f}" for q in s))
        print("    Fn45:     " + " ".join(f"{q}:{fn[q]:.2f}" for q in s))
        print("    пластинки:" + " ".join(f"{q}:{pl[q]:.2f}" for q in s))
        print("    T первой: " + " ".join(f"{q}:{tf[q]:.0f}" for q in s))
