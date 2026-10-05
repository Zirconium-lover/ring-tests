"""Сводка перекалибровки β: ошибка по Lepine (RHF по изображению) и по Cinbiz (доля радиальных пакетов
Fn — доля длины на снимке под 45–135°, против логистики с серединой 160 МПа: 0.05 при 145, 0.95 при 177). python calib_beta_report.py папка"""
import sys
import json
import glob
from collections import defaultdict
import numpy as np

LEP = {0: 0.0735, 200: 0.5735, 250: 0.7665}
cinbiz = lambda s: 1.0 / (1.0 + np.exp(-np.log(19.0) / 15.0 * (s - 160.0)))
KT_600 = 1.380649e-23 * 600


def load(d):
    g = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(d + "/*.json"):
        m = json.load(open(f))
        g[(m["beta"], m["sigma_cap"], m.get("screen", False))][int(m["sigma_app"])].append(m)
    return g


def table(d):
    g = load(d)
    rows = []
    for k, by in g.items():
        s = sorted(by)
        img = {q: np.mean([m["RHF_image"] for m in by[q]]) for q in s}
        pk = {q: np.nanmean([m["Fn45_image"] for m in by[q]]) for q in s}
        pl = {q: np.mean([m["RHF45_plates"] for m in by[q]]) for q in s}
        eL = np.sqrt(np.mean([(img[q] - LEP[q]) ** 2 for q in LEP if q in img])) if all(q in img for q in LEP) else np.nan
        eC = np.sqrt(np.nanmean([(pk[q] - cinbiz(q)) ** 2 for q in s]))
        rows.append((np.hypot(eL, eC) / np.sqrt(2), eL, eC, k, s, img, pk, pl))
    rows.sort(key=lambda r: r[0])
    return rows


if __name__ == "__main__":
    rows = table(sys.argv[1])
    for tot, eL, eC, k, s, img, pk, pl in rows:
        V = k[0] * 1e-6 * KT_600 / 0.072 * 1e27
        print(f"β {k[0]:<5} потолок {k[1]:<5} экр {int(k[2])} V≈{V:4.0f} нм³ | ошибка {tot:.3f} (Lepine {eL:.3f}, Cinbiz {eC:.3f})")
        print("    изобр:  " + " ".join(f"{q}:{img[q]:.2f}" for q in s))
        print("    Fn45:   " + " ".join(f"{q}:{pk[q]:.2f}" for q in s))
