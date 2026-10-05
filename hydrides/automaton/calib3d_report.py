"""Сводка калибровки 3D: RHF по изображению и по следам, средние по затравкам; ошибка против Lepine.
python calib3d_report.py папка [ключ=RHF_image]"""
import sys
import json
import glob
from collections import defaultdict
import numpy as np

LEP = {0: (0.073, 0.074), 200: (0.595, 0.552), 250: (0.819, 0.714)}
KEY = sys.argv[2] if len(sys.argv) > 2 else "RHF_image"
PAR = ("size_um", "beta", "sigma_cap", "capture_um", "chi_sL", "R_max")


def load(d):
    rows = [json.load(open(f)) for f in glob.glob(d + "/*.json")]
    g = defaultdict(lambda: defaultdict(list))
    for m in rows:
        p = tuple(m.get(k, "-") for k in PAR)
        g[p][int(m["sigma_app"])].append((m[KEY], m["RHF_trace"], m["n_plates"], m["R_mean"]))
    return g


if __name__ == "__main__":
    g = load(sys.argv[1])
    out = []
    for p, bys in g.items():
        e = [np.mean([v[0] for v in bys[s]]) - np.mean(LEP[s]) for s in LEP if s in bys]
        err = float(np.sqrt(np.mean(np.square(e)))) if len(e) == len(LEP) else np.nan
        out.append((err, p, bys))
    out.sort(key=lambda t: (np.isnan(t[0]), t[0]))
    print(" ".join(f"{k:>9}" for k in PAR), " σ: RHF изобр. (след) [затравок]", "  ошибка")
    for err, p, bys in out:
        cells = []
        for s in sorted(bys):
            v = np.array(bys[s])
            cells.append(f"{s}: {v[:, 0].mean():.2f}±{v[:, 0].std():.2f} ({v[:, 1].mean():.2f}) [{len(v)}]")
        print(" ".join(f"{str(x):>9}" for x in p), " | ".join(cells), f"  {err:.3f}")
