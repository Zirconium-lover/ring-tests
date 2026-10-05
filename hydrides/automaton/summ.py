"""Сводка перебора: python summ.py папка [папка ...]"""
import json, glob, sys
import numpy as np
from collections import defaultdict
LEP = {0: 0.0735, 200: 0.5735, 250: 0.7665}
rows = [json.load(open(f)) for d in sys.argv[1:] for f in glob.glob(d + "/*.json")]
d = defaultdict(list)
for r in rows:
    d[(r.get("chi0"), r.get("chi_s"), r["beta"], r["sigma_cap"], r["sigma_app"])].append(r)


def f(v, k):
    x = np.array([np.nan if r.get(k) is None else r[k] for r in v], float)
    return np.nanmean(x) if np.isfinite(x).any() else np.nan


S = (0, 100, 150, 200, 250, 300)
for key in sorted({k[:4] for k in d}, key=str):
    get = lambda s, k: f(d.get(key + (s,), []), k)
    img = [get(s, "RHF_image") for s in S]; pk = [get(s, "packet_RHF") for s in S]
    e = np.sqrt(np.nanmean([(img[S.index(s)] - v) ** 2 for s, v in LEP.items()]))
    ep = np.sqrt(np.nanmean([(pk[S.index(s)] - v) ** 2 for s, v in LEP.items()]))
    print(f"χ0={key[0]} s={key[1]} β={key[2]} cap={key[3]} | img " + " ".join(f"{v:.2f}" for v in img)
          + " | pk " + " ".join(f"{v:.2f}" for v in pk)
          + f" | рад {np.nanmean([get(200, 'dev_in_radial'), get(250, 'dev_in_radial')]):.0f} окр {get(0, 'dev_in_circ'):.0f}"
          + f" | пакеты(0) {get(0, 'frac_in_packets'):.2f} | ош {e:.3f}/{ep:.3f}")
