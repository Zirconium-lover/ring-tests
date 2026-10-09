"""Рисунок: трубы Э635 под давлением (Плясов и др. 2023, метод 2) — модель против опыта по F_l (рис. 6) и RHC
(рис. 4: RHCP, RHCP по радиальным, RHCF в квадратах 240 мкм; модель «снята» так же — утолщение травлением 3 мкм,
пиксель 3.5 мкм, та же обработка). Поле модели 240 × 240 мкм, σmax при 400 °C, дальше спадает ∝ T.

python fig_tubes.py папка_расчётов [--label ...] [--out файл]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhc_model as RM  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
CAT = {150.0: "#1baf7a", 210.0: "#2a78d6", 300.0: "#eda100", 400.0: "#eb6834"}
EXP_FL = {0: [0.09, 0.115, 0.12], 50: [0.165, 0.17, 0.18, 0.185],
          70: [0.26, 0.275, 0.29, 0.30, 0.33, 0.34, 0.35, 0.36, 0.37], 90: [0.36, 0.365, 0.37, 0.375, 0.38, 0.42],
          110: [0.28, 0.30, 0.305, 0.315, 0.325, 0.345, 0.41, 0.44],
          140: [0.30, 0.305, 0.31, 0.325, 0.33, 0.34, 0.35, 0.365, 0.375, 0.385, 0.395, 0.415, 0.42, 0.43, 0.445]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--label", default="модель")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "fig29_tubes.png"))
    a = ap.parse_args()
    ex = json.load(open(os.path.join(HERE, "data_lit", "plyasov2023_fig4_rhc.json")))["rows"]
    by = defaultdict(lambda: defaultdict(list))
    cache = os.path.join(a.folder, "rhc_spec.json")
    done = json.load(open(cache)) if os.path.exists(cache) else {}
    for f in sorted(glob.glob(os.path.join(a.folder, "H_*.json"))):
        m = json.load(open(f))
        key = os.path.basename(f)
        if key not in done:
            _, hyd, _, dx = RM.load(f)
            r = RM.rhc_spec(hyd, dx, 3.0, 3.5)
            done[key] = r
            json.dump(done, open(cache, "w"))
        r = done[key]
        by[float(m["H_ppm"])][int(m["sigma_app"])].append(dict(F_l=m["F_l_5"], **r))
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    panels = [("F_l", "F_l — доля радиальных (рис. 6)", None),
              ("RHCP", "RHCP (рис. 4)", "RHCP_any_sq240"),
              ("RHCP_rad", "RHCP только по радиальным (рис. 4)", "RHCP_radial_only_sq240"),
              ("RHCF", "RHCF (рис. 4)", "RHCF_any_sq240")]
    for ax, (key, title, ekey) in zip(axs, panels):
        if ekey is None:
            for s, v in EXP_FL.items():
                ax.plot([s] * len(v), v, "^", mfc="none", mec=INK, ms=6, ls="none", label="опыт" if s == 0 else None)
        else:
            xs = [r["sigma_max"] for r in ex]; ys = [r[ekey][0] for r in ex]; es = [r[ekey][1] for r in ex]
            ax.errorbar(xs, ys, es, fmt="^", mfc="none", mec=INK, ecolor=INK, ms=7, capsize=3, label="опыт (150–450 ppm)")
        for h in sorted(by):
            ss = sorted(by[h])
            y = [np.mean([q[key] for q in by[h][s]]) for s in ss]
            ax.plot(ss, y, "-o", color=CAT.get(h, MUTED), lw=2, ms=5, label=f"{a.label}, {h:g} ppm")
        ax.set_title(title, loc="left", fontsize=10); ax.set_ylim(0, 1.05)
        ax.set_xlabel("σmax при 400 °C, МПа", color=MUTED); ax.grid(color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Трубы Э635 под давлением (Плясов и др. 2023, метод 2): модель против опыта", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120, facecolor="white", bbox_inches="tight")
    for h in sorted(by):
        print(h, {s: {k: round(float(np.mean([q[k] for q in by[h][s]])), 2) for k in ("F_l", "RHCP", "RHCP_rad", "RHCF")}
                  for s in sorted(by[h])})
    print("→", a.out)


if __name__ == "__main__":
    main()
