"""Перекалибровка β и потолка ближнего поля (2D) по двум мишеням:
Lepine — RHF по изображению 0.073 / 0.57 / 0.77 при 0 / 200 / 250 МПа;
Cinbiz et al. 2016 — резкий одноосный переход: доля радиальных пакетов (под 45–135°) ~0 при 145 МПа
и ~1 при 177 МПа (тот же лист CWSR Zircaloy-4, ~180 ppm).
python calib_beta.py папка '{"beta": [...], "sigma_cap": [...]}' [процессов]"""
import os
import sys
import json
import itertools
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_hydride import Params, run  # noqa: E402
from ca_analysis import packet_metrics, render, hs  # noqa: E402

SIG = [0, 100, 125, 150, 175, 200, 250]


def image_measures(r):
    """Анализ снимка, как image_rhf: RHF по Simon и доля длины под 45–135° (Fn, как у Cinbiz) на масштабе
    3 мкм, и классический RHF по связным гидридам длиннее 5 мкм."""
    g, um = render(r, um_out=0.65, blur_um=1.0)
    g = np.clip(g + np.random.default_rng(0).normal(0, 0.03, g.shape), 0, 1)
    res, _ = hs.analyse("ca", um=um, g=g, field=True, scales_um=(3.0,), line_um=5.0, n_layers=3)
    sc = res["scales"][0] if res["scales"] else dict(RHF_simon=np.nan, Fn=np.nan)
    return sc["RHF_simon"], sc["Fn"], res.get("RHF_objects", np.nan)


def metrics(r):
    m, C, T = packet_metrics(r)
    m["RHF_image"], m["Fn45_image"], m["RHF_objects"] = image_measures(r)
    m["RHF45_plates"] = float((T["L"] * (T["dev"] > 45)).sum() / T["L"].sum())
    big = [c for c in C if c["ext"] >= 15.0]
    if big:
        ext = np.array([c["ext"] for c in big]); dev = np.array([c["dev"] for c in big])
        m["RHF45_packets"] = float((ext * (dev > 45)).sum() / ext.sum())
    else:
        m["RHF45_packets"] = np.nan
    return m


def job(a):
    out, kw = a
    name = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    fn = os.path.join(out, name + ".json")
    if os.path.exists(fn):
        return
    t0 = time.time()
    r = run(Params(capture_um=35.0, cap_local_only=True, **kw))
    m = metrics(r)
    m.update(kw, time_s=time.time() - t0)
    np.savez_compressed(os.path.join(out, name + ".npz"),
                        plates=np.array([[q["c"][0], q["c"][1], q["psi"], q["half"]] for q in r["plates"]]))
    json.dump(m, open(fn, "w"), default=float)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    grid = json.loads(sys.argv[2])
    grid.setdefault("sigma_app", SIG); grid.setdefault("seed", [1, 2])
    keys = list(grid)
    jobs = [(out, dict(zip(keys, v))) for v in itertools.product(*[grid[k] for k in keys])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, jobs)):
            if (i + 1) % 20 == 0:
                print(i + 1, "из", len(jobs), flush=True)
    print("готово", len(jobs))
