"""Проверка по Исаенковой и др. (Non-ferrous Metals 2023, №1, с. 41–48): трубы Э635 без внешнего
напряжения — доля радиальных гидридов растёт с водородом (RHC ≈ 0.10 при 100 ppm → ≈ 0.19 при 700 ppm,
Fn ≈ 0.12 → 0.23). python trends_e635.py папка [процессов]"""
import os
import sys
import itertools
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool import MATERIALS, Texture, History, Model, simulate, save  # noqa: E402
from thermo import T_line, TSS  # noqa: E402

OUT = sys.argv[1]
MODES = {"seq": Model(), "seq_mean": Model(cap_local_only=True)}


def job(c):
    fn = os.path.join(OUT, c["name"])
    if os.path.exists(fn + ".json"):
        return
    Tmax = float(T_line(c["H"], TSS["E635"]["TSSD"])) + 15
    r = simulate(MATERIALS["E635"], Texture(chi0=38.5), History(H_ppm=c["H"], T_max=Tmax, sigma=0.0),
                 MODES[c["mode"]], seed=c["seed"])
    r["config"]["case"] = c
    save(r, fn)
    print(c["name"], f"RHF {r['metrics']['RHF_image']:.2f}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    L = [dict(name=f"H{H}_{m}_seed{k}", H=H, mode=m, seed=k)
         for H, m, k in itertools.product([700, 500, 300, 100], list(MODES), [1, 2, 3])]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for _ in pool.imap_unordered(job, L):
            pass
