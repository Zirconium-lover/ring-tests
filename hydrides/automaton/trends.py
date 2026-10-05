"""Проверка инструмента на общие закономерности (не на один опыт).
python trends.py папка [процессов]
  H     — порог переориентации от водорода (полное растворение: T_max = TSSD(H) + 15 °C);
          ориентир — Desquines et al., JNM 453 (2014): σ_th = 110 + 65·(1 − e^(−H/65)) МПа (Zry-4)
  Tmax  — неполное растворение: при T_max ниже TSSD часть окружных гидридов остаётся
Режимы модели: const (как в калибровке) и T (β ∝ 1/T, потолок ∝ σ_y(T))."""
import os
import sys
import json
import itertools
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool import MATERIALS, Texture, History, Model, simulate, save  # noqa: E402
from thermo import T_line, TSS  # noqa: E402

OUT = sys.argv[1]
SIG = [0, 50, 100, 150, 200, 250]
MODES = {"const": Model(), "T": Model(beta_mode="1/T", cap_mode="sy")}


def cases():
    L = []
    for H, s, seed, mode in itertools.product([60, 100, 180, 300, 450], SIG, [1, 2], ["const", "T"]):
        Tmax = float(T_line(H, TSS["E635"]["TSSD"])) + 15
        L.append(dict(name=f"H{H}_s{s}_{mode}_seed{seed}", H=H, Tmax=round(Tmax, 1), s=s, seed=seed, mode=mode))
    for Tmax, s, seed in itertools.product([350, 370, 390, 420], [0, 100, 150, 200, 250], [1, 2]):
        L.append(dict(name=f"Tmax{Tmax}_s{s}_seed{seed}", H=180, Tmax=Tmax, s=s, seed=seed, mode="const"))
    return L


def job(c):
    fn = os.path.join(OUT, c["name"])
    if os.path.exists(fn + ".json"):
        return
    r = simulate(MATERIALS["Zry4_SR"], Texture(chi0=30.0), History(H_ppm=c["H"], T_max=c["Tmax"], sigma=c["s"]),
                 MODES[c["mode"]], seed=c["seed"], cache=os.path.join(OUT, "pre"))
    r["config"]["case"] = c
    save(r, fn)
    print(c["name"], f"RHF {r['metrics']['RHF_image']:.2f}", flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "pre"), exist_ok=True)
    L = cases()
    # длинные (много водорода) — первыми
    L.sort(key=lambda c: -c["H"])
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for _ in pool.imap_unordered(job, L):
            pass
    print("готово", len(L))
