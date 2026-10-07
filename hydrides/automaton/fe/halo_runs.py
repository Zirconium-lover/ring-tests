"""«Пластический ореол» одиночной пластинки: поля σ, ε_p и p в матрице Мизеса после выделения пластинки под
нагрузкой — для переноса пластической релаксации в автомат.

Основание: при одинаковых модулях гидрида и матрицы напряжения упругопластического решения в точности равны
упругому решению с собственной деформацией ε* + ε_p (ε_p — найденная пластическая деформация матрицы). Если
ореолы ε_p одиночных пластинок приблизительно складываются, упругий Фурье автомата с ε* + Σ ε_p (вместо
обрезки поля потолком σ_cap) даст пластически релаксированное поле соседей. Проверка складываемости — по
стопкам stack_runs.py (halo_check.py).

Пластинка в центре ячейки, нормаль по y (оси ячейки = оси пластинки: x — t, y — n), длина 2a, толщина 0.6 мкм.
Нагрузка — окружное σθ под углом α к нормали пластинки (α = 0 — радиальная пластинка, 90° — окружная) и
осевое σz: σ_tt = σθ sin²α, σ_nn = σθ cos²α, σ_tn = σθ sin α cos α. Обобщённая плоская деформация.
Ячейка 30 мкм, для коротких пластинок (a ≤ 1.25 мкм) — 15 мкм. Нагрузка — из hill_runs.LOADS или U<σθ>.
python halo_runs.py папка случай[,случай...] [процессов]   (случай = a2.5_al0_U110: полудлина, угол, нагрузка)
python halo_runs.py папка таблица [процессов]   — вся таблица для автомата (A_TAB × AL_TAB × S_TAB)"""
import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hill_fft as hf  # noqa: E402
from hill_runs import LOADS  # noqa: E402

H, TH = 0.1, 0.6
A_TAB = (0.6, 0.9, 1.25, 1.75, 2.5)
AL_TAB = (0, 30, 60, 90)
S_TAB = (75, 125, 175, 250)


def cell_of(a):
    return 15.0 if a <= 1.25 else 30.0


def load_of(lk):
    return LOADS[lk] if lk in LOADS else (float(lk[1:]), 0.0)


def parse(case):
    a, al, lk = case.split("_")
    return float(a[1:]), float(al[2:]), lk


def job(args):
    out, case = args
    fn = os.path.join(out, case + ".json")
    if os.path.exists(fn):
        return case
    a, al, lk = parse(case)
    t0 = time.time()
    hf.WORKERS = 1
    L_CELL = cell_of(a)
    n = int(round(L_CELL / H))
    c = hf.Cell((n, n, 1), H)
    mask = hf.rect_mask((n, n, 1), H, a, TH, 1)
    pl = hf.Plate(c, mask, hf.misfit(1))
    sth, sz = load_of(lk)
    r = np.radians(al)
    Sig = [sth * np.sin(r) ** 2, sth * np.cos(r) ** 2, sz, 0.0, 0.0, np.sqrt(2) * sth * np.sin(r) * np.cos(r)]
    hist = pl.run(Sig, nt=10)
    sh = (6, n, n)
    np.savez_compressed(os.path.join(out, case + ".npz"), sig=pl.sig.reshape(sh).astype(np.float32),
                        ep=pl.ep.reshape(sh).astype(np.float32), p=pl.p.reshape(n, n).astype(np.float32),
                        mask=mask[:, :, 0])
    json.dump(dict(case=case, a=a, alpha=al, load=lk, Sig=Sig, W=hist[-1]["W"], g=hist[-1]["W"] / hf.EPS_N,
                   s_plate=hist[-1]["s_plate"], pl_area=float((pl.p > 1e-4).sum() * H * H), h=H, L=L_CELL,
                   time_s=time.time() - t0), open(fn, "w"))
    print(case, "g", round(hist[-1]["W"] / hf.EPS_N, 1), "время", round(time.time() - t0), flush=True)
    return case


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    cases = sys.argv[2].split(",")
    if sys.argv[2] == "таблица":
        cases = [f"a{a:g}_al0_0" for a in A_TAB] + [f"a{a:g}_al{al}_U{s}" for a in A_TAB for s in S_TAB for al in AL_TAB]
        cases.sort(key=lambda c: -parse(c)[0])                  # длинные (дорогие) — первыми
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 1) as pool:
        list(pool.imap_unordered(job, [(out, c) for c in cases]))
    print("готово", len(cases))
