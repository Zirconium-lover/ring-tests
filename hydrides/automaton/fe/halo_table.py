"""Сборка таблицы ореолов для автомата (halo.py) из расчётов halo_runs.py: ε_p в осях пластинки в окне
|s| ≤ a + W_S, |q| ≤ W_Q (окно — см. ниже), сетка 0.1 мкм.
python halo_table.py папка_расчётов [выход.npz]"""
import os
import sys
import json
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halo_runs import A_TAB, AL_TAB, S_TAB, H  # noqa: E402

R2 = np.sqrt(2.0)


def load(d, a, al, s):
    case = f"a{a:g}_al0_0" if s == 0 else f"a{a:g}_al{al}_U{s}"
    z = np.load(os.path.join(d, case + ".npz"))
    ep = z["ep"].astype(np.float64)
    return np.stack([ep[0], ep[1], ep[2], ep[5] / R2])            # tt, nn, zz, tn (тензорный сдвиг)


if __name__ == "__main__":
    D = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                              "data_halo", "halo_tab.npz")
    # окно: за кончиком 5 мкм, по нормали ±5.5 — в нём ≥ 97 % |ε_p| (кроме окружной при 250 МПа: 91–94 %,
    # полосы течения уходят дальше); по порогу 1e-5 окно выходит за ячейку коротких пластинок
    WS, WQ = 5.0, 5.5
    print("окно: за кончиком", WS, "мкм, по нормали ±", WQ, "мкм")
    res = dict(A=np.array(A_TAB, float), AL=np.array(AL_TAB, float), S=np.array(S_TAB, float), h=H, WS=WS, WQ=WQ)
    for i, a in enumerate(A_TAB):
        F = []
        for s in (0,) + S_TAB:
            row = []
            for al in AL_TAB:
                f = load(D, a, 0 if s == 0 else al, s)
                n = f.shape[1]; c = n // 2
                hs = int(round((a + WS) / H)); hq = int(round(WQ / H))
                assert c - hs >= 0 and c - hq >= 0, "окно больше ячейки"
                row.append(f[:, c - hs:c + hs, c - hq:c + hq])           # чётное число точек: s = ±0.05, ±0.15, …
            F.append(np.stack(row))
        res[f"F{i}"] = np.stack(F).astype(np.float32)                 # (1 + nS, nα, 4, ns, nq)
        print("a", a, res[f"F{i}"].shape)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res)
    print("записано", out, round(os.path.getsize(out) / 1e6, 1), "МБ")
