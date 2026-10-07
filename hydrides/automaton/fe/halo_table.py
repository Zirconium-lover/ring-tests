"""Сборка таблицы ореолов для автомата (halo.py) из расчётов halo_runs.py: ε_p в осях пластинки в окне
|s| ≤ a + W_S, |q| ≤ W_Q (окно — по наибольшей протяжённости |ε_p| > 1e-5 по всем случаям), сетка 0.1 мкм.
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
    # окно
    WS = WQ = 0.0
    for a in A_TAB:
        for s in (0,) + S_TAB:
            for al in ((0,) if s == 0 else AL_TAB):
                f = load(D, a, al, s)
                n = f.shape[1]; x = (np.arange(n) + 0.5) * H - n * H / 2
                T, N = np.meshgrid(x, x, indexing="ij")
                m = np.abs(f).max(0) > 1e-5
                WS = max(WS, float(np.abs(T[m]).max()) - a); WQ = max(WQ, float(np.abs(N[m]).max()))
    WS = np.ceil(WS * 10) / 10 + 0.2; WQ = np.ceil(WQ * 10) / 10 + 0.2
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
                row.append(f[:, c - hs:c + hs, c - hq:c + hq])           # чётное число точек: s = ±0.05, ±0.15, …
            F.append(np.stack(row))
        res[f"F{i}"] = np.stack(F).astype(np.float32)                 # (1 + nS, nα, 4, ns, nq)
        print("a", a, res[f"F{i}"].shape)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **res)
    print("записано", out, round(os.path.getsize(out) / 1e6, 1), "МБ")
