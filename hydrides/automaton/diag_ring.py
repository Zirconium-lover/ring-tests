"""Диагностика колец Э635 у предела текучести: почему водород стягивается в несколько толстых полос у растянутой
поверхности (участок 90°). Варианты: базовый; сдвиг от нагрузки с нулевым средним (app_center); без ореолов; оба.
Поле 850 × 120 мкм (толщина целиком, по окружности вдвое уже, чем в ring).

python diag_ring.py папка [участок S1_90|S2_90p|S1_0|S2_0] [процессов]
"""
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e635_plyasov as E  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1]; site = sys.argv[2] if len(sys.argv) > 2 else "S1_90"
    os.makedirs(out, exist_ok=True)
    base = next(c for c in E.cases("ring") if c["ring"] == site and c["bias_dT"] == 5.0)
    base = dict(base, size_um=(850.0, 120.0))
    sets = {"A": [dict(), dict(app_center=True), dict(halo=False), dict(app_center=True, halo=False)],
            "cap": [dict(halo_cap=0.14), dict(halo_cap=0.072)],
            "smax": [dict(halo_smax=150.0), dict(halo_smax=100.0)]}
    var = sets[os.environ.get("DIAG_SET", "A")]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        list(pool.imap_unordered(E.job, [(out, dict(base, **v)) for v in var]))
    print("готово")
