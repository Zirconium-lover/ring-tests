"""Проверка по Desquines и др., JNM 453 (2014) 131: оболочка Zry-4 после снятия напряжений, кольца C-образной
формы. Табл. 7: нижний порог σ_0% (первые радиальные) и верхний σ_100% (все радиальные) в зависимости от
водорода при T_max = 450 °C (5 °C/мин до 300 °C, далее 0.4 °C/мин) и 350 °C (0.4 °C/мин).

Модель — с ореолами, параметры калибровки по Lepine (фора bias_dT, по умолчанию 21 °C); к Desquines ничего
не подгоняется. Нерастворившиеся гидриды строятся автоматически (init_auto).
python desquines_check.py папка [фора] [процессов]
"""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_kin as ck  # noqa: E402

H = '"gb": [true], "gb_dT": [1.5], "grow_kin": [true], "free_z": [true], "halo": [true], "sigma_cap": [1e9], "cross_tol": [15.0]'
SIG = (0, 25, 50, 75, 100, 150, 200)
CASES = [dict(T_max=450.0, rate=5.0, rate2=0.4, T_rate2=300.0, H_ppm=h) for h in (75.0, 192.0, 552.0)] + \
        [dict(T_max=350.0, rate=0.4, H_ppm=h) for h in (63.0, 141.0, 322.0, 540.0)]

if __name__ == "__main__":
    out = sys.argv[1]
    bias = float(sys.argv[2]) if len(sys.argv) > 2 else 21.0
    os.makedirs(out, exist_ok=True)
    base = dict(bias_dT=bias, gb=True, gb_dT=1.5, grow_kin=True, free_z=True, halo=True, sigma_cap=1e9,
                cross_tol=15.0, init_auto=True, T_end=100.0, seed=1)
    jobs = [(out, dict(base, **c, sigma_app=s)) for c in CASES for s in SIG]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as pool:
        list(pool.imap_unordered(ck.job, jobs))
    print("готово", len(jobs))
