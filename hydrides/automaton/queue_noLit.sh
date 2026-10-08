#!/bin/bash
# Очередь расчётов без литературы (идемпотентно: calib_kin.py пропускает готовые случаи).
# Запуск: bash queue_noLit.sh папка_scratchpad
S=$1
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/opt/mvenv/bin/python
H='"gb": [true], "gb_dT": [1.5], "grow_kin": [true], "free_z": [true], "halo": [true], "sigma_cap": [1e9], "cross_tol": [15.0]'
# 1. сходимость: поле 120 мкм, сетка 0.4 и 0.2; шаг по температуре 0.25 (поле 240)
$PY calib_kin.py $S/conv_dx "{\"bias_dT\": [17.0], $H, \"size_um\": [[120.0, 120.0]], \"dx\": [0.4, 0.2], \"sigma_app\": [125, 175, 250]}" 3
$PY calib_kin.py $S/conv_dT "{\"bias_dT\": [17.0], $H, \"dT_max\": [0.25], \"sigma_app\": [125, 175, 250]}" 3
# 2. продолжение калибровки с ореолами: фора 19 и 21
$PY calib_kin.py $S/calib_halo2 "{\"bias_dT\": [19.0, 21.0], \"cross_tol\": [15.0], \"gb\": [true], \"gb_dT\": [1.5], \"grow_kin\": [true], \"free_z\": [true], \"halo\": [true], \"sigma_cap\": [1e9]}" 3
# 3. мало водорода: 60 ppm без нагрузки, D×0.2/1/5; серия по водороду при T_max = 400 °C
$PY calib_kin.py $S/lowH_D "{\"bias_dT\": [17.0], $H, \"H_ppm\": [60.0], \"D0\": [1.58e-7, 7.9e-7, 3.95e-6], \"sigma_app\": [0]}" 3
$PY calib_kin.py $S/Hser "{\"bias_dT\": [17.0], $H, \"T_max\": [400.0], \"H_ppm\": [60.0, 100.0, 250.0, 450.0, 600.0], \"sigma_app\": [0, 150, 200], \"seed\": [1]}" 3
echo QUEUE_DONE
