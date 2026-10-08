#!/bin/bash
# Четвёртая очередь: после третьей — сетка 0.2 мкм с нормировкой мест зарождения (nu_dx), сравнить с conv_dx (0.4).
S=$1
cd "$(dirname "$0")"
until grep -q QUEUE3_DONE $S/queue_noLit3.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/opt/mvenv/bin/python
H='"gb": [true], "gb_dT": [1.5], "grow_kin": [true], "free_z": [true], "halo": [true], "sigma_cap": [1e9], "cross_tol": [15.0]'
$PY calib_kin.py $S/conv_dx2 "{\"bias_dT\": [17.0], $H, \"size_um\": [[120.0, 120.0]], \"dx\": [0.2], \"sigma_app\": [125, 175], \"seed\": [1, 2, 3, 4, 5, 6]}" 3
$PY calib_kin.py $S/conv_dx2 "{\"bias_dT\": [17.0], $H, \"size_um\": [[120.0, 120.0]], \"dx\": [0.2], \"sigma_app\": [250], \"seed\": [1, 2]}" 3
echo QUEUE4_DONE
