#!/bin/bash
# Пятая очередь: сходимость по сетке с полем, усреднённым по зародышу (nuc_um), и запретной зоной в мкм (occ_um).
S=$1
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/opt/mvenv/bin/python
H='"gb": [true], "gb_dT": [1.5], "grow_kin": [true], "free_z": [true], "halo": [true], "sigma_cap": [1e9], "cross_tol": [15.0], "nuc_um": [0.115], "occ_um": [0.4]'
$PY calib_kin.py $S/conv_dx3 "{\"bias_dT\": [17.0], $H, \"size_um\": [[120.0, 120.0]], \"dx\": [0.4, 0.2], \"sigma_app\": [125, 175], \"seed\": [1, 2, 3, 4, 5, 6]}" 3
$PY calib_kin.py $S/conv_dx3 "{\"bias_dT\": [17.0], $H, \"size_um\": [[120.0, 120.0]], \"dx\": [0.4, 0.2], \"sigma_app\": [250], \"seed\": [1, 2]}" 3
echo QUEUE5_DONE
