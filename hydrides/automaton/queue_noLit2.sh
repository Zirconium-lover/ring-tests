#!/bin/bash
# Вторая очередь: дождаться первой (QUEUE_DONE в логе), затем добрать затравки для сходимости по сетке.
S=$1
cd "$(dirname "$0")"
until grep -q QUEUE_DONE $S/queue_noLit.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/opt/mvenv/bin/python
H='"gb": [true], "gb_dT": [1.5], "grow_kin": [true], "free_z": [true], "halo": [true], "sigma_cap": [1e9], "cross_tol": [15.0]'
$PY calib_kin.py $S/conv_dx "{\"bias_dT\": [17.0], $H, \"size_um\": [[120.0, 120.0]], \"dx\": [0.4, 0.2], \"sigma_app\": [125, 175], \"seed\": [3, 4, 5, 6]}" 4
echo QUEUE2_DONE
