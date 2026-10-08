#!/bin/bash
# Третья очередь: после второй — серия по водороду с нерастворившимися гидридами (init_auto).
S=$1
cd "$(dirname "$0")"
until grep -q QUEUE2_DONE $S/queue_noLit2.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/opt/mvenv/bin/python
H='"gb": [true], "gb_dT": [1.5], "grow_kin": [true], "free_z": [true], "halo": [true], "sigma_cap": [1e9], "cross_tol": [15.0]'
$PY calib_kin.py $S/Hser2 "{\"bias_dT\": [17.0], $H, \"T_max\": [400.0], \"H_ppm\": [250.0, 450.0, 600.0], \"init_auto\": [true], \"sigma_app\": [0, 150, 200], \"seed\": [1, 2]}" 3
echo QUEUE3_DONE
