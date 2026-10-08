#!/bin/bash
# После Э635 (метод 2): проверка по Desquines 2014 с форой Lepine (21 °C).
S=$1
cd "$(dirname "$0")"
until grep -q E635_Q2_DONE $S/queue_e635b.log 2>/dev/null; do sleep 60; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/opt/mvenv/bin/python desquines_check.py $S/desq 21.0 4
echo DESQ_DONE
