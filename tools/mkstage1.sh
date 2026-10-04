#!/usr/bin/env bash
# Колоды этапа 1: серия по высоте H и трению μ + контроль C3D8I.
#
#   tools/mkstage1.sh [OUT_DIR]       (по умолчанию runs/stage1/decks)
#
# Колоды в git не хранятся (1–3 МБ каждая): они однозначно задаются
# генератором tools/mksector.py и этим списком.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:-"$HERE/../runs/stage1/decks"}
mkdir -p "$OUT"
for H in 3 5 8 12; do
    for mu in 0 0.05 0.2; do
        python3 "$HERE/mksector.py" -o "$OUT/H${H}_mu${mu}.inp" --H $H --mu $mu
    done
done
# контроль запирания C3D4: та же сетка на C3D8I (вся серия H × μ)
for H in 3 5 8 12; do
    for mu in 0 0.05 0.2; do
        python3 "$HERE/mksector.py" -o "$OUT/H${H}_mu${mu}_C3D8I.inp" --H $H --mu $mu --elem C3D8I
    done
done
# эталон: квадратичные C3D20R, контакт surface-to-surface (удалений нет)
python3 "$HERE/mksector.py" -o "$OUT/H5_mu0.05_C3D20R_s2s.inp" --H 5 --mu 0.05 --elem C3D20R --ctype s2s
# сетки C3D4 для этапа 2 (удаление — только C3D4): разбиение на 24 тетраэдра
# и измельчение вдвое, сравниваются с эталоном C3D20R
python3 "$HERE/mksector.py" -o "$OUT/H5_mu0.05_x24.inp" --H 5 --mu 0.05 --tet x24
python3 "$HERE/mksector.py" -o "$OUT/H5_mu0.05_h0.05.inp" --H 5 --mu 0.05 --size 0.05
