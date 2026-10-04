#!/usr/bin/env bash
# Колоды проверки числа сегментов оправки: 12 сегментов (сектор 15°)
# против 8 (22.5°), постановка и сетка этапа 1 (C3D8I, полсегмента ×
# полвысоты), H = 3 и 8 мм.
#
#   tools/mknseg.sh [OUT_DIR]       (по умолчанию runs/nseg/decks)
#
# 8 сегментов при μ = 0 и 0.05 уже посчитаны в этапе 1 (runs/stage1);
# здесь добавлено μ = 0.07 (выбранное для этапа 2) для обоих чисел сегментов.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:-"$HERE/../runs/nseg/decks"}
mkdir -p "$OUT"
for H in 3 8; do
    for mu in 0 0.05 0.07; do
        python3 "$HERE/mksector.py" -o "$OUT/N12_H${H}_mu${mu}_C3D8I.inp" --H $H --mu $mu --elem C3D8I --sector 15
    done
    python3 "$HERE/mksector.py" -o "$OUT/N8_H${H}_mu0.07_C3D8I.inp" --H $H --mu 0.07 --elem C3D8I
done
