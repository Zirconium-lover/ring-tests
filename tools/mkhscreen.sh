#!/usr/bin/env bash
# Колоды проверки минимальной высоты: где очаг (прижатая зона у кромки,
# середина сегмента, над зазором, у торца), η и ω на пути нагружения.
# Полный сегмент × полная высота (как этап 2), но C3D8I без повреждения;
# 8 сегментов, H = 5, 6, 8 мм, μ = 0.05 и 0.07, ход до u_r = 2.5 мм.
#
#   tools/mkhscreen.sh [OUT_DIR]       (по умолчанию runs/hscreen/decks)
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:-"$HERE/../runs/hscreen/decks"}
mkdir -p "$OUT"
for H in 5 6 8; do
    for mu in 0.05 0.07; do
        python3 "$HERE/mksector.py" -o "$OUT/F_H${H}_mu${mu}_C3D8I.inp" --H $H --mu $mu --elem C3D8I \
            --full-seg --full-height --ur 2.5 --dt0 0.004 --dtmax 0.004
    done
done
