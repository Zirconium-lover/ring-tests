#!/usr/bin/env bash
# Колоды этапа 2: разрушение кольца H = 8 мм, полный сегмент 45° × полная высота.
#
#   tools/mkstage2.sh [OUT_DIR]       (по умолчанию runs/stage2/decks)
#
# Постановка — notes/stage2_setup.md. Колоды в git не хранятся.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:-"$HERE/../runs/stage2/decks"}
mkdir -p "$OUT"
EPS_BASE="0 0.85 0.3333 0.80 0.6 0.55 1.0 0.35 1.5 0.20"
COMMON="--H 8 --full-seg --full-height --tet x24 --mu 0.07 --damage --ur 3.3 --dt0 0.0015 --dtmax 0.003 --dtmin 1e-6 --printfreq 5"
# проверка постановки на грубой сетке
python3 "$HERE/mksector.py" -o "$OUT/test_coarse.inp" $COMMON --size 0.2 --uf 0.01 --epsf-eta $EPS_BASE
# основной расчёт
python3 "$HERE/mksector.py" -o "$OUT/H8_mu0.07_full_uf0.01.inp" $COMMON --size 0.1 --uf 0.01 --epsf-eta $EPS_BASE
