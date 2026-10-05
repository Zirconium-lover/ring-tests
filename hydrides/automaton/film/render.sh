#!/bin/bash
# Рендер всех сцен (1080p, 30 к/с) и склейка. Использование: ./render.sh [качество -ql|-qh] [сцены...]
set -e
cd "$(dirname "$0")"
Q=${1:--qh}; shift || true
SCENES=${@:-S01_Intro S02_Misfit S03_Favour S04_Algorithm S05_Texture S06_Rule45 S07_Check S08_Growth S08b_Series S08c_Zoom S09_Wall S10_Threshold S11_Summary}
MEDIA=${FILM_MEDIA:-media}
mkdir -p "$MEDIA"
MANIM=${MANIM:-/opt/mvenv/bin/manim}
for s in $SCENES; do
  $MANIM $Q --frame_rate 30 --disable_caching --media_dir "$MEDIA" scenes.py $s > "$MEDIA/$s.log" 2>&1 && echo "$s ok" || { echo "$s FAILED"; tail -30 "$MEDIA/$s.log"; }
done
