#!/bin/bash
# Кадры рисунков учебника: ./render.sh [сцены...]  (по умолчанию все F*), результат — ../figs/<сцена>.png
set -e
cd "$(dirname "$0")"
export TB_DATA=${TB_DATA:-data}
MEDIA=${TB_MEDIA:-/tmp/tb_media}
MANIM=${MANIM:-/opt/mvenv/bin/manim}
SCENES=${@:-$(grep -o "^class F[0-9A-Za-z_]*" scenes.py | cut -d' ' -f2)}
mkdir -p ../figs "$MEDIA"
for s in $SCENES; do
  if $MANIM -s -qh --disable_caching --media_dir "$MEDIA" scenes.py $s > "$MEDIA/$s.log" 2>&1; then
    f=$(ls -t "$MEDIA"/images/scenes/${s}*.png | head -1)
    /opt/mvenv/bin/python -c "
import sys; from PIL import Image, ImageChops
im = Image.open(sys.argv[1]).convert('RGB'); bg = Image.new('RGB', im.size, (255, 255, 255))
b = ImageChops.difference(im, bg).getbbox(); m = 28
im.crop((max(0, b[0]-m), max(0, b[1]-m), min(im.width, b[2]+m), min(im.height, b[3]+m))).save(sys.argv[2])
" "$f" ../figs/$s.png
    echo "$s ok"
  else
    echo "$s FAILED"; tail -15 "$MEDIA/$s.log"
  fi
done
