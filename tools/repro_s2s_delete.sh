#!/usr/bin/env bash
# Воспроизведение: контакт surface-to-surface + удаление элементов -> падение.
#
#   CCX_ARCH2=/path/to/ccx-arch2 tools/repro_s2s_delete.sh [OUT_DIR]
#
# Генерирует две одинаковые колоды (сектор кольца C3D4 + жёсткий сегмент
# C3D8, вязкое повреждение с удалением), отличающиеся только типом контакта,
# и запускает обе с одним окружением. Ожидание на ccx-arch2 c75ad9b:
#   s2s: rc=134 "free(): invalid next size (normal)" (или rc=139) на первой
#        итерации после первой пачки удалений (inc 22, 12 элементов);
#   n2s: rc=0, [FRACTURE COMPLETE], 120 удалённых элементов.
set -u
: "${CCX_ARCH2:?set CCX_ARCH2}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:-"$HERE/../runs/repro_s2s"}
mkdir -p "$OUT/decks"
for t in s2s n2s; do
    python3 "$HERE/mksmoke.py" -o "$OUT/decks/$t.inp" --mode contact --ur 1.5 --ctype $t
    "$HERE/run_case.sh" "$OUT/decks/$t.inp" "$OUT/$t" CCX_FRACTURE_TERMINATION=PHI0:PHI1 >/dev/null 2>&1
    printf "%s: rc=%s last=%s deleted=%s %s\n" $t \
        "$(grep return_code "$OUT/$t/provenance.txt" | cut -d= -f2)" \
        "$(tail -1 "$OUT/$t/m.sta" | awk '{print "inc " $2 " theta " $5}')" \
        "$(grep -vc '^#' "$OUT/$t/m.damage" 2>/dev/null)" \
        "$(grep -m1 -oE 'free\(\): [a-z ()]+|FRACTURE COMPLETE' "$OUT/$t/run.log")"
done
