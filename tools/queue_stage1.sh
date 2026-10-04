#!/usr/bin/env bash
# Очередь расчётов этапа 1: N параллельных потоков, в каждом расчёты по
# порядку, после каждого — постобработка.
#
#   CCX_ARCH2=... tools/queue_stage1.sh [NWORKERS] [CASE ...]
#
# Без списка — вся серия (кроме уже посчитанных: есть return_code в
# provenance.txt). Запускать через setsid nohup; журнал runs/$STAGE/queue.log.
# STAGE — каталог серии в runs/ (по умолчанию stage1), колоды в runs/$STAGE/decks.
set -u
: "${CCX_ARCH2:?set CCX_ARCH2}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
NW=${1:-2}; shift || true
STAGE=${STAGE:-stage1}
DECKS="$ROOT/runs/$STAGE/decks"
LOG="$ROOT/runs/$STAGE/queue.log"
if [ "$#" -gt 0 ]; then CASES=("$@"); else
    CASES=(H3_mu0.05 H8_mu0.05 H12_mu0.05 H3_mu0 H5_mu0 H8_mu0 H12_mu0
           H3_mu0.2 H5_mu0.2 H8_mu0.2 H12_mu0.2 H5_mu0.05_C3D8I)
fi
THREADS=${THREADS:-$(( $(nproc) / NW ))}; [ "$THREADS" -lt 1 ] && THREADS=1
LOCKDIR="$ROOT/runs/$STAGE/.claims"; mkdir -p "$LOCKDIR"

worker() {
    local w=$1 c
    for c in "${CASES[@]}"; do
        mkdir "$LOCKDIR/$c" 2>/dev/null || continue          # уже взят другим потоком
        if grep -q return_code "$ROOT/runs/$STAGE/$c/provenance.txt" 2>/dev/null; then
            continue
        fi
        echo "$(date +%F' '%T) w$w start $c" >> "$LOG"
        OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
            "$HERE/run_case.sh" "$DECKS/$c.inp" "$ROOT/runs/$STAGE/$c" > /dev/null 2>&1
        local rc=$?
        python3 "$HERE/post_stage1.py" "$ROOT/runs/$STAGE/$c" > "$ROOT/runs/$STAGE/$c/post.log" 2>&1
        echo "$(date +%F' '%T) w$w done  $c rc=$rc $(tail -1 "$ROOT/runs/$STAGE/$c/m.sta")" >> "$LOG"
    done
}
echo "$(date +%F' '%T) queue: ${#CASES[@]} cases, $NW workers x $THREADS threads" >> "$LOG"
for w in $(seq 1 "$NW"); do worker "$w" & done
wait
echo "$(date +%F' '%T) queue finished" >> "$LOG"
