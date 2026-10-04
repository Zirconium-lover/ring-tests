#!/usr/bin/env bash
# Запуск колоды на решателе ccx-arch2.
#
#   CCX_ARCH2=/path/to/ccx-arch2 tools/run_case.sh DECK RUN_DIR [NAME=VALUE ...]
#
# Окружение решателя: значения по умолчанию из test/s3rad/run_s3rad.sh
# (ccx-arch2) плюс проверенный набор CCX_DAMAGE_* (fractrure-tests/runs/run.sh).
# Позиционные NAME=VALUE переопределяют, NAME= снимает переменную.
# BG=1 — запуск через setsid nohup в фоне (для долгих расчётов): скрипт
# сразу возвращается, итог пишется в RUN_DIR/provenance.txt по завершении.
set -u
: "${CCX_ARCH2:?set CCX_ARCH2 to the ccx-arch2 checkout}"
DECK=${1:?deck}; RUN_DIR=${2:?run dir}; shift 2
EXE=${CCX_EXE:-"$CCX_ARCH2/build-mkl/ccx_2.23_pardiso"}
SOLVER_COMMIT=$(git -C "$CCX_ARCH2" rev-parse HEAD 2>/dev/null)
[ -x "$EXE" ] || { echo "solver not found: $EXE" >&2; exit 2; }
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$PWD/$RUN_DIR" ;; esac
NCPU=$(nproc)

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$NCPU}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$NCPU}
export MKL_CBWR=${MKL_CBWR:-COMPATIBLE}
# --- run_s3rad.sh defaults ---
export CCX_DAMAGE_AUTOSPC=${CCX_DAMAGE_AUTOSPC:-1.e-3}
export CCX_DAMAGE_AUTOSPC_FORCE=${CCX_DAMAGE_AUTOSPC_FORCE:-1}
export CCX_DAMAGE_DEADALL=${CCX_DAMAGE_DEADALL:-1.e-2}
export CCX_DAMAGE_DELETE_MAT=${CCX_DAMAGE_DELETE_MAT:-ALL}
export CCX_DAMAGE_LINESEARCH=${CCX_DAMAGE_LINESEARCH:-ADAPTIVE}
export CCX_DAMAGE_REEQ_RESCUE2=${CCX_DAMAGE_REEQ_RESCUE2:-1}
export CCX_DAMAGE_REEQ_SCALE=${CCX_DAMAGE_REEQ_SCALE:-PHYSICAL}
export CCX_DAMAGE_TOPOLOGY=${CCX_DAMAGE_TOPOLOGY:-DEFERRED}
export CCX_DAMAGE_TR_DOGLEG=${CCX_DAMAGE_TR_DOGLEG:-1}
export CCX_DAMAGE_VISCOSITY=${CCX_DAMAGE_VISCOSITY:-1.e-4}
export CCX_PARDISO_REUSE_SYMBOLIC=${CCX_PARDISO_REUSE_SYMBOLIC:-1}
# --- проверенный набор (fractrure-tests/runs/run.sh) ---
export CCX_DAMAGE_VISCOUS_DAMPING=${CCX_DAMAGE_VISCOUS_DAMPING:-2e-4}
export CCX_DAMAGE_FACET_DEBRIS=${CCX_DAMAGE_FACET_DEBRIS:-1}
export CCX_DAMAGE_GRADUAL_DELETE=${CCX_DAMAGE_GRADUAL_DELETE:-8}
export CCX_DAMAGE_DEADSOLE=${CCX_DAMAGE_DEADSOLE:-1e-3}
export CCX_DAMAGE_DEADSOLE_LAW=${CCX_DAMAGE_DEADSOLE_LAW:-1}
export CCX_DAMAGE_BARE_MASK=${CCX_DAMAGE_BARE_MASK:-0.95}
export CCX_DAMAGE_BARE_MASK_LAW=${CCX_DAMAGE_BARE_MASK_LAW:-1}
export CCX_DAMAGE_HINGE=${CCX_DAMAGE_HINGE:-0.95}
export CCX_DAMAGE_HINGE_PENDANT=${CCX_DAMAGE_HINGE_PENDANT:-2}
export CCX_DAMAGE_HINGE_CLUSTER=${CCX_DAMAGE_HINGE_CLUSTER:-1}
export CCX_DAMAGE_HINGE_CLUSTER_BIG=${CCX_DAMAGE_HINGE_CLUSTER_BIG:-1}
export CCX_DAMAGE_TR_STICKY=${CCX_DAMAGE_TR_STICKY:-1}
export CCX_DAMAGE_RESCUE_CUTBACKS=${CCX_DAMAGE_RESCUE_CUTBACKS:-1}
export CCX_DAMAGE_TR_MAXEVAL=${CCX_DAMAGE_TR_MAXEVAL:-100000}
export CCX_DAMAGE_TR_MAXFACT=${CCX_DAMAGE_TR_MAXFACT:-100000}
export CCX_DAMAGE_TR_MAXARM=${CCX_DAMAGE_TR_MAXARM:-1000}

for kv in "$@"; do
    name=${kv%%=*}; value=${kv#*=}
    case "$name" in ''|*[!A-Za-z0-9_]*) echo "bad override: $kv" >&2; exit 2;; esac
    [ "$value" = "$kv" ] && { echo "override must be NAME=VALUE: $kv" >&2; exit 2; }
    if [ -z "$value" ]; then unset "$name"; else export "$kv"; fi
done

# CCX_ARCH2 — путь для наших скриптов; решатель считает её неизвестным
# переключателем и пишет предупреждение в [SWITCHES], поэтому убираем
unset CCX_ARCH2

mkdir -p "$RUN_DIR"
[ "$(realpath "$DECK")" = "$(realpath -m "$RUN_DIR/m.inp")" ] || cp "$DECK" "$RUN_DIR/m.inp"
{
    echo "deck=$DECK"
    echo "deck_sha256=$(sha256sum "$DECK" | awk '{print $1}')"
    echo "executable=$EXE"
    echo "executable_sha256=$(sha256sum "$EXE" | awk '{print $1}')"
    echo "solver_commit=$SOLVER_COMMIT"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "overrides=$*"
    echo "--- environment ---"
    env | grep -E '^(CCX_|OMP_|MKL_)' | sort
} > "$RUN_DIR/provenance.txt"

run() {
    local start=$(date +%s)
    ( cd "$RUN_DIR" && "$EXE" -i m > run.log 2>&1 )
    local rc=$?
    { echo "return_code=$rc"; echo "wall_seconds=$(( $(date +%s) - start ))"; } >> "$RUN_DIR/provenance.txt"
    return $rc
}
if [ "${BG:-0}" = 1 ]; then
    export -f run; export EXE RUN_DIR
    setsid nohup bash -c run > /dev/null 2>&1 < /dev/null &
    echo "started in background: $RUN_DIR (pid $!)"
else
    run; rc=$?
    echo "rc=$rc  $(tail -1 "$RUN_DIR/m.sta" 2>/dev/null)"
    exit $rc
fi
