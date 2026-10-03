#!/usr/bin/env bash
# Собрать решатель ccx-arch2 на закреплённом коммите и проверить его на test/fast.
#
#   CCX_ARCH2=/path/to/ccx-arch2 tools/build_solver.sh [--check]
#
# В ccx-arch2 ничего не коммитится: репозиторий переключается в detached HEAD
# на SOLVER_COMMIT, сборка идёт вне дерева (build-mkl/).
set -eu
: "${CCX_ARCH2:?set CCX_ARCH2 to the ccx-arch2 checkout}"
SOLVER_COMMIT=${SOLVER_COMMIT:-c75ad9bbaf78691eabc21dbf0c93165166038e36}
SOLVER_BRANCH=${SOLVER_BRANCH:-claude/eager-davinci-a09b3f}

cd "$CCX_ARCH2"
if [ "$(git rev-parse HEAD)" != "$SOLVER_COMMIT" ]; then
    git fetch origin "$SOLVER_BRANCH"
    git checkout --detach "$SOLVER_COMMIT"
fi
NPROC=${NPROC:-$(nproc)} ./src/build_mkl.sh
EXE="$CCX_ARCH2/build-mkl/ccx_2.23_pardiso"
echo "solver: $EXE  sha256=$(sha256sum "$EXE" | awk '{print $1}')"

if [ "${1:-}" = "--check" ]; then
    # Эталон кейса fast-wrapped из test/regress/cases.json:
    # rc=0, last_inc=533, theta=1.0, 64 удаления.
    RUN=${CHECK_DIR:-$(mktemp -d)}/fast_check
    OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc) CCX_EXE="$EXE" \
        test/fast/run_fast.sh "$RUN"
    last=$(tail -1 "$RUN/m.sta" | awk '{print $2}')
    ndel=$(grep -vc '^#' "$RUN/m.damage")
    echo "fast-wrapped: last_inc=$last deletions=$ndel (expected 533 / 64)"
    [ "$last" = 533 ] && [ "$ndel" = 64 ]
fi
