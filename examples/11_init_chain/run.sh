#!/usr/bin/env bash
# 11_init_chain/run.sh -- bootstrap seed_pot.tadah on first run, then
# hand off to the shared driver.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON_DIR="$HERE/../common"
TADAH=${TADAH:-tadah}

if [[ ! -f "$HERE/seed_pot.tadah" ]]; then
  echo "[11] no seed_pot.tadah found; bootstrapping via a quick DLIB:BFGS run"
  cat > "$HERE/seed.hpo" <<'HPO'
OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB DLIB
    ALGO BFGS
    MAXEVAL 20
  ENDSTRATEGY
ENDOPTIMIZER

LOSS L2
FAILSCORE 1e16
PC_ERMSE 0 0.0001
PC_LAMMPS --script ../common/lammps/Ecoh.in --hotvar hot_a0 2.5 4.5 3.302 --varloss Ecoh 8.10 800

OPTIM RCUT2B (1) 4.0 6.0

OUTPUT 1
BOUTPUT 1
HPO
  (cd "$HERE" && \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    "$TADAH" hpo --hpotarget seed.hpo \
                 --validation "$COMMON_DIR/config.val" \
                 -c "$COMMON_DIR/config.tadah" > /dev/null 2>&1)
  cp "$HERE/best_pot.tadah" "$HERE/seed_pot.tadah"
  rm -f "$HERE/seed.hpo"
fi

exec bash "$COMMON_DIR/run.sh"
