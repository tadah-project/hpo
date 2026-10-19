#!/usr/bin/env bash
# run_mpi_smoke.sh -- smoke-test the TADAH_HPO_MPI build of the parallel
# annealing drivers (TADAH:ANNEAL_PT / ANNEAL_PA).
#
# What it checks:
#   * -np 1 and -np 2 runs COMPLETE for both drivers;
#   * the final best loss is IDENTICAL across rank counts (the drivers'
#     determinism contract: global-slot RNG streams, per-pair swap-coin
#     streams, root-side resampling, global budget slicing): the
#     HPO_BEST_LOSS line of best_pot.tadah, the best potential of all
#     ranks, which a run writes at either rank count;
#   * at -np 2 every rank writes its outputs with its suffix (.r0, .r1):
#     best_loss.tadah.r0 and .r1, hpo_run.log.r0 and .r1.
#
# It runs the target files anneal_pt.hpo and anneal_pa.hpo of the Ta test
# set (tests/data/ta/target_files) on its training and validation
# configurations (tests/data/ta/common).
#
# Usage:
#   TADAH_MPI_BIN=/path/to/mpi-enabled/tadah \
#   MPIRUN=/usr/bin/mpirun \
#   bash HPO/scripts/run_mpi_smoke.sh [workdir]
#
# Notes:
#   * The binary must be built with -DTADAH_HPO_MPI=ON against an MPI
#     LAMMPS library.
#   * MPIRUN must match the MPI the binary links (watch out for a conda
#     Intel-MPI mpirun shadowing the system one on PATH).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)          # HPO repo root
BIN=${TADAH_MPI_BIN:?set TADAH_MPI_BIN to the TADAH_HPO_MPI tadah binary}
MPIRUN=${MPIRUN:-mpirun}
WORK=${1:-$(mktemp -d /tmp/tadah_mpi_smoke.XXXXXX)}
TA="$HERE/tests/data/ta"
COMMON="$TA/common"

mkdir -p "$WORK"
ln -sfn "$COMMON" "$WORK/common"

fail=0
run_case() {
  local name=$1 cfg=$2 np=$3
  local d="$WORK/${name}_np${np}"
  mkdir -p "$d"
  cp "$cfg" "$d/config.hpo"
  ( cd "$d" && OMP_NUM_THREADS=1 "$MPIRUN" -np "$np" --oversubscribe \
      "$BIN" hpo --hpotarget config.hpo \
      --validation ../common/config.val -c ../common/config.tadah \
      > run.stdout 2>&1 ) || { echo "FAIL: $name np$np did not complete"; fail=1; return; }
  awk '$2 == "HPO_BEST_LOSS" {print $3; exit}' "$d/best_pot.tadah" \
      2>/dev/null > "$d/BEST" && [ -s "$d/BEST" ] ||
    { echo "FAIL: $name np$np wrote no best_pot.tadah with HPO_BEST_LOSS"; fail=1; }
}

for algo in pt pa; do
  cfg="$TA/target_files/anneal_$algo.hpo"
  run_case "$algo" "$cfg" 1
  run_case "$algo" "$cfg" 2
  b1=$(cat "$WORK/${algo}_np1/BEST" 2>/dev/null || echo NA1)
  b2=$(cat "$WORK/${algo}_np2/BEST" 2>/dev/null || echo NA2)
  if [ "$b1" = "$b2" ]; then
    echo "PASS: $algo best identical across rank counts ($b1)"
  else
    echo "FAIL: $algo np1 best=$b1 != np2 best=$b2"
    fail=1
  fi
  if [ -f "$WORK/${algo}_np2/best_loss.tadah.r0" ] && \
     [ -f "$WORK/${algo}_np2/best_loss.tadah.r1" ] && \
     [ -f "$WORK/${algo}_np2/hpo_run.log.r0" ] && \
     [ -f "$WORK/${algo}_np2/hpo_run.log.r1" ]; then
    echo "PASS: $algo np2 rank-suffixed outputs present"
  else
    echo "FAIL: $algo np2 rank-suffixed outputs missing"
    fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "MPI smoke: ALL PASS (workdir: $WORK)"
else
  echo "MPI smoke: FAILURES (workdir kept: $WORK)"
fi
exit "$fail"
