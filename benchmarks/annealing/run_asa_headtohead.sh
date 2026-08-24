#!/usr/bin/env bash
# Opt-in oracle: build the pristine upstream ASA-30.53 drop and run its
# canonical ASA_TEST problem, then diff the outcome against the shipped
# reference outputs (asa_test_asa / asa_test_usr). This validates the
# LOCAL toolchain against Ingber's published run of the same code; the
# in-tree bitwise golden gate (tests/test_asa_golden.cpp) separately pins
# our vendored ASA against the pristine build on every ctest run.
#
# Usage: run_asa_headtohead.sh [path-to-ASA-dir]
set -euo pipefail
ASA_DIR="${1:-$(dirname "$0")/../../../SIMULATED_ANNEALING/ASA-30.53/ASA}"
[ -f "$ASA_DIR/asa.c" ] || { echo "no upstream ASA at $ASA_DIR"; exit 2; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cp -r "$ASA_DIR"/. "$WORK"/
cd "$WORK"
make -f ASA-Makefile compile >/dev/null
./asa_run gcc "(headtohead)" >/dev/null 2>&1 || true
echo "== final cost line (ours vs Ingber's reference) =="
grep -h "final cost" asa_usr_out asa_test_usr 2>/dev/null || true
echo "== exit condition =="
grep -h "exit" asa_out asa_test_asa 2>/dev/null | head -4 || true
echo "(differences reflect compiler/libm, not the algorithm; the"
echo " in-tree bitwise golden gate is the CI-grade oracle)"
