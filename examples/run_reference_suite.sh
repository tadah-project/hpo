#!/usr/bin/env bash
# run_reference_suite.sh -- execute HPO examples and verify each against
# its reference_fingerprint.json. Designed for CI consumption (Session D
# wires this into pipelines).
#
# CONTRACT
#   * Run modes (mutually exclusive):
#       --fast            run only examples flagged "fast": true (default)
#       --full            run every numbered example
#       --example=<NN>    run a single example by leading number (e.g. 03)
#   * --no-clean          keep prior outputs / checkpoints; otherwise the
#                         driver scrubs them before each run
#   * Exit codes:
#       0  every executed example passed its fingerprint
#       1  at least one example failed (numerical regression)
#       2  at least one example skipped (missing fingerprint) -- non-fatal
#          unless mixed with code 1
#       3  one or more runs aborted before producing output (build broken)
#   * Output: one line per example (PASS / FAIL / SKIP / MISSING) plus a
#     final summary table.
#
# ENVIRONMENT
#   TADAH                 path to tadah CLI (defaults to `tadah` on PATH).
#                         Forwarded into common/run.sh.
#   OMP_NUM_THREADS       forwarded; defaults to 1 inside common/run.sh.
#
# TYPICAL USAGE
#   # CI smoke (fast subset, ~5 s on a quiet workstation):
#   bash HPO/examples/run_reference_suite.sh --fast
#
#   # Release validation (every example, ~50 s):
#   bash HPO/examples/run_reference_suite.sh --full
#
#   # Re-check a specific example without re-running:
#   bash HPO/examples/run_reference_suite.sh --example=03 --check-only
#
# This script is the single source of truth for "did the example bundle
# regress?" -- fingerprints live next to each example, the comparator
# helper lives in tools/compare_fingerprint.py, and tolerance choices
# are documented in tools/tolerance_overrides.json.

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON_RUN="$HERE/common/run.sh"
COMPARATOR="$HERE/tools/compare_fingerprint.py"

mode="fast"
target=""
clean_first=1
check_only=0

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --fast)        mode="fast" ;;
    --full)        mode="full" ;;
    --example=*)   mode="single"; target="${arg#--example=}" ;;
    --no-clean)    clean_first=0 ;;
    --check-only)  check_only=1 ;;
    -h|--help)     usage ;;
    *)             echo "unknown arg: $arg" >&2; exit 64 ;;
  esac
done

# Pick examples that match the chosen mode.
declare -a examples=()
for dir in "$HERE"/[0-9]*; do
  [[ -d "$dir" ]] || continue
  name=$(basename "$dir")
  fp="$dir/reference_fingerprint.json"
  case "$mode" in
    full)   examples+=("$dir") ;;
    fast)
      if [[ -f "$fp" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$fp')).get('fast', True) else 1)"; then
        examples+=("$dir")
      fi
      ;;
    single)
      if [[ "$name" == ${target}_* ]]; then examples+=("$dir"); fi
      ;;
  esac
done

if (( ${#examples[@]} == 0 )); then
  echo "no examples matched mode=$mode target=${target:-}" >&2
  exit 64
fi

# Run + compare.
declare -i pass=0 fail=0 skip=0 missing=0
declare -a results
declare -a fail_names

for dir in "${examples[@]}"; do
  name=$(basename "$dir")

  if (( check_only == 0 )); then
    if (( clean_first )); then
      rm -f "$dir/checkpoint.tadah"
    fi
    # Use the example's own run.sh when present (it may bootstrap a
    # seed_pot.tadah or other prerequisites); otherwise fall through to
    # the shared driver.
    if [[ -x "$dir/run.sh" ]]; then
      driver="$dir/run.sh"
    else
      driver="$COMMON_RUN"
    fi
    if ! ( cd "$dir" && bash "$driver" ) >/dev/null 2>&1; then
      echo "  RUN-FAIL  $name (driver returned non-zero)"
      fail+=1
      fail_names+=("$name (run failed)")
      results+=("$name|RUN-FAIL")
      continue
    fi
  fi

  rc=0
  out=$(python3 "$COMPARATOR" "$dir" 2>&1) || rc=$?
  echo "$out"
  case "$rc" in
    0) pass+=1;     results+=("$name|PASS") ;;
    1) fail+=1;     results+=("$name|FAIL"); fail_names+=("$name") ;;
    2) skip+=1;     results+=("$name|SKIP") ;;
    3) missing+=1;  results+=("$name|MISSING") ;;
    *) fail+=1;     results+=("$name|UNKNOWN($rc)"); fail_names+=("$name (rc=$rc)") ;;
  esac
done

echo
echo "==================== summary ===================="
printf '%-32s | %-8s\n' "example" "result"
echo "------------------------------------------------"
for line in "${results[@]}"; do
  IFS='|' read -r n r <<<"$line"
  printf '%-32s | %-8s\n' "$n" "$r"
done

echo
printf 'pass=%d fail=%d skip=%d missing=%d (total=%d, mode=%s)\n' \
  "$pass" "$fail" "$skip" "$missing" "${#examples[@]}" "$mode"

# Exit policy: failures dominate, then missing, then skip is non-fatal.
if (( fail > 0 ));    then exit 1; fi
if (( missing > 0 )); then exit 3; fi
exit 0
