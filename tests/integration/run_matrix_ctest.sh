#!/usr/bin/env bash
# Wrapper invoked from the CTest entry hpo_strategy_matrix. Without
# TADAH_HPO_INTEGRATION_BUNDLE there is nothing to run: it exits with 77,
# the entry's SKIP_RETURN_CODE, so CTest reports it as Skipped, not Passed.
# A 77 from the matrix itself would read as a skip too; it is reported as
# a failure (1) instead.
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
if [[ -z "${TADAH_HPO_INTEGRATION_BUNDLE:-}" ]]; then
  echo "TADAH_HPO_INTEGRATION_BUNDLE not set — skipping (set it to a bundle dir to run the matrix)"
  exit 77
fi
status=0
"$script_dir/strategy_matrix.sh" "$TADAH_HPO_INTEGRATION_BUNDLE" || status=$?
if [[ $status -eq 77 ]]; then
  echo "strategy_matrix.sh exited with 77; reported as a failure, not a skip" >&2
  exit 1
fi
exit "$status"
