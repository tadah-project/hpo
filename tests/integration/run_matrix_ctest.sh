#!/usr/bin/env bash
# Wrapper invoked from the CTest integration entry. Skips gracefully
# when TADAH_HPO_INTEGRATION_BUNDLE is not set.
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
if [[ -z "${TADAH_HPO_INTEGRATION_BUNDLE:-}" ]]; then
  echo "TADAH_HPO_INTEGRATION_BUNDLE not set — skipping (set it to a bundle dir to run the matrix)"
  exit 0
fi
exec "$script_dir/strategy_matrix.sh" "$TADAH_HPO_INTEGRATION_BUNDLE"
