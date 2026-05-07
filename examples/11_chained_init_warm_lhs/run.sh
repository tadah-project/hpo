#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=$(cd "$HERE/../common" && pwd)

if [[ ! -f "$HERE/seed_pot.tadah" ]]; then
  echo "[run.sh] seed_pot.tadah not found — bootstrapping from example 01"
  ( cd "$HERE/../01_baseline_bfgs" && bash "$COMMON/run.sh" >/dev/null )
  cp "$HERE/../01_baseline_bfgs/best_pot.tadah" "$HERE/seed_pot.tadah"
fi
exec bash "$COMMON/run.sh"
