#!/usr/bin/env bash
# common/run.sh -- shared driver for HPO/examples.
#
# Each example dir does `bash ../common/run.sh` to:
#   - clean prior outputs
#   - invoke `tadah hpo` from the example dir with the shared config.tadah
#     and config.val
#   - print a one-line summary on completion
#
# Defaults are conservative (single-thread, lenient bad-pot filter) so
# the bundle is reproducible on any machine. Override via env vars.

set -euo pipefail

EXAMPLE_DIR=${EXAMPLE_DIR:-$(pwd)}
COMMON_DIR=$(cd "$(dirname "$0")" && pwd)
TADAH=${TADAH:-tadah}  # override with TADAH=/path/to/tadah if not on PATH
T=${OMP_NUM_THREADS:-1}

if ! command -v "$TADAH" >/dev/null 2>&1 && [[ ! -x "$TADAH" ]]; then
  echo "[run.sh] tadah binary not found at '$TADAH'; set TADAH=/path/to/tadah" >&2
  exit 1
fi

cd "$EXAMPLE_DIR"
rm -f best_*.tadah loss.tadah outvar.tadah params.tadah stats.tadah \
      pot.tadah hpo_run.log log.lammps hpo_phase_times.csv \
      hpo_script_times.csv
rm -rf newBestPotsDir

echo "[run.sh] running $(basename "$EXAMPLE_DIR") (OMP_NUM_THREADS=$T)"
# Hot-var ranges and bad-pot caps live in the example's config.hpo as
# PC_LAMMPS --hotvar / PC_<METRIC> --skip-above / --skip-rel flags.
# The legacy TADAH_HPO_* env vars are no longer consulted.
OMP_NUM_THREADS=$T \
"$TADAH" hpo \
  --hpotarget config.hpo \
  --validation "$COMMON_DIR/config.val" \
  -c "$COMMON_DIR/config.tadah" \
  > run.stdout 2>&1 || true

if [[ -f hpo_run.log ]]; then
  best=$(awk '/Best total loss/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9.eE+-]+$/) { print $i; break } }' hpo_run.log | head -1)
  evals=$(awk '/Total evaluations/ { print $4 }' hpo_run.log | head -1)
  status=$(awk -F': *' '/Optimiser status/ { print $2 }' hpo_run.log | head -1)
  printf '  done: best=%s  evals=%s  status=%s\n' "${best:-?}" "${evals:-?}" "${status:-?}"
else
  echo "  done: no hpo_run.log (check run.stdout)"
fi
