#!/usr/bin/env bash
# 16_fixweight_mixed -- like common/run.sh but with a LOCAL config.tadah
# (the shared one plus FIXINDEX/FIXWEIGHT pinning).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
COMMON_DIR=$(cd "$HERE/../common" && pwd)
TADAH=${TADAH:-tadah}
T=${OMP_NUM_THREADS:-1}
source "$COMMON_DIR/journal_status.sh"   # journal_status

if ! command -v "$TADAH" >/dev/null 2>&1 && [[ ! -x "$TADAH" ]]; then
  echo "[run.sh] tadah binary not found at '$TADAH'; set TADAH=/path/to/tadah" >&2
  exit 1
fi

cd "$HERE"
rm -f best_*.tadah loss.tadah outvar.tadah params.tadah stats.tadah \
      pot.tadah hpo_run.log log.lammps hpo_phase_times.csv \
      hpo_script_times.csv
rm -rf newBestPotsDir

echo "[run.sh] running $(basename "$HERE") (OMP_NUM_THREADS=$T)"
OMP_NUM_THREADS=$T \
"$TADAH" hpo \
  --hpotarget config.hpo \
  --validation "$COMMON_DIR/config.val" \
  -c config.tadah \
  > run.stdout 2>&1 || true

if [[ -f hpo_run.log ]]; then
  best=$(awk '/Best total loss/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9.eE+-]+$/) { print $i; break } }' hpo_run.log | head -1)
  evals=$(awk '/Total evaluations/ { print $4 }' hpo_run.log | head -1)
  status=$(journal_status hpo_run.log)
  printf '  done: best=%s  evals=%s  status=%s\n' "${best:-?}" "${evals:-?}" "${status:-?}"
else
  echo "  done: no hpo_run.log (check run.stdout)"
fi
