#!/usr/bin/env bash
# 22_dm_ream_embf0 -- run tadah hpo on a DM_REAM refit seed (no DBFILE, no
# --validation) and assert the EMBF0 ZERO contract on the outputs:
#   - every evaluation's isolated-atom energy (Eiso) is exactly 0.0,
#   - best_pot.tadah declares EMBF0 ZERO.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TADAH=${TADAH:-tadah}
T=${OMP_NUM_THREADS:-1}

if ! command -v "$TADAH" >/dev/null 2>&1 && [[ ! -x "$TADAH" ]]; then
  echo "[run.sh] tadah binary not found at '$TADAH'; set TADAH=/path/to/tadah" >&2
  exit 1
fi

cd "$HERE"
rm -f best_*.tadah loss.tadah outvar.tadah params.tadah stats.tadah \
      hpo_run.log log.lammps hpo_phase_times.csv hpo_script_times.csv \
      hpo_search_space.tsv hpo_optim_resolved.hpo
rm -rf newBestPotsDir

echo "[run.sh] running $(basename "$HERE") (OMP_NUM_THREADS=$T)"
OMP_NUM_THREADS=$T "$TADAH" hpo --hpotarget config.hpo -c ta2_refit_seed.tadah \
  > run.stdout 2>&1 || true

python3 - <<'PY'
import sys
rows = [l.split() for l in open("outvar.tadah") if l.strip()]
hdr = next(r for r in rows if r[0] == "#ITERATION")
i = hdr.index("Eiso")
data = [r for r in rows if not r[0].startswith("#")]
# a rejected evaluation (e.g. the relaxed lattice left the --hotvar range)
# reports FAILSCORE in every output instead of an energy
FAIL = 1e16
evald = [r for r in data if float(r[i]) != FAIL]
bad = [r[i] for r in evald if float(r[i]) != 0.0]
pot = open("best_pot.tadah").read().split("\n")
zero = any(l.split()[:2] == ["EMBF0", "ZERO"] for l in pot if l.strip())
print(f"  EMBF0 check: {len(evald)} evaluated ({len(data) - len(evald)} "
      f"rejected), Eiso != 0 in {len(bad)}; best_pot EMBF0 ZERO: {zero}")
if bad or not evald or not zero:
    sys.exit("  ! EMBF0 ZERO contract violated")
PY

if [[ -f hpo_run.log ]]; then
  best=$(awk '/Best total loss/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9.eE+-]+$/) { print $i; break } }' hpo_run.log | head -1)
  evals=$(awk '/Total evaluations/ { print $4 }' hpo_run.log | head -1)
  status=$(awk -F': *' '/Optimiser status/ { print $2 }' hpo_run.log | head -1)
  printf '  done: best=%s  evals=%s  status=%s\n' "${best:-?}" "${evals:-?}" "${status:-?}"
fi
