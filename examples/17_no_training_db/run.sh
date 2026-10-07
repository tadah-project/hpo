#!/usr/bin/env bash
# 17_no_training_db -- two-step driver:
#   1. `tadah train` on the shared config -> pot.tadah (fitted WEIGHTS)
#   2. rewrite pot.tadah into an HPO seed: pin every coefficient with
#      FIXINDEX 1-K / FIXWEIGHT = WEIGHTS, drop the DBFILE lines
#   3. `tadah hpo` WITHOUT a training DBFILE and WITHOUT --validation --
#      the loss is the LAMMPS cohesive energy declared in config.hpo.
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
      pot.tadah hpo_seed.tadah hpo_run.log log.lammps \
      hpo_phase_times.csv hpo_script_times.csv train.stdout
rm -rf newBestPotsDir

echo "[run.sh] step 1: tadah train (fit the seed potential)"
OMP_NUM_THREADS=$T "$TADAH" train -c "$COMMON_DIR/config.tadah" \
  > train.stdout 2>&1
[[ -f pot.tadah ]] || { echo "  ! train produced no pot.tadah"; exit 1; }

echo "[run.sh] step 2: pin all coefficients (WEIGHTS -> FIXINDEX/FIXWEIGHT)"
python3 - <<'EOF'
import re

with open("pot.tadah") as f:
    lines = f.readlines()

weights = None
out = []
for ln in lines:
    key = ln.split(None, 1)[0] if ln.split() else ""
    if key == "DBFILE":
        continue  # no training data in the HPO seed
    if key in ("FIXINDEX", "FIXWEIGHT"):
        continue  # regenerate below
    if key == "WEIGHTS":
        weights = ln.split()[1:]
    out.append(ln)

assert weights, "pot.tadah carries no WEIGHTS line"
out.append(f"FIXINDEX 1-{len(weights)}\n")
out.append("FIXWEIGHT " + " ".join(weights) + "\n")

with open("hpo_seed.tadah", "w") as f:
    f.writelines(out)
print(f"  hpo_seed.tadah: {len(weights)} coefficients pinned")
EOF

echo "[run.sh] step 3: tadah hpo (no DBFILE, no --validation)"
OMP_NUM_THREADS=$T \
"$TADAH" hpo \
  --hpotarget config.hpo \
  -c hpo_seed.tadah \
  > run.stdout 2>&1 || true

if [[ -f hpo_run.log ]]; then
  best=$(awk '/Best total loss/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9.eE+-]+$/) { print $i; break } }' hpo_run.log | head -1)
  evals=$(awk '/Total evaluations/ { print $4 }' hpo_run.log | head -1)
  status=$(journal_status hpo_run.log)
  printf '  done: best=%s  evals=%s  status=%s\n' "${best:-?}" "${evals:-?}" "${status:-?}"
else
  echo "  done: no hpo_run.log (check run.stdout)"
fi
