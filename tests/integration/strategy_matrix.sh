#!/usr/bin/env bash
#
# strategy_matrix.sh — run a small grid of init-strategy/run-policy
# combinations against any HPO target bundle and report best-loss /
# wall-time per combination.
#
# Generic: takes a bundle directory as its only required argument; does
# not bake in any system-specific paths or scripts. Each combination is
# run in its own scratch dir so outputs never collide.
#
# Usage:
#   strategy_matrix.sh <bundle_dir> [-o <out_dir>] [-b <tadah_binary>] [-m <manifest>]
#
# bundle_dir — must contain Ta_config.tadah-style files (any name), a
#   config.hpo, and a config.val. We invoke `tadah hpo -c <cfg> --hpotarget
#   config.hpo --validation config.val` from within the scratch dir.
#
# Manifest — newline-separated list of "name | <hpo body>" entries. A
#   default manifest is shipped alongside this script. Each entry's
#   <hpo body> is appended to the *original* config.hpo's loss/optim
#   sections, with the OPTIMIZER block replaced by the per-entry block.
#
set -euo pipefail

usage() {
  sed -n '2,/^set -e/p' "$0" | sed 's/^# *//'
  exit 1
}

bundle=""
out_dir=""
tadah_bin=""
manifest=""
while (( "$#" )); do
  case "$1" in
    -o) out_dir="$2"; shift 2 ;;
    -b) tadah_bin="$2"; shift 2 ;;
    -m) manifest="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) bundle="$1"; shift ;;
  esac
done

[[ -n "$bundle" ]] || usage
[[ -d "$bundle" ]] || { echo "bundle dir not found: $bundle" >&2; exit 2; }
bundle=$(cd "$bundle" && pwd)
[[ -f "$bundle/config.hpo" ]] || { echo "missing $bundle/config.hpo" >&2; exit 2; }
[[ -f "$bundle/config.val" ]] || { echo "missing $bundle/config.val" >&2; exit 2; }

cfg_main=$(ls "$bundle"/*.tadah | grep -v 'best_\|pot.tadah\|loss.tadah\|outvar.tadah\|params.tadah\|stats.tadah' | head -1)
[[ -f "$cfg_main" ]] || { echo "no main *.tadah config found in $bundle" >&2; exit 2; }

# Resolve tadah binary
if [[ -z "$tadah_bin" ]]; then
  if [[ -n "${TADAH_PATH:-}" && -x "$TADAH_PATH/build_tadah/tadah" ]]; then
    tadah_bin="$TADAH_PATH/build_tadah/tadah"
  elif command -v tadah >/dev/null; then
    tadah_bin=$(command -v tadah)
  else
    echo "tadah binary not found; pass -b <path>" >&2
    exit 2
  fi
fi
[[ -x "$tadah_bin" ]] || { echo "not executable: $tadah_bin" >&2; exit 2; }

# Resolve manifest
script_dir=$(cd "$(dirname "$0")" && pwd)
[[ -n "$manifest" ]] || manifest="$script_dir/strategy_matrix.manifest"
[[ -f "$manifest" ]] || { echo "manifest not found: $manifest" >&2; exit 2; }

[[ -n "$out_dir" ]] || out_dir="$bundle/strategy_matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out_dir"
summary_csv="$out_dir/summary.csv"
summary_md="$out_dir/summary.md"
echo "name,wall_s,best_loss,evals,exit" > "$summary_csv"

# Extract the non-OPTIMIZER body of the user's config.hpo (loss and
# OPTIM lines) so we can re-attach a per-entry OPTIMIZER block.
non_opt_body="$out_dir/_body.hpo"
awk '
  /^OPTIMIZER/      { in_block=1; next }
  /^ENDOPTIMIZER/   { in_block=0; next }
  in_block          { next }
  { print }
' "$bundle/config.hpo" > "$non_opt_body"

run_one() {
  local name="$1"
  local body="$2"
  local scratch="$out_dir/$name"
  mkdir -p "$scratch"
  cp -r "$bundle/." "$scratch/"
  rm -f "$scratch/best_"*.tadah "$scratch/loss.tadah" "$scratch/outvar.tadah" \
        "$scratch/params.tadah" "$scratch/stats.tadah" "$scratch/pot.tadah" \
        "$scratch/hpo_run.log"

  {
    printf '%s\n' "$body"
    cat "$non_opt_body"
  } > "$scratch/config.hpo"

  local cfg_basename
  cfg_basename=$(basename "$cfg_main")
  local t0=$SECONDS
  local exit_code=0
  ( cd "$scratch" && "$tadah_bin" hpo -c "$cfg_basename" --hpotarget config.hpo --validation config.val ) \
    > "$scratch/run.log" 2>&1 || exit_code=$?
  local t1=$SECONDS
  local wall=$(( t1 - t0 ))

  local best_loss="NaN"
  local evals="0"
  if [[ -f "$scratch/best_loss.tadah" ]]; then
    best_loss=$(awk '{ if (NR>1) print $2 }' "$scratch/best_loss.tadah" | tail -1)
    [[ -z "$best_loss" ]] && best_loss="NaN"
  fi
  if [[ -f "$scratch/loss.tadah" ]]; then
    evals=$(awk 'END { print NR-1 }' "$scratch/loss.tadah")
  fi
  echo "$name,$wall,$best_loss,$evals,$exit_code" >> "$summary_csv"
  printf '  -> %-24s wall=%4ss  best=%s  evals=%s  exit=%s\n' \
         "$name" "$wall" "$best_loss" "$evals" "$exit_code"
}

# Read manifest: lines of the form "name | <hpo body>"
# `name` must be a valid filesystem identifier.
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  name="${line%%|*}"
  body="${line#*|}"
  name=$(echo "$name" | xargs)        # trim
  body=$(printf '%b' "$body")          # honour \n etc.
  echo "[matrix] running $name"
  run_one "$name" "$body"
done < "$manifest"

# Pretty markdown summary
{
  echo "# strategy_matrix summary"
  echo
  echo "Bundle: \`$bundle\`"
  echo "Tadah:  \`$tadah_bin\`"
  echo
  echo "| name | wall (s) | best loss | evals | exit |"
  echo "|---|---:|---:|---:|---:|"
  tail -n +2 "$summary_csv" | awk -F, '{ printf("| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5) }'
} > "$summary_md"

echo
echo "summary CSV : $summary_csv"
echo "summary MD  : $summary_md"
