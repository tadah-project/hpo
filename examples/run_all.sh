#!/usr/bin/env bash
# Run every numbered example in series, then print a one-row summary
# per example. Each example writes its own `hpo_run.log` and the
# best/eval-count is parsed from there.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

failed=0
for dir in "$HERE"/[0-9]*; do
  [[ -d "$dir" ]] || continue
  name=$(basename "$dir")
  echo "==================== $name ===================="
  if [[ -x "$dir/run.sh" ]]; then
    if ! ( cd "$dir" && bash ./run.sh ); then
      failed=$((failed+1))
      echo "  ! $name failed"
    fi
  else
    if ! ( cd "$dir" && bash "$HERE/common/run.sh" ); then
      failed=$((failed+1))
      echo "  ! $name failed"
    fi
  fi
done

echo
echo "==================== summary ===================="
printf '%-32s | %12s | %6s | %s\n' "example" "best loss" "evals" "status"
echo "------------------------------------------------------------------------"
for dir in "$HERE"/[0-9]*; do
  [[ -d "$dir" ]] || continue
  name=$(basename "$dir")
  log="$dir/hpo_run.log"
  if [[ -f "$log" ]]; then
    best=$(awk '/Best total loss/ { for (i=1;i<=NF;i++) if ($i ~ /^[0-9.eE+-]+$/) { print $i; break } }' "$log" | head -1)
    evals=$(awk '/Total evaluations/ { print $4 }' "$log" | head -1)
    status=$(awk -F': *' '/Optimiser status/ { print $2 }' "$log" | head -1)
    printf '%-32s | %12s | %6s | %s\n' "$name" "${best:-?}" "${evals:-?}" "${status:-?}"
  else
    printf '%-32s | %12s | %6s | %s\n' "$name" "?" "?" "no log"
  fi
done

exit $failed
