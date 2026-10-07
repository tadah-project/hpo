# journal_status.sh -- sourced by the example scripts that print a run
# summary (common/run.sh, run_all.sh, and the run.sh of 16, 17 and 27).
#
#   journal_status <hpo_run.log>
#
# prints the journal's "Optimiser status" (the RESULT section's trailer).
journal_status() {
  awk -F': *' '/Optimiser status/ { print $2 }' "$1" | head -1
}
