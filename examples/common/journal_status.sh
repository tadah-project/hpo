# journal_status.sh -- sourced by the example scripts that print a run
# summary (common/run.sh, run_all.sh, and the run.sh of 16, 17 and 27).
#
#   journal_status <hpo_run.log>
#
# prints the journal's "Optimiser status" (the RESULT section's trailer),
# the whole value: a status has colons of its own ("TADAH:RANDOM: MAXEVAL
# reached", "ASA: MAXEVAL reached", "STOPPED -- stagnation: ..."). A script
# that must tell success from failure tests tadah's exit status, not this
# text.
journal_status() {
  sed -n 's/^[[:space:]]*Optimiser status[[:space:]]*: //p' "$1" | head -1
}
