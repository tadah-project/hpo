# 05 — WARM-start refinement

Reads the best parameters of a previous run from `seed_pot.tadah` and
refines from there. This is the typical workflow after a global search
has identified a basin — switch the optimiser to a local refinement and
seed it with the previous optimum.

## Knobs introduced

- `INIT { STRATEGY WARM <file> }` — opens a previously written
  `best_pot.tadah` and pulls the value of every OPTIM-flagged key it
  can find. Keys not present in the file fall through to the next
  chain entry (or the in-config value when CONFIG is implicit).
- `INIT { WARM_ON_MISMATCH warn|error|skip }` — controls behaviour when
  WARM cannot fill every OPTIM key. `warn` (default) logs to stderr
  and falls through; `error` throws; `skip` returns no point.
- The implementation also scans `# HOT_A0 <script> <a0>` lines saved
  by the previous run and primes the warmstart-cache so the LAMMPS
  scripts begin their box/relax from the converged lattice constant
  (avoids landing in a different basin and reporting a different loss
  for what should be the same parameter vector).

## Reorder rule

Even if you write `STRATEGY WARM ...` after another strategy in your
`INIT { ... }` chain, the resolver moves it to position 0 — iter 0 is
always the warm point when a chain contains one. The diagnostic
`[HPO] InitStrategy: WARM moved to chain front` fires when this happens.

## How to seed

The example's `run.sh` checks for `seed_pot.tadah`; if missing, it runs
example 01 first and copies the resulting `best_pot.tadah` here. You
can also bring your own.
