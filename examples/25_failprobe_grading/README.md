# 25 -- graded failures (FAILSCORE_GRADING, PC_FAILPROBE)

Example 02's seeded random search with a tight lattice-constant window,
so that some candidates fail, and graded failure losses.

What this shows:
- `--hotvar hot_a0 3.25 3.33 3.302` on the cohesive-energy script: a
  candidate whose relaxed bcc lattice leaves the window (3.25-3.33 A
  around Ta's 3.30 A) fails. Eight of the 60 candidates do (evaluations
  9, 13, 28, 31, 41, 42, 51 and 57): the journal's summary counts them
  (`Graded failures : 8  (13.3%)`, `A=8 (hot-var out of range)`). Its
  per-evaluation `rejecting (hot_a0=... outside [3.25, 3.33] ...)`
  warnings are rate-limited -- the first five, then every power of two --
  so it shows the first five and the eighth.
- `FAILSCORE_GRADING ON` with `FAILSCORE 1e9`: a failure scores a
  graded value in `[0.999e9, 1e9)` that falls as the lattice approaches
  the window (band A), instead of the constant `FAILSCORE`: the eight
  failed rows of `loss.tadah` have a `TOTAL_LOSS/N` of about `9.990e8`
  (from `9.99005e8` to `9.99089e8`).
  Feasible candidates are scored as usual. Example 02 draws the same
  candidates (the same `SEED`); two of its four best ones (evaluations
  42 and 28, its second and fourth) relax just outside the window, its
  best, evaluation 29, inside it, so the best here is example 02's. The
  reference fingerprint pins it and also requires the two summary lines
  of the journal.
- `PC_FAILPROBE --script ../common/lammps/failprobe.in --outvar S
  --s0 0.5`: the minimiser-free sanity probe from the documentation, run
  at the front of every evaluation. Its scalar `S` (`outvar.tadah`
  column `S`) is 0 for a sane potential, as here. It grades the failures
  that cannot measure themselves -- a run killed at its deadline, a
  crashed potential build (bands C and D) -- which this well-behaved
  box never produces. No `--gate`: set one only after calibrating `S`
  on your seed.

A random search draws the same candidates whatever their losses, which
keeps the fingerprint robust; the ordering the graded values give is
what the annealers (examples 15, 18-20) and the local optimisers use to
walk out of a failing region.

Run:

    bash ../common/run.sh
