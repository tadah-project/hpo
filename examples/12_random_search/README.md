# 12 — TADAH:RANDOM uniform-random sweep

Pure random sampling inside the bounds. The TADAH:RANDOM engine takes
the in-config point as eval 0 (so the user's prior is always part of
the explored set) and then samples uniformly for `MAXEVAL - 1` more
evals.

## Knobs introduced

- `STRATEGY { TYPE PLAIN, LIB TADAH, ALGO RANDOM }` — the bundled
  brute-force sampler. No acquisition strategy, no local refinement.
- `STRATEGY { SEED <int> }` — seeds the RNG (the C `srand`-equivalent
  inside the TADAH adapter). Different from `INIT { SEED }` (which
  seeds only the init chain).

## Diagnostic line

`[HPO] RANDOM search: seed=1234` confirms the seed and that the
TADAH:RANDOM dispatch path was taken (vs DLIB:RAND or NLOPT).

## Why it is useful

Random search is a shockingly strong baseline in low-D problems and
is easy to compare against. Run it alongside example 04 (Sobol-seeded
GFS) and example 03 (LHS+MULTISTART) to see whether the smarter
engines actually beat random for your problem.
