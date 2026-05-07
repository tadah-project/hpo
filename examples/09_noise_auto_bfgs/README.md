# 09 — NOISE.AUTO: derive fd_step from box geometry

`DLIB:BFGS` uses a finite-difference gradient. The default step
(`PARAM fd_step = 1e-6`) is much smaller than the noise floor of a
typical LAMMPS-driven evaluation, so the computed gradient is
dominated by noise rather than signal. The `NOISE { AUTO }` block
derives a more reasonable step from the transformed-space box diagonal.

## Knobs introduced

- `NOISE { AUTO }` — bareword form (parses as `true`). Equivalent to
  `NOISE { AUTO true }`.
- The heuristic: `fd_step = max(1e-3 × diag / sqrt(dim), 1e-6)`. With
  the box `[4,6] × [0.1,0.4]`, the diagonal is sqrt(2² + 0.3²) ≈ 2.022,
  dim = 2, so `fd_step ≈ 1.43e-3`. Much wider than the default.

## What to look for

- `[HPO] NOISE.AUTO: PARAM fd_step set to 1.43e-03 (diag=2.022, dim=2)`
  — confirms the heuristic fired and shows what value was set.
- `Run policy: NOISE.AUTO : true` recorded in `hpo_run.log`.
- If you also wrote `PARAM fd_step <value>`, the heuristic is
  suppressed: `[HPO] NOISE.AUTO: PARAM fd_step already set explicitly;
  user value retained`.
- For non-`DLIB:BFGS` algorithms, `NOISE { AUTO }` is a no-op with a
  stderr warning so the user notices the keyword had no effect.

## Alternatives

- For derivative-free local refinement, switch to `NLOPT:LN_SBPLX` or
  `LN_BOBYQA`. See example 10 for that pattern.
