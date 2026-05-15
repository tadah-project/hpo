# 06 -- explicit random start

Introduces the `INIT` block.

What this shows:
- `INIT { STRATEGY RANDOM K }` -- draws `K` uniformly-distributed
  points from inside the OPTIM bounds.
- `INCLUDE_CONFIG false` -- drop the implicit `config.tadah` seed; the
  init chain owns iter 0.
- `SEED <N>` -- make the draw reproducible across runs.

`K` controls how many points the optimiser sees as initial seeds.
For a single-shot local optimiser (DLIB:BFGS, NLopt LN_*) only the
first seed is consumed; for multi-seed consumers like `DLIB:GFS` all
`K` are used.

Available init strategies: `CONFIG`, `RANDOM`, `LHS`, `SOBOL`,
`PRIOR`, `WARM`. The next few examples cover each.
