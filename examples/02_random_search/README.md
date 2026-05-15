# 02 -- random search baseline

Built-in random sampler.

What this shows:
- `LIB TADAH`, `ALGO RANDOM` -- TADAH's uniform random sampler.
- `SEED` makes the sequence reproducible.
- A 2-dimensional search space (`RCUT2B`, `SGRIDMB`).

When to use this: as a baseline. If a gradient method does not beat
random search at the same `MAXEVAL`, the gradient run is stuck.
