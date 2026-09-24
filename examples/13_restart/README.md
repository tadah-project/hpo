# 13 -- RESTART decorator

Auto-recover when the inner optimiser stalls.

What this shows:
- `STRATEGY { ... RESTART { ... } }` -- decorator that wraps the
  outer strategy. Knobs:
  - `AFTER_STAGNATION <N>` -- restart after N evals without improvement.
  - `MAX <K>` -- give up after K restarts.
  - `STRATEGY <init-strategy>` -- how to seed each restart
    (`RANDOM`, `LHS`, `SOBOL`); `STRATEGY` is the alias of the canonical
    `METHOD` keyword.
  - `SEED <n>` (not set here) -- restart r draws its point with
    `SEED + r`, so the restart points repeat from run to run; without
    it each restart draws from `std::random_device`.

Useful for unattended overnight runs where a single BFGS getting
stuck in a poor local minimum shouldn't waste the whole job.
