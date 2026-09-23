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

Useful for unattended overnight runs where a single BFGS getting
stuck in a poor local minimum shouldn't waste the whole job.
