# 09 -- descriptor prior

The starting point comes from the descriptor's own recommendation.

What this shows:
- `STRATEGY PRIOR` -- descriptors that register a `suggested_value`
  hand it back as the iter-0 seed. Descriptors that have not
  registered one fall back to the bounds midpoint, with a warning in
  `hpo_run.log`.

Use when you have no `seed_pot.tadah` from a previous run and the
in-config value is a placeholder.
