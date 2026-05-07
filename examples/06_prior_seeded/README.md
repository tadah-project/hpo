# 06 — PRIOR-seeded BFGS

Each descriptor type can register a physical default for each of its
numeric arguments via `D_Base::suggested_value(key, arg_pos)`. PRIOR
walks the OPTIM map and asks each descriptor; for OPTIMs whose
descriptor has nothing registered (or the key is not descriptor-owned)
the resolver falls back to the bounds midpoint.

## Knobs introduced

- `INIT { STRATEGY PRIOR }` — no arguments. Generates exactly one
  starting point; all stochastic strategies (RANDOM/LHS/SOBOL) still
  need their K argument.

## What to look for

- `[HPO] PRIOR: N parameter(s) seeded from descriptor priors` — shows
  how many came from registered priors.
- `[HPO] PRIOR: N parameter(s) used bounds-midpoint fallback` — shows
  how many fell back. A high count usually means the descriptor type
  is new or doesn't expose a suggested_value override.
- The clamp-to-bounds step ensures the prior is always inside the
  user's configured box.
