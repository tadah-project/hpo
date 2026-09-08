# 03 -- derivative-free local (NLOPT:LN_SBPLX)

Switching optimiser families.

What this shows:
- `LIB NLOPT` opens the full NLopt catalogue (prefix conventions:
  `LN_*` = local, no gradient; `GN_*` = global, no gradient;
  `LD_*` = local, gradient; `G*_*` = global, ...).
- `STEP` is the initial step as a *fraction of each OPTIM width*
  (`0 < STEP <= 0.5`; `0.1` = 10% of every box). NLopt receives one
  step per dimension, so a narrow box and a wide box get proportionate
  first moves; `hpo_run.log` prints the effective per-dimension step
  and how LN_SBPLX consumes it. Leave STEP out to use NLopt's own
  per-dimension default (25% of each width).

`LN_SBPLX` is a robust choice when the loss surface is noisy.
