# 03 -- derivative-free local (NLOPT:LN_SBPLX)

Switching optimiser families.

What this shows:
- `LIB NLOPT` opens the full NLopt catalogue (prefix conventions:
  `LN_*` = local, no gradient; `GN_*` = global, no gradient;
  `LD_*` = local, gradient; `G*_*` = global, ...).
- `STEP` sets the initial simplex / trust-region size, which most
  NLopt algorithms expose.

`LN_SBPLX` is a robust choice when the loss surface is noisy.
