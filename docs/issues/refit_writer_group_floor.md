# Refit hpotarget writer: group-wide bound floor manufactures the within-key width spread

**Where:** `MLIP/src/hpotarget_writer.cpp:31-62, 82-101` (`optim_group_floor`, `optim_group_bound`,
`emit_optim_group`), consumed by `tadah refit` / `--retarget`.

**What happens:** each `OPTIM FIXWEIGHT (i) lo hi` line is `v ± max(OPTIM_PCT%·|v|, floor)` with
`floor = 1e-3 · max|v|` over the WHOLE key group. For the Ta EAM refit (`HPO_REFIT_2B_EAM/refit_current/Ta`,
`max|v| = 38206.8`) the floor is 38.2, so 13 of 28 coefficients (|v| = 0.015 … 37) get the identical
`±38.2` box regardless of their own magnitude, 13 of 28 boxes span zero, and widest/narrowest is
exactly `pct / 0.1% = 50`. The header comment says "±5 %" but only 15/28 dims are.

**Why it matters:** with per-dimension step scaling and the unit box the
optimiser sees *widths*, so the writer's floor — not physics — decides the narrowest box every
step derives from. `HPO_REFIT_2B_EAM/demos/Ta` (LN_BOBYQA on that space) died `nlopt roundoff-limited`
at 109/200 evals; the other four elements ran to MAXEVAL. Not diagnosed.

**Proposal:** per-coefficient or magnitude-banded floors (e.g. `floor_i = 1e-3 · |v_i|` floored at an
absolute ε derived from the coefficient's basis-function scale), or emit
`OPTIM FIXWEIGHT (i) AUTO FRAC 0.05 FLOOR <eps>` lines now that HPO derives seed-relative boxes
itself. Add a writer guard for boxes narrower than 1e-6 of |v| (the run21 `SEMBFUNC(13)`
1e-4-wide box is a retarget/hand-edit artefact the guard would catch).

**Test:** writer unit test on a synthetic coefficient vector spanning 1e-2 … 1e4: no two coefficients
of different magnitude get identical boxes; no box spans zero unless the seed is within ε of zero.
