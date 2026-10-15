# ASA reannealing uses |x|-relative tangents and caps at the initial temperature

**Where:** vendored `HPO/external/asa/asa.c:5195-5210, 5244-5245, 4873-4889`; adapter `src/asa_options.cpp`
(`Reanneal_Parameters` default 1, `Delta_X` 1e-3).

**What happens:** ASA's generating kernel is width-relative (`x + z(T_i)·range_i`), so it is invariant
under the unit-box transform, but reannealing rescales the per-dim temperatures by
`max_tangent / tangent_i` with `tangent_i = (C(x_i(1+Δ)) − C_best) / (Δ·x_i + 1e-18)` — a raw-unit
derivative whose offset is multiplicative in `x_i`. A coordinate at 0 (every lower bound of the unit box)
gets no tangent (the journal says so), and the reannealed temperature is capped at the initial one, so
reannealing re-heats insensitive dims rather than cooling sensitive ones.

**Proposal:** decide the `Reanneal_Parameters` default for HPO (0 keeps the width-invariant kernel
pure); if reannealing stays on, a fenced `TADAH-PATCH` hunk could compute tangents with an absolute
offset `Δ·range_i` instead of `Δ·x_i` (see `external/asa/TADAH-PATCHES.md` for the patch policy and the
golden-neutrality gate that must stay green).

**Test:** ASA golden test with `Reanneal_Parameters 1` on a 2-dim bowl with widths 1 and 1e-3 under
`OPTIM_SCALE NONE` vs `AUTO`: temperatures must evolve identically once tangents are width-relative.
