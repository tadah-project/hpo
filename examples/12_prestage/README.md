# 12 -- PRESTAGE strategy

Global explore + local refine, with mixed optimiser families.

What this shows:
- `STRATEGY { TYPE PRESTAGE, EXPLORE { ... } }` -- two-stage flow.
  The `EXPLORE` block runs a global sweep; the outer `LIB`/`ALGO`
  refines the top-K candidates.
- `TOPK <N>` -- how many EXPLORE candidates feed the refine stage.
- Mixed families: `NLOPT:GN_CRS2_LM` (Controlled Random Search,
  population-based global) for EXPLORE, `DLIB:BFGS` for REFINE.

`MAXEVAL` inside `EXPLORE` is the EXPLORE-stage budget; the outer
`MAXEVAL` is the per-refine budget (total cost is bounded by
`EXPLORE_MAXEVAL + TOPK * outer_MAXEVAL`).
