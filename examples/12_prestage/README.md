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
`MAXEVAL` is the refinement's, split over the `TOPK` candidates (ceil
division), so the total is at most
`EXPLORE_MAXEVAL + TOPK * ceil(outer_MAXEVAL / TOPK)`: here 60 + 3 x 10
= 90 (the journal says `MULTISTART per-start MAXEVAL = 10 (split from
outer MAXEVAL)`).
