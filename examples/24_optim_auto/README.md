# 24 -- seed-relative search boxes (OPTIM AUTO)

Example 03 with the `SGRIDMB` box derived from the seed instead of
written out.

What this shows:
- `OPTIM SGRIDMB (1) AUTO`: the box is sized around the value in
  `../common/config.tadah` (`SGRIDMB 0.2`). `SGRIDMB` is positive-only,
  so the default rule is `RATIO 2`, the geometric box `[0.2/2, 0.2*2]`
  = `[0.1, 0.4]` -- exactly the explicit `SGRIDMB` box of example 03 and
  the other examples that search `SGRIDMB` (02, 04, 06-08, 11, 12, 14,
  15, 18-21, 25).
- `hpo_optim_resolved.hpo`: the resolved boxes as a paste-ready explicit
  `OPTIM` block, and the `origin` column of `hpo_search_space.tsv`
  (`AUTO RATIO`).
- Parity with example 03 (`NLOPT:LN_SBPLX`, `STEP 0.1`): the same box
  gives the same run, so `loss.tadah`, `params.tadah` and the reference
  fingerprint match 03's.

`RCUT*` keys cannot be `AUTO` (the neighbour list is sized from an
explicit upper bound), so `RCUT2B` keeps its `4.0 6.0` box.

Run:

    bash ../common/run.sh
