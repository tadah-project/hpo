# 04 -- skip obviously-broken potentials

Reject bad pots before the expensive LAMMPS phase.

What this shows:
- `PC_<METRIC>  <target> <weight>  --skip-above <cap>  --skip-rel <ratio>`
  attaches two cheap pre-filters to a validation metric:
  - `--skip-above`: skip if `|metric| > cap` (cap in the metric's
    native units, e.g. meV/atom for ERMSE).
  - `--skip-rel`: skip if `metric > ratio * best_so_far`.
- Both filters apply to every PC_LAMMPS line in the same iter --
  one bad pot saves all subsequent LAMMPS calls in that iter.

Supported metrics: `EMAE`, `ERMSE`, `ErRMSE`, `ERSQ`, `FRMSE`, `SRMSE`.
