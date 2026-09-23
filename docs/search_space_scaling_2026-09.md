# Search-space scaling: design note (2026-09)

Why: `tadah hpo` handed every backend the raw OPTIM box; real EAM/blip searches mix widths across
5–10 decades (run15 9.7e5×, run23/26 2.5e7–3.7e7×, run21 4.9e9×) and 23–42 of ~90–174 dims straddle
zero, so `LOG_HP` could not help the worst offenders; `STEP` was one absolute scalar for every dim,
unreported, and meant something different per backend.

## Contract
- `SearchSpace` (`include/tadah/hpo/search_space.h`) owns per-dim shapes `LIN | LOG | ASINH c` and the
  affine map to `[0,1]` (`OPTIM_SCALE AUTO`, the default) or none (`NONE`, legacy raw coordinates,
  byte-identical to 1.4.x). Built once by `HPOTargets`, shared by every sub-spec (`bind_space_recursive`), the
  host and the journal.
- `OptimizerSpec::low/high` are INTERNAL; `in()/out()` map REAL<->INTERNAL; `host()` receives REAL;
  persisted artefacts are REAL-space. (The BASINS basin pools, removed in 1.5.1, were REAL-space too.)
- `STEP` = fraction of each dim's internal width, `(0, 0.5]`, per-key overrides, per-dim vector to
  NLopt (`initial_step.h`); rejected via `OptimizerCaps::StepUse` where a backend would ignore it.
- Analytical gradients are pulled back by `dtheta/du`; NUMERICAL FD uses the in-house width-relative
  kernel under AUTO (Ceres under NONE; `PARAM fd_kernel`).
- Startup validation (`scale_validation.{h,cpp}`): STEP policy at parse time; seeds inside the box,
  width audit inside `validate_hpo_setup`.
- Journal: SEARCH SPACE table (policy, width ratio, per-dim scale/origin/real+internal bounds/start),
  `Initial step` + `Step consumption` lines, run-time per-dim step table (NLopt), bound-occupancy
  tables at milestones and in DIAGNOSTICS; sidecars `hpo_search_space.tsv`, `hpo_optim_resolved.hpo`.
- AUTO boxes from the CONFIG seed (`OPTIM KEY (i) AUTO ...`), positive-only keys from the refit writer's
  table (`mlip::engines::hpotarget_group_specs()`).

## [0,1] guards (user chose [0,1] over [1,2])
in-house FD kernel under AUTO (Ceres' delta collapses to sqrt(eps) at u~0); per-dim `xtol_abs` vector +
XTOL_REL-only HINT for SBPLX/NM; ANNEAL `resolvable_coordinate_scale` floored at eps*width; ASA u=0
reanneal note; DLIB:GFS hidden-ln WARNING (NONE only).

## Gate results (2026-09-08)
`TADAH_OPTIM_SCALE=NONE`: 21/21 example fingerprints byte-identical. `AUTO`: 14/21 identical; 03/07/08
within 0.2 %, 06/12/21 better, 04 (DLIB:BFGS, 30 evals) 5.6 % worse = basin selection (at 4x budget NONE
finds the SGRIDMB~0.34 basin at 0.0643, AUTO the 0.24 basin at 0.0739 that example 03 also finds).
Default flipped to `AUTO` on 2026-09-08 (user decision); `NONE` remains the escape hatch.

## Not built
BOUND_ADAPT (`docs/issues/bound_adapt_design.md`, evidence-gated), CENTER WARM for AUTO boxes, the
tickets in `docs/issues/`.
