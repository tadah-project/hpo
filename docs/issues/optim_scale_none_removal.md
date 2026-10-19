# Remove the OPTIM_SCALE NONE escape hatch one release after the AUTO flip; alias STEP/STEP_FRAC

`OPTIM_SCALE NONE` (raw internal coordinates, ln for LOG dims) exists so 1.4.x runs stay byte-identical
and for A/B (`TADAH_OPTIM_SCALE`). AUTO has been the default since 2026-09-08
(`docs/search_space_scaling_2026-09.md`), and NONE keeps two coordinate conventions alive (Ceres vs
in-house FD kernel default, |x|-relative tolerances mixing units). Precondition of the removal: each
test that sets `OPTIM_SCALE NONE` or `TADAH_OPTIM_SCALE` (`git grep -l "OPTIM_SCALE NONE" tests`:
`test_scale_parser.cpp`, `test_hpo_rhoform.cpp`, `test_startup_validation_strategy.cpp`,
`test_seed_anchor.cpp`, `test_hpo_termination.cpp`, `test_fd_stencil.cpp` and others) checks its
result under AUTO, or goes with NONE. Plan: deprecate with a WARNING for one release, then remove
together with the legacy `log_scaled` mask fallback in `OptimizerSpec::in()/out()` and the
`analyse_logscale()` test shim (tests build a `SearchSpace` directly).

Also: make `STEP` and the ANNEAL family's `PARAM STEP_FRAC` one registry-owned key (same meaning:
fraction of transformed width per dim) instead of two spellings.
