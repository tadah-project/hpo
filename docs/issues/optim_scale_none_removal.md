# Remove the OPTIM_SCALE NONE escape hatch one release after the AUTO flip; alias STEP/STEP_FRAC

`OPTIM_SCALE NONE` (raw internal coordinates, ln for LOG dims) exists so 1.4.x runs stay byte-identical
and for A/B (`TADAH_OPTIM_SCALE`). Once AUTO is the default and the reference suite has been
regenerated, NONE keeps two coordinate conventions alive (Ceres vs in-house FD kernel default,
|x|-relative tolerances mixing units). Plan: deprecate with a WARNING for one release, then remove
together with the legacy `log_scaled` mask fallback in `OptimizerSpec::in()/out()` and the
`analyse_logscale()` test shim (tests build a `SearchSpace` directly).

Also: make `STEP` and the ANNEAL family's `PARAM STEP_FRAC` one registry-owned key (same meaning:
fraction of transformed width per dim) instead of two spellings.
