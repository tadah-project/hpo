# Route the refit CompactFitter's per-dim NLopt steps through make_step_plan

`src/refit_compact_fitter.cpp:623-635` hand-rolls a per-dimension initial step vector
(`0.12·span` for knot centres, `0.4·eta0` for widths) and drives `nlopt::opt` directly; the refit
stage `OPTIMIZER` blocks accept `STEP` / `LOG_HP` / `OPTIM_SCALE` syntactically but ignore them
(`hpo_refit_engine.cpp:69-100`). Phase 0 rejects them; unification is deferred.

**Proposal:** build a `SearchSpace` from `theta_box`, express the two tuned fractions as
`StepOverride`s (`STEP CGRID 0.12`, `STEP SGRID …`) and call `make_step_plan`; then honour a
user `STEP` in refit stages. The tuned numerics must be reproduced first (fingerprint the refit
demos before switching).
