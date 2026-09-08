# BOUND_ADAPT — opt-in, boundary-only OPTIM box adaptation (phase 7, evidence-gated)

Not implemented; designed. Wait for real-run evidence from the phase-5 bound-occupancy audit before
building it (user decision 2026-09-07: last phase, gated on audit data).

**Contract:** `BOUND_ADAPT … ENDBOUND_ADAPT` inside `STRATEGY`: `TRIGGER RESTART|ROUND` (ROUND requires
`TYPE BASINSV2`), `GROW <g>` (2.0, > 1), `SHRINK <s>` (0 = off), `EDGE <e>` (0.02), `MAX_FACTOR <f>` (8),
`MAX_ROUNDS <n>` (3), `ALLOW_KEYS …`, `EXCLUDE_KEYS …`. Applied ONLY by `RestartPolicy` (before the INIT
chain regenerates the start) and by `BasinsV2Policy` at a round boundary — never inside a driver, because
under `OPTIM_SCALE AUTO` the box *is* the coordinate system. Grow the pinned side so the width multiplies
by GROW (LOG/ASINH dims grow in t-space; positive-only keys never cross 0); optional SHRINK centred on the
best when the stage converged interior (never outside the original box); cumulative growth capped by
MAX_FACTOR / MAX_ROUNDS; cutoff keys (`RCUT*`, NN list sized for the original high bound) and hard
Cut_Dummy bump-channel keys excluded unless listed in ALLOW_KEYS (the bump gate must re-arm).

**Mechanics:** `SearchSpace::with_boxes()` (exists, version+1) → `HPOTargets::adopt_search_space()`
(rewrites optimap bounds, rebinds spec + sub-specs) → `HPO_Host::on_search_space_changed()` (re-arm the
occupancy audit and pin audit) → journal block old→new per dim → re-dump `hpo_optim_resolved.hpo` →
checkpoint `[SEARCH_SPACE]` flush. Pools are real-space and starts are regenerated in real space, so no
internal-space state survives a version change. Refuse under MPI (RESTART/BASINSV2 are single-rank).

**Tests:** `propose()` rules (grow only the marked side, geometric growth for LOG, caps, exclusions,
shrink never outside the original box); RESTART integration on a stub host; run → kill → edit one bound
→ resume (refusal vs adoption); manifest row `bound_adapt_restart`.
