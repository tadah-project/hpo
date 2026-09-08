# Journal / TSV labels are 0-based token positions; OPTIM lines are 1-based

`KEY[i]` in `hpo_run.log` (SEARCH SPACE, step table, pin audit), `hpo_step_scales.tsv`, ASA
diagnostics and PT tables uses the 0-based raw config-token position (`AMPGRIDMB[0]` for
`OPTIM AMPGRIDMB (1)`), while `OPTIM`, `Opti::print` "Index:" and `best_params.tadah` headers are 1-based.
`DimSpec::user_index` (phase 1) already carries the 1-based OPTIM index; the sidecar
`hpo_search_space.tsv` prints both.

**Proposal:** switch every journal label to `KEY(idx)` (1-based, matching the OPTIM line) in one commit;
consumers: `HPOTargets::build_search_space` label builder, `algorithms_asa.cpp:600`,
`algorithms_pt.h:2179` (TSV), pin audit, tests asserting `KEY[i]`. External scripts parsing
`hpo_step_scales.tsv` must adapt — CHANGELOG "Changed" line.
