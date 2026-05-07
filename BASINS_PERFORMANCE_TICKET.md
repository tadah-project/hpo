# BASINS performance improvements — autonomous implementation ticket

You are a coding agent picking up this ticket cold. The user is not
available for clarifications. Implement, test, and verify the five
recommendations below in the listed order. Each one ships independently
with its own regression test, performance measurement, and short
write-up.

## Pre-conditions (do not start until both are true)

1. The bug ticket `HPO/HPO_EVAL_OUTPUT_BUGS.md` (Bugs 1–5) is fully
   resolved. Verify with:
   ```
   cd /home/marcin/PROJECTS/TADAH/develop_wt/build_tadah && \
     ctest -R '^(test_hpo|test_basin|test_progress|test_tolerance|test_checkpoint|test_writer|test_noise|test_resume|test_eval_thread)' \
     --output-on-failure
   ```
   Every listed test must pass. If `test_eval_thread_throw_recovery`
   does not yet exist, Bug 5 is unresolved — stop and surface that to
   the user before continuing.

2. A clean build of `tadah` succeeds:
   ```
   TADAH_PATH=/home/marcin/PROJECTS/TADAH/develop_wt \
     cmake --build /home/marcin/PROJECTS/TADAH/develop_wt/build_tadah --target tadah -j 8
   ```

If either pre-condition fails, do not implement any of this ticket's
recommendations. Investigate and resolve the upstream issue first, or
report blockage in your final write-up.

---

## Shared context for all five recommendations

### Source root and build
```
/home/marcin/PROJECTS/TADAH/develop_wt
TADAH_PATH=$_/. cmake --build /home/marcin/PROJECTS/TADAH/develop_wt/build_tadah --target tadah -j 8
```

### Files you will be editing
| File | Role |
|---|---|
| `HPO/include/tadah/hpo/basins_policy.h` | BasinsPolicy declaration |
| `HPO/include/tadah/hpo/basins_policy_impl.hpp` | BasinsPolicy template impl (run, run_global_stage, run_local_stage, build_stage_spec, commit_checkpoint, save, load) |
| `HPO/include/tadah/hpo/optimizer_spec.h` | `BasinsSpec` struct — extend with new knobs |
| `HPO/src/hpo_targets.cpp` | Parser for new keywords in `OPTIMIZER … LIB BASINS … ENDOPTIMIZER` |
| `HPO/include/tadah/hpo/basin_pool.h` + `basin_pool.cpp` | Basin pool (heads, dedup, eviction) |
| `HPO/include/tadah/hpo/tolerance_ladder.h` + `.cpp` | Auto/explicit tolerance ladders |
| `HPO/src/hpo_log.cpp` | Add new log lines for diversification, ladder clamps, pool-collapse stop |

### Reference workdirs (read-only data sources for measurement)
- `/tmp/hpo_bug_verify/` — completed 5h baseline run with `BUDGET_MAX_EVALS 300`, MAXEVAL 25000, NLOPT GN_CRS2_LM + LN_SBPLX defaults, hit MAXEVAL_REACHED with best=396.95 plateaued from eval 605 onward across 5 unproductive rounds. Use as the **performance baseline**.
- `/home/marcin/TEMP/Izzy/Ta_LAMMPS_fit_typical_mod/ATOMISED/workdir/ITER5/` — full Ta MLIP HPO setup with `BUDGET_MAX_EVALS 15000`, `Ta_config.basins.tadah` (load transforms enabled), 22 PC_LAMMPS targets. Use as the **integration test** workdir. Always copy first; never modify in place.

### Per-recommendation lifecycle
For each recommendation R1…R5:

1. **Implement** the change and any required new keywords/parameters.
2. **Compile** `tadah` and the tests. Build failure → fix root cause,
   not symptoms.
3. **Write a unit/regression test** that verifies the behaviour change
   in isolation (small synthetic objective, runs in seconds). Test must
   fail on the pre-recommendation binary (verify by stashing the change,
   running the test, asserting failure; restore the change before
   continuing).
4. **Run the integration smoke** on a scratch copy of `/tmp/hpo_bug_verify`
   inputs at reduced budget (e.g. MAXEVAL 5000, BUDGET_MAX_EVALS 200) so
   the run completes in ≤ 30 minutes. Capture:
   - total wall time
   - eval count and per-eval mean
   - best loss vs eval-number trajectory (extract from
     `grep "NEW BEST" hpo_run.log`)
   - per-round timing breakdown (extract from
     `grep -E "round=. (GLOBAL|LOCAL)" hpo_run.log`)
   - count of `nlopt failure`, `LN_SBPLX`, `BAD_POT skipping` warnings
5. **Compare** the integration smoke against the prior baseline (R0 =
   ticket-start, R1, R2, …). Report deltas:
   - Δ best loss after N evals (expect: equal-or-better at every N)
   - Δ time-to-first-best (expect: equal-or-faster)
   - Δ rounds before convergence (expect: equal-or-fewer)
   - Δ wall-clock per round (expect: equal-or-faster)
6. **Commit** the change as a single atomic commit:
   `[BASINS] R<N>: <one-line summary>` with the test, code, and
   docstring updates together. Use HEREDOC for the message.
7. **Mark the recommendation done** in this file (append a `## R<N>
   resolution` section as you go).

If a recommendation regresses any baseline metric without a
load-bearing reason, **stop and surface the regression**. Do not
proceed to the next recommendation. Roll back via `git revert HEAD` if
necessary; do not let regressions accumulate.

You may add helper utilities (e.g. a Python script in
`/home/marcin/PROJECTS/TADAH/develop_wt/HPO/scripts/` to extract
metrics from `hpo_run.log` and emit a CSV) — keep them small and
single-purpose.

### What "passes" means for the suite

After every recommendation, the full HPO+CORE test suite must remain
green:
```
ctest -R '^(test_hpo|test_basin|test_progress|test_tolerance|test_checkpoint|test_writer|test_noise|test_resume|test_eval_thread)' --output-on-failure
```

The `test_factory_cutoffs` failure in MODELS is pre-existing and
unrelated; ignore it.

### Do not change

- The HPO writer-thread design (Bug 4 fix is load-bearing).
- The `RunPolicy` decorator pattern.
- BASINS state-machine grammar at the user level — only add new
  keywords, do not rename existing ones.
- The on-disk format of `checkpoint.tadah` for fields that already
  exist. New fields go in new sections or with new keys.

---

## R1 — Diversify GLOBAL stage seeding in rounds k > 1

### Problem
`BasinsPolicy::run_global_stage` currently seeds the round-k>1 GLOBAL
optimiser from a single point: `pool.heads()[0].params` (the best
basin). The remaining 15+ basins, with diverse parameter vectors and
losses ranging 5–10× apart, never seed a GLOBAL search. Evidence:
in `/tmp/hpo_bug_verify`, rounds 2 through 5 produced **zero best-loss
improvement** despite each running a full GN_CRS2_LM stage at
progressively tighter ftol_rel.

### Required change
1. Extend `BasinsSpec` (in `optimizer_spec.h`) with:
   - `std::size_t global_seed_top_n = 4;` — number of top pool basins
     to use as initial population (default 4, capped at pool size).
   - `std::size_t global_seed_lds_topup = 0;` — number of additional
     LDS samples to mix in (default 0; user can opt in).
2. Add the parser keys `SEED_TOP_N` and `GLOBAL_SEED_LDS_TOPUP`
   in `hpo_targets.cpp::process_optim_config`.
3. In `BasinsPolicy::run_global_stage`, when `round > 1`, build the
   initial-population vector from:
   - `min(global_seed_top_n, pool.size())` basins (sorted by loss
     ascending) → their `params`
   - `global_seed_lds_topup` fresh LDS samples in the bbox (use the
     existing LDS sampler from `init_strategies`; if not exposed,
     extract a small static helper)
4. Pass the population to GN_CRS2_LM via `nlopt_set_population` (NLopt
   C++ API: `opt.set_population(N)` then seed by calling `opt.optimize`
   with the first member; for the rest, populate via NLopt's
   internal LDS — or pre-evaluate the seeds and use the lowest-loss as
   the optimisation start). The cleanest path is to keep using the
   single-point seed but add `set_population(top_n + lds_topup)` so
   GN_CRS2_LM allocates a larger internal population. Do NOT try to
   overwrite NLopt's internal population vector — use the public API.
5. If the user picks a non-population-based EXPLORE.ALGO (e.g. DLIB:GFS),
   the diversification should fall back to: pre-evaluate all top-N
   basins, take the best, and use it as the single seed (no behaviour
   change there). Log this fallback at INFO level.

### Test
`HPO/tests/test_basins_diversification.cpp` (Catch2):
- Synthetic objective: 5-D bumpy landscape with two basins at
  different locations.
- Run BASINS with `BASIN_POOL_MAX 4`, `SEED_TOP_N 4`,
  `MAXEVAL 200`, `BUDGET_MAX_EVALS 50`.
- Assert that across rounds 2–3 the GLOBAL stages explore both
  basins (track which basin each best falls into; with old behaviour
  one basin dominates).
- Assert `set_population` was called with ≥ 4.

### Performance measurement
Run the integration smoke on a scratch copy of `/tmp/hpo_bug_verify`
inputs at MAXEVAL 5000, BUDGET_MAX_EVALS 200, SEED_TOP_N 4.
Compare against R0 (no diversification):
- best-loss-after-{500,1000,2000,5000} evals
- count of distinct basins visited (cluster the per-iter params
  using L2 in normalised bbox; report the count)

Pass criterion: **at least one of (i) best loss after 5000 evals is
≤ R0's best loss after 5000 evals, OR (ii) basin diversity (distinct
clusters in the params trajectory) is at least 2× R0's**. R0's best
plateaued at 396.95 by eval 605; if R1's best at eval 5000 is ≤ 396.95
and the trajectory shows visits to ≥ 2 clusters, R1 is delivering on
its goal.

If pass criterion fails, the diversification mechanism is not
working as designed — investigate before proceeding.

---

## R2 — Clamp LOCAL ladder against the noise-calibration floor

### Problem
`NOISE.CALIBRATION` runs N evals at the seed point, computes
`rel_spread`, and writes `suggested FTOL_REL = max(10*rel_spread, 1e-6)`
to the journal — but this value is purely informational. The
configured `REFINE.FTOL_LADDER` can extend below it without any clamp.
Evidence: `/tmp/hpo_bug_verify` had `REFINE.FTOL_LADDER 1e-3 1e-4 1e-5
1e-6 1e-7` and `suggested FTOL_REL = 1.0e-6`. Round 5's LOCAL stage at
ftol=1e-7 burned 56 minutes refining 16 basins with zero improvement —
the entire stage was below the noise floor.

### Required change
1. Persist the noise-calibration result (rel_spread, suggested_ftol)
   into the `[HPOLOG]` section of the checkpoint so it survives
   resumes. Already done? Verify and add if missing.
2. After NOISE.CALIBRATION runs (or on resume, after `cpm.try_load`),
   pass `suggested_ftol` into `BasinsSpec` (or into a new
   `OptimizerSpec::noise_floor` field).
3. In `BasinsPolicy::ensure_ladder` (called once at run start), clamp
   `tol_loc_ladder_` and `tol_glo_ladder_` against the floor:
   ```cpp
   const double floor = 0.1 * noise_floor_;   // one decade below
   while (!tol_loc_ladder_.empty() && tol_loc_ladder_.back() < floor)
       tol_loc_ladder_.pop_back();
   if (tol_loc_ladder_.empty())
       tol_loc_ladder_.push_back(floor);     // never empty
   ```
   Mirror for `tol_glo_ladder_`.
4. Log the clamp at INFO level: "BASINS: LOCAL ladder clamped to <new>
   rungs (floor=<value> from noise calibration)".
5. Make the clamp opt-out via a new spec field
   `bool clamp_ladder_to_noise_floor = true;` and parser key
   `CLAMP_TO_NOISE_FLOOR true|false`.

### Test
`HPO/tests/test_tolerance_ladder_noise_clamp.cpp` (Catch2):
- Build a `BasinsSpec` with `REFINE.FTOL_LADDER {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}`.
- Set `noise_floor = 1e-6`.
- Call `ensure_ladder()`.
- Assert ladder is `{1e-3, 1e-4, 1e-5, 1e-6}` (1e-7 dropped).
- Repeat with `noise_floor = 1e-3` and assert ladder is `{1e-3}`
  (only the loosest rung survives the floor).
- Repeat with `clamp_ladder_to_noise_floor = false` — assert ladder
  is unchanged.

### Performance measurement
Same workdir as R1. Compare R2 against R1:
- total wall time (expect: shorter when ladder is clamped, by ≈ the
  time of the dropped rungs — for `/tmp/hpo_bug_verify`'s baseline
  that's ~56 min/dropped rung at full MAXEVAL).
- best loss at end (expect: unchanged or marginally better — clamped
  rungs were noise-floor refining anyway).

Pass criterion: **R2 wall time ≤ R1 wall time, AND R2 best loss ≤
1.05 × R1 best loss** (allows small fluctuation since clamping does
not change the optimisation trajectory upstream of the floor).

---

## R3 — Switch default REFINE.ALGO to LN_BOBYQA

### Problem
LN_SBPLX hits "starting step size led to simplex too small in
dimension X: <bound> is too close to x[X]=<bound>" when a
parameter is at a bound. NLopt declares `XTOL_REACHED` and exits
without refining. BasinsPolicy then marks the basin
`exhausted_at_tol=<current ftol>`. No real refinement happened.
Evidence: live ITER5 had 8 such warnings in 2 hours; each one
silently consumed a basin's budget for that tolerance level.

LN_BOBYQA uses a quadratic trust-region with explicit bound
handling and does not collapse at edges. It's also derivative-free
and noise-tolerant, matching SBPLX's strengths without its bound
weakness.

### Required change
1. Change the default in `BasinsSpec`:
   ```cpp
   std::string local_lib  = "NLOPT";
   std::string local_algo = "LN_BOBYQA";   // was: "LN_SBPLX"
   ```
2. In `BasinsPolicy::build_stage_spec`, when REFINE.ALGO is `LN_SBPLX`
   AND a basin has hit a bound on at least one parameter, log an
   INFO suggesting `LN_BOBYQA` for that basin. (Not a behaviour change
   — informational nudge for users who configured LN_SBPLX
   explicitly.)
3. Update `HPO/examples/15_basins_run_and_forget/config.hpo` and
   `HPO/examples/15_basins_run_and_forget/README.md` to reflect the
   new default.

### Test
`HPO/tests/test_basins_local_default_bobyqa.cpp` (Catch2):
- Build `BasinsSpec` with no REFINE.ALGO override.
- Assert `spec.basins.local_algo == "LN_BOBYQA"`.
- Build with explicit `REFINE.ALGO LN_SBPLX` parsed from a string
  config — assert that survives unchanged.

The behavioural fix can also be exercised by a synthetic test:
- Define a 2-D objective with a basin minimum at (0.5, 1.0) and
  bounds [0.0, 1.0] × [0.0, 1.0]. The optimum is at the upper bound
  of dim 1.
- Run BASINS LOCAL refinement starting from (0.5, 0.99).
- Assert the refined point is within `1e-3` of (0.5, 1.0) — SBPLX
  would simplex-collapse and stop at (0.5, 0.99); BOBYQA finds
  (0.5, 1.0).

### Performance measurement
Run the integration smoke on the **live ITER5** workdir (use a scratch
copy at reduced MAXEVAL=5000, BUDGET_MAX_EVALS=200) — this workdir
hits the bound issue, the verify workdir does not. Compare R3 vs R2:
- count of `simplex that was too small` warnings (expect: 0)
- count of basins marked exhausted-at-tol after a bound-near
  refinement (expect: substantially lower)
- best loss after 5000 evals (expect: ≤ R2's best loss; the basins
  that previously fake-exhausted now get genuine refinement)

Pass criterion: **R3 simplex-collapse warnings = 0, AND R3 best loss
≤ R2 best loss**.

---

## R4 — Pool-collapse early stop

### Problem
The current `STOP_NO_IMPROVE_ROUNDS` (default 5) requires 5 full
rounds of zero best-loss improvement before stopping. At ~60 min/round
that's a guaranteed 5 hours of dead time when the pool has actually
collapsed (all basins refined to the same local minimum). Evidence:
`/tmp/hpo_bug_verify` ran rounds 2–5 (4 hours) producing no
improvement; by end of round 1 all 16 basins were within a small
fraction of best=396.95.

### Required change
1. Add a method `BasinPool::heads_loss_spread() const` returning
   `(max - min) / max(min, eps)` over the pool's losses.
2. Extend `BasinsSpec` with:
   - `double pool_collapse_rel_spread = 0.10;` — if pool spread is
     below this AND `rounds_no_improve >= 1`, trigger emergency.
   - Parser key `POOL_COLLAPSE_REL_SPREAD <value>` (zero/negative
     disables the new check).
3. In `BasinsPolicy::run`, after each round's completion check, add:
   ```cpp
   if (sp.basins.pool_collapse_rel_spread > 0.0 &&
       rounds_no_improve_ >= 1 &&
       pool_.heads_loss_spread() < sp.basins.pool_collapse_rel_spread) {
       log.info("BASINS: pool collapsed (spread=" +
                std::to_string(pool_.heads_loss_spread()) +
                " < " + std::to_string(sp.basins.pool_collapse_rel_spread) +
                ") — emergency stop after round=" +
                std::to_string(round_));
       commit_checkpoint();
       break;     // exits run loop, hpo_run() proceeds to clean shutdown
   }
   ```
4. Document the difference vs `STOP_NO_IMPROVE_ROUNDS` in
   `HPO/examples/15_basins_run_and_forget/README.md` — both are
   safety valves; the new one is faster for collapsed-pool problems.

### Test
`HPO/tests/test_basins_pool_collapse_stop.cpp` (Catch2):
- Build a pool with 16 basins all at loss ≈ 100.0 (within 1%).
- Build BasinsSpec with `pool_collapse_rel_spread = 0.05`.
- Set `rounds_no_improve_ = 2`.
- Assert that `BasinsPolicy::should_stop()` (or whatever predicate
  you factor) returns true with the reason "pool collapsed".
- Repeat with one basin at loss=10.0 (spread ~10×), assert
  predicate returns false.

### Performance measurement
Re-run the verify workdir reproducer (full MAXEVAL 25000, full
BUDGET_MAX_EVALS 300) WITHOUT clamps from R2 (so the noise floor
isn't the early-stop driver). Compare against R3:
- total wall time (expect: ≪ R3, since R0/R3 ran for 5 hours and
  this should stop after round 2 if the pool truly collapsed)
- final best loss (expect: same — pool was collapsed, no further
  improvement was available anyway)
- emergency stop reason in journal (expect: "pool collapsed", not
  MAXEVAL_REACHED)

Pass criterion: **R4 wall time on the verify workdir is ≤ 50% of
R3's wall time on the same workdir, AND R4 final best loss is
within 0.1% of R3's**.

---

## R5 — Split BUDGET_MAX_EVALS into GLOBAL_MAX_EVALS and LOCAL_MAX_EVALS_PER_BASIN

### Problem
A single scalar `BUDGET_MAX_EVALS` caps both stages. Evidence: in
`/tmp/hpo_bug_verify` (`BUDGET_MAX_EVALS 300`), every round's GLOBAL
hit `MAXEVAL_REACHED` at 300/300 evals — meaning GLOBAL would have
benefited from more budget. LOCAL, by contrast, was per-basin
convergence-bounded (NLopt's natural FTOL stop fires before the cap).
The user is forced to either over-fund LOCAL or under-fund GLOBAL.

### Required change
1. Deprecate `BUDGET_MAX_EVALS` as the user-facing knob (keep the
   field for back-compat parsing). Add two new fields to `BasinsSpec`:
   - `std::size_t global_max_evals = 5000;` — per-GLOBAL-stage cap.
   - `std::size_t local_max_evals_per_basin = 1000;` — per-basin
     per-tol-rung cap inside a LOCAL stage.
2. Parser keys: `GLOBAL_MAX_EVALS`, `LOCAL_MAX_EVALS_PER_BASIN`.
3. Back-compat: if `BUDGET_MAX_EVALS` is set, use it as the default
   for both new fields (warn at INFO that the key is deprecated).
4. Update `BasinsPolicy::build_stage_spec` to populate the inner
   OptimizerSpec's `MAXEVAL` from the appropriate field per stage.
5. Update `HPO/examples/15_basins_run_and_forget/config.hpo` to use
   the new keys and add a comment explaining the split.

### Test
`HPO/tests/test_basins_eval_budget_split.cpp` (Catch2):
- Parse a config with both `GLOBAL_MAX_EVALS 600` and
  `LOCAL_MAX_EVALS_PER_BASIN 200`. Assert spec fields populate
  correctly.
- Parse a back-compat config with only `BUDGET_MAX_EVALS 300` —
  assert both new fields = 300, INFO warning emitted.
- Parse a config with all three keys — assert the explicit
  GLOBAL/LOCAL win, no warning.

### Performance measurement
Run the verify workdir with `GLOBAL_MAX_EVALS 1500`,
`LOCAL_MAX_EVALS_PER_BASIN 200` (3× more GLOBAL, ⅔ of original
LOCAL). Compare against R4:
- best-loss-after-{500, 1500, 5000, full} evals
- best-loss after round 1 (the round most affected by GLOBAL
  budget)

Pass criterion: **R5 best loss after round 1 strictly improves over
R4's** (more GLOBAL evals = better basins discovered = better
round-1 baseline). If no improvement is observed, GLOBAL was not
actually the bottleneck; document this finding and revert R5 (the
data-driven outcome rules — do not ship a knob for which there is
no evidence of value).

---

## Final report (write to `BASINS_PERFORMANCE_RESULTS.md` next to this ticket)

After all five recommendations are implemented (or rolled back with
documented reason), produce a final report with:

1. **Per-recommendation status**: implemented / reverted / blocked, with
   one-line reason.
2. **Cumulative performance table**, comparing R0 (ticket-start
   baseline) against R5 (all recommendations) on the verify workdir
   reproducer at fixed MAXEVAL=25000:
   - total wall time
   - best loss
   - eval count of best
   - rounds completed
   - count of `nlopt failure` / `LN_SBPLX simplex` / `BAD_POT skipping`
     warnings
3. **Best-loss-vs-eval plot data** (CSV or table) showing R0 and R5
   trajectories so the anytime profile improvement is visible.
4. **Bug encounters log**: any bugs you discovered and fixed during
   this work. Each entry: file:line, symptom, root cause, fix, test
   coverage.
5. **Open questions / follow-ups**: anything the data revealed that
   warrants a separate ticket. Be specific (e.g. "round-2 GLOBAL took
   33 min on the live ITER5 workdir but only 3.6 min on the verify
   workdir at the same BUDGET_MAX_EVALS — investigate why FTOL_REACHED
   fires reliably on the verify objective but not on live ITER5").

The user will read the final report cold; write it to be
self-contained. Use specific numbers from your runs, not "better" or
"faster" — quantify the deltas.

## Working style

You are expected to work autonomously through this ticket. You may:

- Add helper scripts under `HPO/scripts/` for log parsing, metric
  extraction, and CSV emission. Keep them small and testable.
- Refactor BasinsPolicy / BasinPool internals as needed, as long as
  the on-disk checkpoint format stays back-compat for the BASIN_POOL
  and BASINS_POLICY sections (new fields are fine; renamed fields are
  not).
- Skip/defer a recommendation if the data shows it is not
  load-bearing — but document the data and the decision in the final
  report.
- Encounter and fix bugs that block progress — log them in the bug
  encounters section.

You should NOT:

- Change behaviour outside the BASINS code path (no edits to NLopt
  adapter, DLIB adapter, RestartPolicy, MultiStartPolicy, PRESTAGE,
  HPO_Host eval pipeline beyond what R1–R5 require).
- Skip the regression test for a recommendation. If you cannot write
  a test that proves the change works, the change is not done.
- Land a commit that breaks the existing test suite. The whole HPO+CORE
  ctest set must remain green between commits.
- Consult the user mid-ticket. If a fundamental ambiguity blocks
  progress, document it in the final report and continue with whatever
  recommendations are unblocked.

When all five recommendations are processed and the final report is
written, your work on this ticket is complete.
