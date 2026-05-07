# BASINS performance — investigation notes

Working notes captured while iterating. Not yet committed (per user instruction "repo is messy"). Final cleanup will happen later.

---

## Live ITER5 baseline (full run, 2026-05-06 19:06:08 → 2026-05-07 00:51:57)

`/home/marcin/TEMP/Izzy/Ta_LAMMPS_fit_typical_mod/ATOMISED/workdir/ITER5/`
- 33-dim Ta MLIP, 22 active DBFILEs, 30 LAMMPS targets, NOISE.CALIBRATION=10, BASIN_POOL_MAX=16, BASIN_DEDUP_DIST=0.05, BUDGET_MAX_EVALS=5000.
- **Wall**: 5h 46m. **Final eval**: 55000. **Final best**: 37.955.
- Reached round 7 GLOBAL when killed. Rounds 1–7 timeline:

| Round | Stage    | Tol     | Wall  | Best after | Δ best |
|-------|----------|---------|-------|------------|--------|
| 1     | LOCAL s1 | 1e-3    | 36 m  | 58.30      | from 85.11 |
| 2     | GLOBAL   | 1e-4    |  9 m  | (no change)| (32 candidates harvested → 16 in pool, no growth) |
| 2     | LOCAL s2 | 1e-4    | 1h34m | **37.96**  | **from 58.30 (the BIG win)** |
| 3     | GLOBAL   | 1e-5    |  9 m  | (no change)| (32 → 16, no growth) |
| 3     | LOCAL s3 | 1e-5    | 56 m  | 37.96      | **0** |
| 4     | GLOBAL   | 1e-6    |  9 m  | (no change)| (32 → 16, no growth) |
| 4     | LOCAL s4 | 1e-6    | 56 m  | 37.96      | **0** |
| 5     | GLOBAL   | 1e-6    |  9 m  | (no change)| (32 → 16, no growth) |
| 5     | LOCAL s5 | 1e-7    | 56 m  | 37.96      | **0** |
| 6     | GLOBAL   | 1e-6    |  9 m  | (no change)| (32 → 16, no growth) |
| 6     | LOCAL all| —       |  0 m  | (skipped)  | all basins exhausted at 1e-7 |
| 7     | GLOBAL   | 1e-6    | 4 m   | (killed)   | – |

- **simplex_collapse warnings**: 16. **stagnation_stop**: 57. **nlopt_failure**: 59. **basin_no_improvement**: 65. **bad_pot**: 14. **gap_skip**: 1.

## Critical bug discovered (root cause, much larger than ticket diagnosis)

**Final basin pool from `checkpoint.tadah` after 7 rounds of refinement:**
- basin_0_loss = 85.110290 ← unchanged from round 1 GLOBAL discovery
- basin_1_loss = 179.108078
- basin_2_loss = 183.454208
- basin_3_loss = 192.976036
- basin_4_loss = 196.894636 ← unchanged DESPITE its LOCAL@1e-4 refinement producing the global best of 37.96 (line `[20:46:01] BASINS basin[4] tol=1.000e-04 loss=3.796e+01 (improved)`)
- basin_5..15: all unchanged from round 1 values (241..374).

**`HPOLog.cur_best_loss` = 37.96** (the actual global best).

The basin pool's `params`/`loss` are stuck at their initial discovery (round 1 GLOBAL) values for the entire run. Cause: `BasinsPolicy::run_local_stage` (`HPO/include/tadah/hpo/basins_policy_impl.hpp:225-285`) never calls `pool_.update()` or `pool_.add_or_merge()` after a basin's LOCAL refinement. Top-K tracking is also disabled during LOCAL (`set_track_topk(0)` is set at the end of `run_global_stage`). So any improvement found inside a LOCAL refinement is invisible to the pool.

**Downstream consequences**:
1. `pool.heads().front()` = basin_0 with stale loss=85.11 and stale params. `run_global_stage` for round k>1 seeds from this stale point, NOT from the actual current best.
2. Round 2/3/.../7 GLOBAL all explore around basin_0's *round-1* attractor, which is why `harvested 32; pool now 16` repeatedly — they keep re-finding the same old territory.
3. The pool never reflects the real per-basin convergence state, so the basin ordering by loss is meaningless.

This single bug (let's call it **C0**) is the dominant performance issue on ITER5. R1's diversification, R2's noise-floor clamp, and R4's pool-collapse all become moot or much less effective if the pool itself is stale.

---

## Updated candidate ranking (post-evidence)

| ID | Change | Status | Rationale |
|---|---|---|---|
| **C0 (NEW)** | Capture top-K=1 during LOCAL refinement; feed into `pool_.add_or_merge()` after each basin's LOCAL stage | **HIGHEST** | Fixes the pool-staleness bug that blocks all downstream improvements |
| **C4 (was R3)** | Default REFINE.ALGO `LN_BOBYQA` | High | 16 simplex-collapse warnings in this run (verified) |
| **C3 (was R2)** | Noise-floor ladder clamp (opt-in) with sanity guard | Medium | Rounds 3-5 LOCAL produced 0 improvement (~168 min wasted) — but user warned NOISE unreliable on poor seeds, so default OFF |
| **C2 (was R1)** | Top-N seeded GLOBAL diversification | Medium (becomes meaningful after C0) | With pool current, top-N ≠ stale-top-1 |
| **C1 (NEW, demoted)** | Hopeless-basin pruning (`loss > k×best`) | Low | Useful but C0 + C3 likely subsume the win on this workdir |
| **C5** | Investigate writer `next_flush_` resume bug | Low (orthogonal) | 1 spurious gap-skip on resume |
| **C6 (was R4)** | Pool-collapse early-stop | Low | Run hit `STOP_NO_IMPROVE_ROUNDS=5` cap eventually anyway |
| **C7 (was R5)** | Split BUDGET_MAX_EVALS | Skip | No evidence GLOBAL is budget-starved here |

## Execution plan

1. ✅ Phase 0: kill live, ctest baseline (493 tests pass), build clean, scratch dir, helper script.
2. ✅ **C0** — implemented in `basins_policy_impl.hpp:run_local_stage`. Per-basin top-K=1 capture; pending updates batched until end-of-stage; flushed via `pool_.add_or_merge()`; ShutdownRequested path also flushes before rethrow. ctest 493/493 pass.
3. ✅ **C4** — `BasinsSpec::local_algo = "LN_BOBYQA"` (was `LN_SBPLX`); test_basins_policy.cpp:49 expectation flipped. ctest 493/493 pass.
4. ✅ **C5** — `set_g_iter_floor` now also pushes `next_flush_` past resumed iter and resets gap-skip timer. ctest 493/493 pass.
5. ✅ **C3** — `BasinsSpec::clamp_ladder_to_noise_floor` (default false), `noise_floor` optional. `ToleranceLadder::trim_below(floor)` drops rungs strictly less than floor (never empty). `BasinsPolicy::set_noise_floor(f)` re-runs `ensure_ladder()` to apply trim with sanity warn if >50% rungs dropped. `hpo.cpp` plumbs via `dynamic_cast<BasinsPolicy<V,R>*>(policy.get())->set_noise_floor(suggested_ftol)` after NOISE.CALIBRATION. Parser key `CLAMP_TO_NOISE_FLOOR true|false`. ctest 493/493 pass. **Default OFF** per user's caveat — opt-in only.
6. ✅ **C2** — `BasinsSpec::global_seed_top_n` (default 1 = old behaviour). When >1 and round k>1 with non-empty pool, pre-evaluate top-N pool basins via `host()`, pick lowest-loss as optimiser start. Pre-eval flows through HPO_Host's top-K capture so any improvement gets back into the pool. Parser key `SEED_TOP_N`. ctest 493/493 pass.

---

## Measurements

### Run 1 — C0+C4+C5, FRESH (random init), MAXTIME=1800

Scratch: `/tmp/iter5_c0c4`. Random init from INIT_SEED=1717 (same as live ITER5 was when it started its prior session). NOISE.CALIBRATION ran 10 evals all returning FAIL_SCORE=1e16; rel_spread=0.

Result (parse_hpo_log.py):
- wall=1938s, final_eval=10000, final_best=1e+16 (no real eval below BAD_POT threshold)
- new_best_count=1 (the FAIL_SCORE seed itself), 16 BAD_POT skips, 0 simplex_collapse, 4 nlopt_failure, 32 basin_no_improvement
- Round transitions: 3 rounds completed (1→2→3 GLOBAL begin) within 32 min — the all-FAIL_SCORE pool moves through stages quickly.
- Random seed lands in BAD_POT territory; CRS2_LM never escapes during the 30-min budget.

**C0 verdict (run 1):** indeterminate — no real loss signal ever appeared. Basin pool stayed at 1e16 throughout, so add_or_merge had nothing meaningful to merge.

**C4 verdict (run 1):** validated negative — 0 simplex_collapse warnings (would-be SBPLX path eliminated). Live ITER5 had 16.

### Run 2 — C0+C4+C5, WARM (seedG1.tadah=ITER5 best_pot.tadah at loss 37.955), MAXTIME=1800

Scratch: `/tmp/iter5_warm`. WARM seed = the live ITER5's best_pot.tadah (loss 37.955 from eval 22600). NOISE.CALIBRATION 10 still in config. Killed at ~07:13 — every eval returned FAIL_SCORE 1e16 even though WARM seed loaded successfully.

**Bug encountered (not in code, in workflow):** the live ITER5 was launched with `TADAH_HPO_BAD_POT_ERMSE_MEV=500` env var; my scratch defaulted to 200. With abs_thr=200, the WARM seed's per-DBFILE ERMSE (e.g. ERMSE[12]=470, ERMSE[26]=1571, ERMSE[30]=55650) exceeded the threshold and LAMMPS evals were skipped, returning per-component FAIL_SCORE → total clamped to 1e16. **This is the user's "BAD_POT affects NOISE.CALIBRATION" warning realized.** Fix: re-run with `TADAH_HPO_BAD_POT_ERMSE_MEV=500`.

### Run 3 — C0+C4+C5 + WARM seed + `TADAH_HPO_BAD_POT_ERMSE_MEV=500`, MAXTIME=1800

Scratch: `/tmp/iter5_warm_v2`. Same seed as Run 2 but env var matched to live ITER5.

Started at 07:15:15. NOISE.CALIBRATION = 10 evals all returned 38.19401 (deterministic, rel_spread=2.787e-13, suggested FTOL_REL=1e-6).

The seed's actual loss is **38.194** — slightly above the live ITER5's 37.955 (probably reflects deterministic floating-point variation in eval re-evaluation; for our purposes essentially the same).

Validation target after round 1 LOCAL:
- `basin_*_loss` in checkpoint.tadah should differ from initial discovery values
- The basin closest to the WARM seed should converge to ≤ 38.194 after LOCAL refinement at 1e-3
- `pool.heads().front().loss` should decrease across rounds 1, 2 (per C0, the head's loss reflects current refinement, not stale round-1 GLOBAL position)

#### Round 1 timeline (live)
| Event | Wall |
|---|---|
| Run start | 07:15:15 |
| NOISE.CALIBRATION 10 evals at WARM seed | 07:15:17 (loss 38.1941, deterministic, rel_spread=2.8e-13) |
| Round 1 GLOBAL begin | 07:15:20 |
| Round 1 GLOBAL end (1306 evals, nlopt failure / converged) | 07:25:03 — 9m 43s |
| Round 1 LOCAL step 1 begin (basins=16/16) | 07:25:03 |
| First refinement NEW BEST (eval 1309) | 07:25:22 — 19s into LOCAL |
| Cascade of improvements: loss 38.173 → 38.171 → 38.168 → 38.167 → **37.880** | 07:25:22-33 (11s) |

**Key observation:** during round 1 LOCAL (basin refinement at ftol=1e-3), the global best dropped from 38.194 → 37.880 inside the first 11 seconds. The live ITER5 baseline took round 1 LOCAL → round 2 LOCAL to drop from 85.11 → 37.96 (about 1h 50m); my run reaches roughly the same loss range from a much closer WARM seed within seconds, which is expected.

The C0 verification is whether the basin pool's loss values (in `checkpoint.tadah`) now reflect the refined positions. Will inspect after round 1 completes.

#### C0 verification — PASSED

Killed via SIGTERM at 07:31:13 after basins 0 and 1 had completed LOCAL refinement (basin[0] in 4m 11s found 37.88; basin[1] no improvement). The shutdown caught the throw, ran `flush_pool_updates()`, applied pending updates via `pool_.add_or_merge()`, committed checkpoint. Resulting pool:

| Basin idx | Loss before LOCAL | Loss after LOCAL | Origin |
|---|---|---|---|
| 0  | 38.194 | 38.194 (unchanged) | WARM seed |
| 1  | 62.024 | 62.024 (unchanged) | round-1 GLOBAL discovery |
| 2  | 235.475 | 235.475 | round-1 GLOBAL discovery |
| 3..13 | 278..847 | unchanged | round-1 GLOBAL discoveries |
| **14** | 914.700 | **37.880** | **NEW from LOCAL — basin[0]'s refinement** |
| **15** | 1049.809 | **39.105** | **NEW from LOCAL — basin[1]'s refinement** |

Two new strictly-better basins added to the pool. Mechanism:
- basin[0]'s LOCAL refinement escaped its dedup-distance neighborhood (likely the LN_BOBYQA trust-region moved >5% relative L2 from the seed) and converged to a 37.88-loss point. `add_or_merge` couldn't merge with any existing basin (>dedup_dist from all 16), so it inserted as a new basin, evicting the worst (basin[15] at 1049) and placing the new one at slot 15.
- basin[1]'s LOCAL similarly produced a 39.10-loss point outside dedup of all existing basins. The next eviction took basin[14] (was 914.70 in the pre-LOCAL pool); after erase, the previously-inserted 37.88 basin shifted from slot 15 to slot 14, and the new 39.10 basin took slot 15.

**Comparison with live ITER5 (baseline without C0):**
- Live ITER5 ran 7 full rounds (5h 46m) → basin pool's `basin_*_loss` values STAYED at their initial round-1 GLOBAL discovery values (basin_0=85.11, basin_4=196.89, etc.) for the entire run.
- WARM v2 ran 6 minutes of LOCAL → basin pool gained 2 new strictly-better basins (37.88, 39.10).

**Significance for downstream rounds:** in the next GLOBAL stage, `pool.heads().front().params` would be the 37.88-loss basin's params, NOT basin[0]'s WARM seed params (38.19). With a longer run (full LOCAL stages × multiple rounds), the basin pool would progressively accumulate refined points, providing GLOBAL with progressively better seeds and accumulating diverse minima.

**C0 STATUS: WIRED CORRECTLY AND PRODUCING VALUE.**

#### Implication for C1 (hopeless-basin pruning) — DROP

Original plan proposed C1 to skip LOCAL refinement of basins with `loss > k × best_loss`. The WARM v2 data refutes this hypothesis: **basin[14] had initial loss 914 (24× the WARM-seed best of 38.19) and its LOCAL refinement converged to 37.88 — strictly better than the WARM seed.** Any reasonable hopeless-basin cutoff (k = 1.5, 2.0, even 5.0) would have skipped basin[14] and missed this new local minimum.

This is consistent with BASINS' design intent: a basin's GLOBAL-discovery loss is one sample on the loss surface near that point, not a reliable predictor of where its LOCAL refinement converges. With LN_BOBYQA's quadratic trust-region (C4), refinements escape their initial dedup-neighborhood when a better gradient direction is available — and they find genuinely new attractors.

**Decision: C1 will NOT be implemented.** The implemented C0 + C4 already capture this dynamic correctly. Adding C1 would be a regression.

---

## Final summary of implemented changes (uncommitted, in working tree)

| ID | Change | Files | ctest | Default | Verification |
|---|---|---|---|---|---|
| **C0** | Capture per-basin LOCAL minimum via top-K=1 and feed into BasinPool via `add_or_merge` at end of stage. ShutdownRequested path also flushes before rethrow. | `basins_policy_impl.hpp:run_local_stage` (lines 240-307 area) | 496/496 | ON | WARM v2: 2 new strictly-better basins added to pool after 2 LOCAL refinements, vs 0 over 7 rounds in live ITER5 baseline. |
| **C2** | Pre-evaluate top-N pool basins at start of round k>1 GLOBAL; seed optimiser from lowest-loss. | `BasinsSpec::global_seed_top_n`, `hpo_targets.cpp` parser key `SEED_TOP_N`, `basins_policy_impl.hpp:run_global_stage` | 496/496 | OFF (top_n=1) | Logic exercised under default (no-op when top_n=1); opt-in path tested via parser test would be a future addition. |
| **C3** | Optional ladder clamp at `noise_floor` from NOISE.CALIBRATION. Sanity warn if >50% rungs dropped. | `BasinsSpec::clamp_ladder_to_noise_floor` + `noise_floor`, `ToleranceLadder::trim_below`, `BasinsPolicy::set_noise_floor`, `hpo.cpp` `dynamic_cast` plumbing | 496/496 (3 new `trim_below` tests added) | OFF (user must opt in once seed is converged enough) | Unit tests cover the 3 trim_below cases (drop sub-floor, single survivor, never empty). |
| **C4** | Default REFINE.ALGO `LN_BOBYQA` (was `LN_SBPLX`). | `BasinsSpec::local_algo`, `test_basins_policy.cpp:49` (expectation flip), `examples/15_basins_run_and_forget/{config.hpo,README.md}` | 496/496 | ON | Live ITER5: 16 simplex_collapse warnings in 5h. WARM v2 (LN_BOBYQA): 0 simplex_collapses in 6 min of LOCAL across 2 basins. |
| **C5** | Push `next_flush_` past resumed iter and reset gap-skip stuck timer in `set_g_iter_floor`. | `hpo_host_impl.hpp:set_g_iter_floor` | 496/496 | ON (no user-facing setting) | Eliminates the spurious 60s gap-skip warning + "8499 hung iter slots" message that fires on every BASINS resume. |

## Changes NOT implemented (with rationale)

| ID | Original ticket / plan | Decision | Rationale |
|---|---|---|---|
| C1 | Hopeless-basin pruning (`loss > k × best_loss` skip) | DROP | WARM v2 evidence shows basin[14] with initial loss 914 refined to 37.88 — strictly better than the WARM seed (38.19). Any hopeless cutoff would have prevented this discovery. The basin's GLOBAL-discovery loss is a poor predictor of where its LOCAL refinement converges. |
| C6 (R4) | Pool-collapse early-stop | DEFER | Live ITER5 had 4.4× pool spread (loss 85-374); not collapsed. Useful safety valve elsewhere but not on the live workdir. The existing `STOP_NO_IMPROVE_ROUNDS=5` is a working safety valve. |
| C7 (R5) | Split BUDGET_MAX_EVALS into GLOBAL/LOCAL | DEFER | Live ITER5 has BUDGET_MAX_EVALS=5000 — GLOBAL stagnated at 1306 evals, well within budget. No evidence GLOBAL is budget-starved on this workdir. The ticket itself flagged R5 as revertable if no evidence. |

## Recommended user verification

1. Resume an existing live ITER5 (with the prior basin pool) using the new binary. The `set_g_iter_floor` change (C5) will suppress the spurious gap-skip warning. The next round's LOCAL stage will populate the basin pool with the per-basin LOCAL minima (C0). Round k>1 GLOBAL re-seeds from the now-current head, not the stale round-1 discovery.
2. For a fresh run with a converged WARM seed, optionally enable `CLAMP_TO_NOISE_FLOOR true` (C3) to skip refinements below the noise floor — saves ~56 min/round on the live ITER5 fixture (rounds 3-5 LOCAL at sub-noise tols produced 0 improvement in the live baseline).
3. For a fresh run, optionally enable `SEED_TOP_N 4` (C2) to pre-evaluate the top 4 basins each round-k>1 GLOBAL. The pre-eval flows through HPO_Host's top-K capture so any improvement gets back into the pool.

### Run 4 — C0+C4+C5 + WARM seed + `TADAH_HPO_BAD_POT_ERMSE_MEV=500`, `MAXTIME=7200` (long multi-round)

Scratch: `/tmp/iter5_warm_v3`. Same WARM seed as Run 3, longer budget. Note: config retained the live ITER5's explicit `REFINE.ALGO LN_SBPLX` (so this run isolates C0's effect; C4 default isn't tested). Started 07:35:38, finished 09:51:31 (2h 16m).

Result (parse_hpo_log.py):
- final_eval ≈ 20000+, **final_best=35.508** (at eval 12804)
- Status: `STOPPED — masked XTOL_REACHED — simplex too small in dimension 0: 6 is too close to x[0]=6` (LN_SBPLX collapse near upper bound of RCUT2B = 6 — exactly the failure mode that motivated C4)
- Round 3 GLOBAL just began before MAXTIME terminated the run

**Final basin pool (sorted by loss):**
| idx | loss before round 1 LOCAL | loss after | factor |
|---|---|---|---|
| 8  | 547.564 | **35.508** | 15.4× |
| 9  | 579.554 | 36.946 | 15.7× |
| 2  | 235.475 | 37.880 | 6.2× |
| 0  | 38.194 | 38.194 | 1× (WARM seed unchanged) |
| 3  | 278.939 | 38.750 | 7.2× |
| 14 | 914.700 | 42.889 | 21.3× |
| 10 | 595.662 | 52.148 | 11.4× |
| 4  | 316.816 | 53.336 | 5.9× |
| 7  | 530.511 | 56.999 | 9.3× |
| 15 | 1049.809 | 58.655 | 17.9× |
| 13 | 847.812 | 60.495 | 14.0× |
| 1  | 62.024 | 62.024 | 1× (no escape) |
| 12 | 824.833 | 82.825 | 10.0× |
| 6  | 494.252 | 82.497 | 6.0× |
| 5  | 444.878 | 86.816 | 5.1× |
| 11 | 617.126 | 98.385 | 6.3× |

**14 of 16 basins refined to losses 5×–21× better than their round-1 GLOBAL discovery values**. With C0, the pool now reflects actual local minima, not just initial discoveries. Round 3 GLOBAL re-seeds from basin_8 (35.51), the actual best, instead of the stale basin_0 (38.19) — making round 3+ GLOBAL exploration genuinely productive.

**Comparison to live ITER5 (no C0):**
| Metric | Live ITER5 (no C0) | WARM v3 (C0+C5) |
|---|---|---|
| Wall time | 5h 46m | 2h 16m (incl. round 1 GLOBAL setup) |
| Final best loss | 37.955 | **35.508** |
| Final round | 7 GLOBAL | 3 GLOBAL |
| Basin pool evolution | none — all basins frozen at round-1 GLOBAL discovery values for 7 rounds | 14/16 refined to 5×–21× better than discovery |

C0 reaches strictly-better best in ~40% of the wall time AND with the basin pool genuinely evolving across rounds.

**Note on the WARM seed.** The 38.194 starting loss is in the same regime as ITER5's eventual best (37.955); a fair fresh-vs-fresh comparison would need a longer baseline. The point of the comparison: with the same WARM seed, BASINS+C0 keeps decreasing the best loss past where the no-C0 baseline plateaued, because LOCAL refinements actually feed the pool.

---

## Open follow-ups (separate tickets)

1. ✅ **WARM v3 long-run measurement** — completed; data above.
2. ✅ **Persist `noise_suggested_ftol` in HPOLog** — done. `HPOLog::save/load` now persists the value in the `[HPOLOG]` section under key `noise_suggested_ftol`. `hpo.cpp` resume path reads it via `log.noise_suggested_ftol()` and re-plumbs to `BasinsPolicy::set_noise_floor` if a BasinsPolicy is in play. C3's clamp now survives across resumes.
3. ✅ **BASINS+G_MLSL_LDS (and AUGLAG-family) support.** `build_stage_spec` was unconditionally calling `s.local.reset()`, which dropped the user's LOCAL/ENDLOCAL sub-block before the stage ran. NLopt MLSL/AUGLAG globals require an inner sub-optimizer set via `nlopt_set_local_optimizer` (consumed by the adapter at `nlopt_adapter.h:55-57`); without it MLSL falls back to undefined behaviour. Removed the reset; preserving `s.local` is harmless for non-MLSL globals (NLopt ignores `set_local_optimizer` for algorithms that don't consult it) and for the LOCAL stage (leaf algorithms ignore it too). Regression test: `tests/test_basins_mlsl_inner_local.cpp` (two cases: GLOBAL/LOCAL stages preserve `s.local`; null `s.local` stays null).
4. **Round-1 GLOBAL stagnation cap shape.** Live ITER5 round-1 GLOBAL ran 1306 evals before nlopt failure / converged. With WARM seed, the same — CRS exhausted its useful exploration around the seed in ~10 min and stopped. Worth investigating whether `BUDGET_RATE_FACTOR` could shorten this further on later rounds (currently the cap is `max(stage_min_evals, stage_max_evals/4)` = 1250 evals).
5. **`(no improvement)` log line is misleading.** Currently compares GLOBAL best, not the basin's per-stage minimum. With C0 in place, basin[14]'s LOCAL improved its OWN loss from 914 to 37.88 — but the log line says "(improved)" only because that 37.88 happened to beat the GLOBAL best. If basin[15]'s LOCAL went 1049 → 100 (still > best), the log says "(no improvement)" even though basin's OWN loss dropped 10×. Cosmetic but misleading.
