# BASINSv2 — priority-driven, lifecycle-aware HPO strategy

BASINSv2 is an additive `STRATEGY { TYPE BASINSV2 }` for the TADAH HPO
framework. It coexists with `TYPE BASINS` (v1) and never alters v1's
behaviour. This file documents the implementation as it lands in the
`HPO/` module.

## What changed vs v1

v1 sweeps every basin through every ftol rung in lockstep, refines all
basins equally each round, and silently evicts duplicates. v2 picks
**one work item per scheduler iteration** based on a priority score,
runs it, updates lifecycle state, and loops. Each basin has its own tol
cursor; basins are not forced through the ladder together.

| Capability | v1 | v2 |
|---|---|---|
| Scheduling | round-based GLOBAL→LOCAL sweep | priority-driven per-iteration |
| Per-basin lifecycle | absent (in-pool / evicted) | ACTIVE / PROMISING / INACTIVE / SUSPECT_DUP / MERGED / KILLED |
| REFINE termination | fixed `BUDGET_RATE_FACTOR` | tol- and dim-derived adaptive plateau |
| Duplicate handling | silent eviction | explicit merge with sibling_root_id, two-step (suspect → confirm) |
| EXPLORE | one CRS call per round | Sobol K=max(8,2·dim) + path-aware filter + brief local refinement |
| Path memory | none | per-basin coarse trajectory, capped at `path_max_points`, smart pruning |
| Verify-before-kill | absent | one tickle refinement per inactive basin, capped at `verify_evals_cap` |
| Stop condition | `STOP_NO_IMPROVE_ROUNDS` | `STOP_NO_IMPROVE_ITERATIONS` (iteration-based) |
| Checkpoint section | `BASINS_POLICY` / `BASIN_POOL` | `BASINS_POLICY_V2` / `BASIN_POOL_V2` (cross-load rejected) |

## Minimal config

```
OPTIMIZER
  STRATEGY
    TYPE BASINSV2
  ENDSTRATEGY
ENDOPTIMIZER
```

Defaults: EXPLORE = NLOPT GN_CRS2_LM (used as the local-refinement algo on
Sobol survivors), REFINE = NLOPT LN_BOBYQA, pool 24, dedup 0.05, merge 0.10,
verify cadence 30 iter, summary every 25 iter, plateau eps factor 0.5.

## Expert keys (all inside `STRATEGY { TYPE BASINSV2 ... }`)

```
BASIN_POOL_MAX                 24
BASIN_DEDUP_DIST              0.05
BASIN_MERGE_DIST              0.10
BASIN_MERGE_LOSS_THRESHOLD    0.05
ARCHIVE_MAX                     48      # cap on MERGED+KILLED retention
INACTIVE_AFTER_EVALS             0      # 0 = derive from refine_budget_max_evals * 0.75
VERIFY_CADENCE                  30
VERIFY_MAX_ATTEMPTS              2
VERIFY_EVALS_CAP                 0      # 0 = derive (10 * dim)
MERGE_CADENCE                   25
EXPLORE_KICK_CADENCE            50      # forces fresh-region EXPLORE
EXPLORE_DIST_FLOOR             0.05
SUMMARY_CADENCE                 25
STARVATION_CAP                 200      # force-pick if a basin is neglected this long
STOP_NO_IMPROVE_ITERATIONS     200
PLATEAU_EPS_FACTOR             0.5      # relative improvement threshold = factor * tol
PATH_MAX_POINTS                 16      # 0 disables path recording
PATH_LOSS_EPS                  0.01
PATH_DIST_EPS                  0.01
PRIORITY_WEIGHTS               2.0 0.5 0.3 0.5 1.0 0.05
                               # improve neglect distance tol_gap state age
VERBOSE_DIAGNOSTICS            false    # print priority-dump every 50 iter
CHECKPOINT_EVERY               60s      # or "100" for evals
```

`EXPLORE { ... }` and `REFINE { ... }` blocks accept the same v1 keys
(`LIB`, `ALGO`, `FTOL_LADDER`, `BUDGET_*`).

## Scheduler

Each iteration `select_work()` picks ONE of:

- **EXPLORE** — bootstrap (empty pool), explore-kick cadence, or no
  basin scores high enough.
- **REFINE(basin, tol)** — argmax of priority score over ACTIVE-family
  basins; uses the basin's own tol cursor.
- **VERIFY_KILL(basin)** — INACTIVE basin past `verify_cadence` and
  still has verify attempts left. One capped refinement; revives the
  basin to ACTIVE on improvement, otherwise increments
  `verify_attempts`.
- **MERGE_PASS** — every `merge_cadence` iters, no LAMMPS evals. Finds
  candidate pairs and either flags `SUSPECT_DUP` (first sighting) or
  confirms the merge (second sighting still close).

## Priority formula

```
priority(b) = W_improve  * norm(b.improvement_ema)
            + W_neglect  * norm(iter - b.last_iteration)
            + W_distance * norm(min_dist_to_other_active(b))
            - W_tol_gap  * norm(log10(b.current_tol_cursor) - log10(target_tol))
            + W_state    * state_bonus(b.state)
            - W_age      * norm(iter - b.birth_iteration)
```

`norm(x)` is min-max normalisation across the active pool. State bonuses:
`PROMISING:+1.0, ACTIVE:0.0, INACTIVE:-0.5, SUSPECT_DUP:-2.0,
MERGED/KILLED:-inf`. Starvation backstop: if a basin's
`iter - last_iteration > STARVATION_CAP`, force-pick it once.

## Adaptive REFINE termination

```
threshold = clamp( max(REFINE.BUDGET_MIN_EVALS,
                        100 * dim,
                        50 * |log10(tol)| * PLATEAU_EPS_FACTOR),
                   0,
                   REFINE.BUDGET_MAX_EVALS )
```

The threshold is passed to `HPO_Host::set_stagnation_threshold()`; the
inner optimiser bails when no global best is found within that many
evals. `100 * dim` is the safety floor for slow exponential descent.

## EXPLORE pipeline

1. Sobol points: K = max(8, 2·dim); seed_skip stamped from iteration.
2. Map to real space (respects `log_scaled` mask), evaluate via host.
3. Filter: drop candidates within `EXPLORE_DIST_FLOOR` of any known
   basin or any of its coarse path points.
4. Top M = max(2, K/4) survivors by loss.
5. Brief local refinement (cap `30 * dim` evals) using EXPLORE block's
   LIB/ALGO from each survivor; endpoint goes through `add_or_merge`.

## Path recording (Phase 4)

`PATH_MAX_POINTS > 0` enables per-basin coarse trajectory recording.
Snapshots fire when relative loss improves by ≥ `PATH_LOSS_EPS` OR
parameters move by ≥ `PATH_DIST_EPS`. Pruning rule: drop the
intermediate point whose loss is closest to the mean of its neighbours
(endpoints sacrosanct). Paths are consulted by EXPLORE's filter and by
MERGE_PASS's `path_overlap` predicate.

## Diagnostics

```
[HPO] BASINSv2 iter=137 select=REFINE basin[id=12] tol=1.0e-05 ema=4.2e-04 stagn_thr=300
[HPO] BASINSv2 basin[id=12] state=ACTIVE->PROMISING ema=4.2e-04 (p75=2.1e-04)
[HPO] BASINSv2 basin[id=07] state=ACTIVE->INACTIVE evals_since_improve=820 (cap=750)
[HPO] BASINSv2 verify basin[id=07] tickle: best 3.4521e-02 -> 3.4521e-02 (no progress) attempt=1/2
[HPO] BASINSv2 merge-pass: 28 active, found 2 candidate-pair(s)
[HPO] BASINSv2 merge basin[id=18] -> basin[id=03] (root) dist=0.073 dloss=1.2%
[HPO] BASINSv2 EXPLORE Sobol K=16 evaluated, M=4 survived (filter dropped 12 in known regions)
[HPO] BASINSv2 round-summary iter=200 active=24 promising=3 inactive=4 suspect=0 merged=8 killed=2 best=3.4109e-02
```

`VERBOSE_DIAGNOSTICS true` prints a full priority-sorted basin table
every 50 iterations.

## Checkpoint

Sections: `BASINS_POLICY_V2` (scheduler state, priority weights, wall
time accumulated) + `BASIN_POOL_V2` (per-basin params, loss, lifecycle,
EMA, tol cursor, verify counters, coarse path). v1 checkpoints are
explicitly NOT readable by v2 — start fresh.

## Implementation map

- `include/tadah/hpo/basin_pool_v2.h` + `src/basin_pool_v2.cpp` —
  `BasinV2`, `BasinPoolV2` (lifecycle, dedup, merge primitives, archive).
- `include/tadah/hpo/basins_v2_priority.h` + `src/basins_v2_priority.cpp`
  — pure priority scoring + adaptive REFINE threshold formula.
- `include/tadah/hpo/basins_v2_explore.h` + `src/basins_v2_explore.cpp`
  — Sobol generator + path-aware filter + top-M.
- `include/tadah/hpo/basins_v2_path.h` + `src/basins_v2_path.cpp` —
  prune_path, should_snapshot, path_overlap.
- `include/tadah/hpo/basins_v2_policy.h` +
  `include/tadah/hpo/basins_v2_policy_impl.hpp` — the templated policy.
- Wired into `optimizer_spec.h`, `hpo_targets.cpp`, `run_policy.h`,
  `hpo.cpp` alongside (not replacing) v1.

## Tests

- `tests/test_basin_pool_v2.cpp` — pool primitives, lifecycle,
  archive eviction, checkpoint round-trip.
- `tests/test_basins_v2_priority.cpp` — priority formula edge cases.
- `tests/test_basins_v2_plateau.cpp` — adaptive REFINE threshold.
- `tests/test_basins_v2_parser.cpp` — config parsing (defaults +
  overrides + cross-key rejection).
- `tests/test_basins_v2_merge.cpp` — find_merge_candidates,
  flag_suspect_pair, confirm_merge, revert_suspect_pair.
- `tests/test_basins_v2_explore.cpp` — Sobol diversity, filter, top-M.
- `tests/test_basins_v2_path.cpp` — prune_path, should_snapshot,
  path_overlap, end-to-end via pool.
