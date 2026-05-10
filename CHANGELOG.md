# Changelog

All notable changes to **Tadah.HPO** (hyper-parameter optimisation) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **`PC_LAMMPS --hotvar <name> <min> <max> [<initial>]`** — declares a script as *hot*. Hot scripts run alone in **phase A** of each eval (in parallel with other hot scripts), expose a value via `variable <name> equal …`, and have it validated against `[min, max]`. Validated values are cached per `(script, name)` and propagated as `${<name>}` strings to every cold (no-`--hotvar`) PC_LAMMPS script in **phase B**. Multiple `--hotvar` flags per line allowed; no cross-references (phase A is parallel). An out-of-range hot value fail-scores the iter and skips phase B.
- **`PC_<METRIC> --skip-above <cap>` / `--skip-rel <ratio>`** — per-metric bad-pot pre-filter on PC_EMAE, PC_ERMSE, PC_ErRMSE, PC_ERSQ, PC_FRMSE, PC_SRMSE. Triggers fire BEFORE the LAMMPS phase, fail-scoring the iter when the validation metric exceeds the cap or the metric > ratio × running best for that metric. Replaces the legacy ERMSE-only env vars (see Removed).

### Changed
- `run_lammps` is now two-phase (hot first, cold second). When no `--hotvar` is declared the behaviour is identical to before: phase A is empty, all scripts run in parallel in phase B. The previous "per-script `${a0}` post-hoc range check" is replaced by the explicit `--hotvar` declaration; the cache is now keyed on `(script, varname)` instead of just `script`.
- `best_pot.tadah` warmstart cache lines: `# HOT_A0 <script> <a0>` → `# HOT <script> <name> <value>`. The legacy format still parses on read for back-compat with older best_pot files.

### Removed
- `TADAH_HPO_HOT_A0_DEFAULT`, `TADAH_HPO_A0_MIN`, `TADAH_HPO_A0_MAX`, `TADAH_HPO_BAD_POT_ERMSE_MEV`, `TADAH_HPO_BAD_POT_ERMSE_REL` environment variables. All four knobs now live in `config.hpo` as PC_LAMMPS / PC_<METRIC> flags. Migration: replace `${hot_a0}` script convention with an explicit `--hotvar hot_a0 <min> <max> [<initial>]` on the producing PC_LAMMPS line; replace `TADAH_HPO_BAD_POT_ERMSE_MEV=N` with `PC_ERMSE … --skip-above N`.

### Fixed
- BASINS `STAGE_REFINE` step previously crashed when `LIB DLIB ALGO BFGS` was used because the BasinsPolicy ladder unconditionally set `ftol_rel`, which DLIB:BFGS does not support. The ladder now writes `ftol_rel` for NLopt and `ftol_abs` for DLIB:BFGS via a new `OptimizerCaps` registry; adding new optimisers is now two changes (run-function + caps entry).
- `StagnationStop` recovery in `HPO_Host::operator()` no longer overwrites the eval's real loss with `FAIL_SCORE` and no longer flags the result as `was_nan_clamped` — both were corrupting `loss.tadah` for evals that successfully completed before the threshold check. The per-eval flood (one warning per concurrent OMP eval that crossed the threshold) is throttled to one diagnostic per stagnation event via a flag cleared by `reset_stagnation()`.
- BASINS `BUDGET_RATE_FACTOR` was parsed but unused; the per-stage stagnation threshold was hardcoded to `MAX_EVALS / 4` (EXPLORE) and `/8` (REFINE). The factor now drives the threshold as documented: `max(BUDGET_MIN_EVALS, BUDGET_MAX_EVALS * BUDGET_RATE_FACTOR)`.

- `LIB TADAH ALGO ANNEAL` — classic Kirkpatrick/Metropolis Simulated Annealing driver (refs: Kirkpatrick 1983; Metropolis 1953). Geometric cooling, fixed-shape Gaussian step in transformed-parameter space, Metropolis acceptance, auto-T0 calibration from a small uniform-bounds random walk (overridable via `PARAM T_INIT`). Tunable knobs: `T_MIN`, `RT`, `L`, `STEP_FRAC`, `T0_PROBES`, `T0_PACCEPT`. Honours `MAXEVAL`/`MAXTIME`/`STOPVAL`.
- **`ALGO ANNEAL` Corana adaptive variant** — opt-in via `PARAM CORANA 1` (refs: Corana, Marchesi, Martini, Ridella, *ACM Trans. Math. Soft.* 13:262, 1987; Brommer, Univ. Stuttgart MSc, 2003). Per-dim step vector `V[i]` adapts every `NS`-step cycle from the per-dim accept ratio (grows when rate > 0.6, shrinks when < 0.4); cools after every `NT` step-adjust cycles. Replaces classic's whole-vector Gaussian + fixed σ with one-dim-at-a-time uniform perturbation + adaptive σ — fixes the cold-T freeze that fixed-σ runs hit on continuous multimodal problems. Tunable knobs: `NS` (default 5), `NT` (default 2), `C` (default 2.0). Per-rung journal line shows `V=[V_min, V_max]` so users can watch step-size adaptation. Banner shows `ANNEAL[CORANA]` vs `ANNEAL[CLASSIC]`.
- **`PARAM V_FLOOR_FRAC`** (default `1e-3`) — explicit lower clamp on `V[i]` as a fraction of bound width. Guards Corana against `V[i]` runaway collapse on `FAILSCORE`-cliff landscapes where every uphill ΔL ~ FAILSCORE rejects for any T, driving accept rate to 0 and `V[i]` → 0 by factor `(1+C)^NT` per rung.
- **`PARAM RESET_TO_BEST 1`** — Brommer (2003) modification: snap the chain back to the best-so-far point at the end of every cooling step. Gives Corana basin memory; addresses the high-T wander failure mode where the chain finds a good point at high T then loses it as it drifts. Adds zero host evals (just bookkeeping).
- `ANNEAL` journal trace — `hpo_run.log` records: a banner echoing the resolved knobs (incl. `sigma=<s>` when `STEP` overrides `STEP_FRAC`), an extended T0 calibration line with `mean|dE|`, `loss range=[min, max]`, `std`, one summary line per cooling rung (`T`, `proposed`, `accepted` (%), `oob` (%), `f_x`, `best`), an end-of-run summary block with totals and accept-rate `min`/`max`, and a single explicit `ANNEAL stopped: <reason> after N host eval(s)` line. Lets users diagnose mistuned `T_INIT`/`RT`/`STEP_FRAC` from the journal alone.

### Fixed
- `ALGO ANNEAL` whole-proposal rejection in high dimensions — replace "any-dim-OOB → reject" with per-dim mirror reflection. Old code rejected ~98% of proposals on a 38-dim Ta workload, evaluating only ~1 of every 50 L-steps; reflection lets every L-step do real work, recovering full chain mobility.

## [1.3.0-beta.1] — TBD

HPO underwent the largest restructure of any module in this release: new init strategies, basins-policy machinery, resume/restart, varloss, dlib-BFGS local optimiser, and a regression-tested 15-example suite.

### Added
- **Init strategies** — multiple ways to seed the HPO sweep (`SOBOL`, `LHC`, `GRID`, `GRID_HIGHEST`, `RANDOM`, `LIST`, `EXACT`); fully documented in the rewritten `hpo.rst`.
- **BASINS run-policy** — basin-hopping mode with `REFINE` follow-up (default for refining promising candidates). Configurable `mode`, `outliers`, `direction`, `dist_metric`, `frac_top`, `frac_pca`, `frac_isolated`.
- **Resume / restart** — checkpointed long runs can be resumed from the most recent journal entry; backed by the new CORE checkpoint + signal-handler infrastructure.
- **Variance-of-loss (`varloss`) acquisition** — `--varloss` / `--outvar` flags select an alternative scoring function over a user-provided set of LAMMPS `equal`-style script variables; HPO now validates these variables exist before launching workers.
- Standalone **dlib BFGS local optimiser** — added alongside the existing nlopt path.
- `HPOLog` run journal — structured per-evaluation log written incrementally.
- Per-dataset validation metrics and stats output files.
- Per-thread `Trainer` / `TrainerHost` cache — when structural parameters do not change between evaluations, the cached object is reused via `Trainer::reinit()` instead of being reconstructed (large speed-up for sweeps that vary only weights or scalar regression parameters).
- **Reference example suite** (`HPO/examples/`, 15 examples) with `run_reference_suite.sh` driver and per-example fingerprint files (`reference_fingerprint.json`) for regression testing. Full suite runs in ~52 s; `--fast` subset (14 examples) in ~5 s.
- `TADAH_ENABLE_ASAN` CMake option.
- **`TADAH:ANNEAL` Simulated Annealing driver** — classic Kirkpatrick/Metropolis SA in the in-tree TADAH library, dispatched from `run_inner()` alongside `TADAH:RANDOM` / `TADAH:GRID`. Operates in transformed (per-`OPTIM` log/identity) space; reflects out-of-bound proposal components back into the box (high-dim robustness fix).
- **Pre-flight audit hook + per-iter health monitor** — wires the MLIP bound-validation pipeline into the HPO surface. Driven by blueprint keys `AUDIT`, `AUDIT_SAMPLE`, `HEALTH_LOG` (in CORE).
- **`EFILTER` / `FFILTER` forwarded from `--validation`** — the validation-side LOAD-TRANSFORMS path now respects the same energy/force filtering as training. Symmetric LOAD-TRANSFORMS log; mean-share breakdown.

### Changed
- **Major restructure** — init strategies, basins policy, resume, varloss reorganised into clearly separated layers; `+21 561` lines net.
- `structural_params_changed` flag retired — the trainer now decides via `Trainer::reinit()` semantics whether a rebuild is needed.
- Best-pot writer rewritten (now v2 layout); flushes outputs on short runs even before the run ends.
- Include-file parsing is now case-insensitive.
- GitLab CI rewritten from scratch — replaces the auto-generated Auto-DevOps configuration with a hand-written pipeline (build, test, lint, docs link to umbrella, release).

### Fixed
- `rcut` optimisation path corrected.
- Linking errors when HPO was built with optional features enabled (multiple commits).
- MPI rank-handle leaks resolved (paired with the MLIP `MPI_Type_free` fix).
- `LammpsRunner` API alignment (after a brief revert/redo cycle).
- Hardcoded `/home/marcin` paths removed from `examples/common/run.sh` (`tadah` is now resolved on `PATH` by default).

### Internal
- 154 per-run output artefacts un-tracked from `examples/` (stay reproducible via the driver, no longer cluttering `git status`).
- `tadah_use_local_module` integration — under `TADAH_DEVELOP=1`, HPO consumes the sibling Tadah.MD and Tadah.LAMMPS sources instead of fetching.
- Substantive README rewrite + correction of stale init/run-policy tables (Session B).
- Community / governance pointer files added.

## [1.2.0] and earlier (condensed)

The `v1.0.0` → `v1.1.1` releases were the "internal playground" era. HPO began life as a small NLOpt wrapper; releases in this period focused on build-system iteration so that HPO could co-build with LAMMPS (CMake + traditional make), `FAIL_SCORE` defaulting, the move to `PC_<type>` performance counters, and a sequence of small fixes. No published changelog covers this era; treat anything before `1.3.0-beta.1` as undocumented in the modern sense.
