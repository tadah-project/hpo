# Changelog

All notable changes to **Tadah.HPO** (hyper-parameter optimisation) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- PC_LAMMPS: the host now builds ONE candidate-potential object per (thread, evaluation) and shares it across every `clear`/`pair_coeff EXTERN` cycle of every script that thread runs (`LammpsRunner::run_script` shared-potential overload; was a full `LammpsTadah` reconstruction per `pair_coeff` — 75x per evaluation on run15-class setups). Bitwise-identical objective; the hot->cold early-reject barrier is unchanged.
- `LammpsRunner::run_script` takes the potential Context by const reference (was by value — one deep copy per sim per evaluation).
- Startup journal now logs the OpenMP pool width (`omp_get_max_threads` and `OMP_NUM_THREADS`) — the PC_LAMMPS script fan-out is capped by it.

### Fixed
- A throwing candidate-potential construction now marks the sim failed before any LAMMPS command executes (previously the EXTERN line could proceed with a null potential).

## [1.3.0-beta.1] — 2026-05-20

HPO underwent the largest restructure of any module in this release: new init strategies, resume/restart, varloss, dlib-BFGS local optimiser, the `TADAH:ANNEAL` driver family (classic + Corana adaptive variant), the `CERES` gradient-based optimiser family, and a regression-tested example suite.

### Added
- **Init strategies** — multiple ways to seed the HPO sweep (`CONFIG`, `RANDOM`, `LHS`, `SOBOL`, `WARM`); fully documented in the rewritten `hpo.rst`.
- **Resume / restart** — checkpointed long runs can be resumed from the most recent journal entry; backed by the new CORE checkpoint + signal-handler infrastructure.
- **Variance-of-loss (`varloss`) acquisition** — `--varloss` / `--outvar` flags select an alternative scoring function over a user-provided set of LAMMPS `equal`-style script variables; HPO now validates these variables exist before launching workers.
- Standalone **dlib BFGS local optimiser** — added alongside the existing nlopt path.
- `HPOLog` run journal — structured per-evaluation log written incrementally.
- Per-dataset validation metrics and stats output files.
- Per-thread `Trainer` / `TrainerHost` cache — when structural parameters do not change between evaluations, the cached object is reused via `Trainer::reinit()` instead of being reconstructed (large speed-up for sweeps that vary only weights or scalar regression parameters).
- **Reference example suite** under `HPO/examples/` with `run_reference_suite.sh` driver and per-example fingerprint files (`reference_fingerprint.json`) for regression testing.
- `TADAH_ENABLE_ASAN` CMake option.
- **`LIB TADAH ALGO ANNEAL` — classic Kirkpatrick/Metropolis Simulated Annealing driver** (refs: Kirkpatrick 1983; Metropolis 1953). Geometric cooling, fixed-shape Gaussian step in transformed-parameter space, Metropolis acceptance, auto-T0 calibration from a small uniform-bounds random walk (overridable via `PARAM T_INIT`). Tunable knobs: `T_MIN`, `RT`, `L`, `STEP_FRAC`, `T0_PROBES`, `T0_PACCEPT`. Honours `MAXEVAL`/`MAXTIME`/`STOPVAL`. Dispatched from `run_inner()` alongside `TADAH:RANDOM` / `TADAH:GRID`; reflects out-of-bound proposal components back into the box (high-dim robustness fix).
- **`ALGO ANNEAL` Corana adaptive variant** — opt-in via `PARAM CORANA 1` (refs: Corana, Marchesi, Martini, Ridella, *ACM Trans. Math. Soft.* 13:262, 1987; Brommer, Univ. Stuttgart MSc, 2003). Per-dim step vector `V[i]` adapts every `NS`-step cycle from the per-dim accept ratio (grows when rate > 0.6, shrinks when < 0.4); cools after every `NT` step-adjust cycles. Replaces classic's whole-vector Gaussian + fixed σ with one-dim-at-a-time uniform perturbation + adaptive σ — fixes the cold-T freeze that fixed-σ runs hit on continuous multimodal problems. Tunable knobs: `NS` (default 5), `NT` (default 2), `C` (default 2.0). Per-rung journal line shows `V=[V_min, V_max]`; banner shows `ANNEAL[CORANA]` vs `ANNEAL[CLASSIC]`.
- **`PARAM V_FLOOR_FRAC`** (default `1e-3`) — explicit lower clamp on `V[i]` as a fraction of bound width. Guards Corana against `V[i]` runaway collapse on `FAILSCORE`-cliff landscapes where every uphill ΔL ~ FAILSCORE rejects for any T, driving accept rate to 0.
- **`PARAM RESET_TO_BEST 1`** — Brommer (2003) modification: snap the chain back to the best-so-far point at the end of every cooling step. Gives Corana basin memory; addresses the high-T wander failure mode.
- `ANNEAL` journal trace — `hpo_run.log` records a banner echoing the resolved knobs, an extended T0 calibration line, per-cooling-rung summary lines, and an explicit `ANNEAL stopped: <reason>` end line. Lets users diagnose mistuned `T_INIT`/`RT`/`STEP_FRAC` from the journal alone.
- **`LIB CERES` family** — `LBFGS` (limited-memory BFGS, Wolfe line search), `WOLFE` (full BFGS, Wolfe line search), `NLCG` (nonlinear conjugate gradient), `STEEPEST_DESCENT` (Armijo line search). All four use Ceres' `GradientProblemSolver`; gradient via `DynamicNumericDiffCostFunction` (central differences; step from `PARAM fd_step`). The Ceres NLLS family is deferred at dispatch with a tailored error.
- **`OptimizerCaps` registry** — single source of truth for `(LIB, ALGO)` capabilities (`is_global` / `is_population` / `is_gradient_based` / `accepts_inner` / `accepts_multi_seed` + ftol routing discriminant). Adding new optimisers is now two changes (run-function + caps entry).
- **`PC_LAMMPS --hotvar <name> <min> <max> [<initial>]`** — declares a `PC_LAMMPS` line as *hot*: a two-phase evaluator. Hot scripts run alone in **phase A** of each iteration (in parallel with other hot scripts), expose a value via `variable <name> equal …`, and have it validated against `[min, max]`. Validated values are cached per `(script, name)` and propagated as `${<name>}` strings to every cold (no-`--hotvar`) `PC_LAMMPS` script in **phase B**. Multiple `--hotvar` flags per line allowed; phase A is parallel so no cross-references. Out-of-range hot values fail-score the iteration and skip phase B.
- **`PC_<METRIC> --skip-above <cap>` / `--skip-rel <ratio>`** — per-metric bad-pot pre-filter on `PC_EMAE`, `PC_ERMSE`, `PC_ErRMSE`, `PC_ERSQ`, `PC_FRMSE`, `PC_SRMSE`. Triggers fire *before* the LAMMPS phase, fail-scoring the iter when the metric exceeds the cap or `metric > ratio × running_best` for the same metric.
- **Pre-flight audit hook + per-iter health monitor** — wires the MLIP bound-validation pipeline into the HPO surface. Driven by blueprint keys `AUDIT`, `AUDIT_SAMPLE`, `HEALTH_LOG` (in CORE).
- **`EFILTER` / `FFILTER` forwarded from `--validation`** — the validation-side LOAD-TRANSFORMS path now respects the same energy/force filtering as training. Symmetric LOAD-TRANSFORMS log; mean-share breakdown.

### Changed
- **Major restructure** — init strategies, resume, varloss reorganised into clearly separated layers.
- `structural_params_changed` flag retired — the trainer now decides via `Trainer::reinit()` semantics whether a rebuild is needed.
- Best-pot writer rewritten (now v2 layout); flushes outputs on short runs even before the run ends.
- Include-file parsing is now case-insensitive.
- `run_lammps` is now two-phase (hot first, cold second). When no `--hotvar` is declared the behaviour is identical to before: phase A is empty, all scripts run in parallel in phase B. The previous per-script post-hoc range check is replaced by the explicit `--hotvar` declaration; the hot-value cache is now keyed on `(script, varname)` instead of just `script`.
- `best_pot.tadah` warmstart cache lines: `# HOT_A0 <script> <a0>` → `# HOT <script> <name> <value>`. The legacy format still parses on read.
- GitLab CI rewritten from scratch — replaces the auto-generated Auto-DevOps configuration with a hand-written pipeline (build, test, lint, docs link to umbrella, release).

### Removed
- `TADAH_HPO_HOT_A0_DEFAULT`, `TADAH_HPO_A0_MIN`, `TADAH_HPO_A0_MAX`, `TADAH_HPO_BAD_POT_ERMSE_MEV`, `TADAH_HPO_BAD_POT_ERMSE_REL` environment variables. All five knobs now live in `config.hpo` as `PC_LAMMPS --hotvar` / `PC_<METRIC> --skip-above` flags. Migration: replace `${hot_a0}` script convention with an explicit `--hotvar hot_a0 <min> <max> [<initial>]` on the producing `PC_LAMMPS` line; replace `TADAH_HPO_BAD_POT_ERMSE_MEV=N` with `PC_ERMSE … --skip-above N`.
- `PRIOR` init strategy (parser path and the in-tree `INIT_STRATEGIES.md` tutorial). The `CONFIG` / `RANDOM` / `LHS` / `SOBOL` / `WARM` chain replaces it.
- `INIT { INCLUDE_CONFIG }` flag and the chain-reordering behaviour it triggered.
- Engineer-facing reference notes `HPO/LOAD_TRANSFORMS.md` and `HPO/INIT_STRATEGIES.md` — their content was folded into the user-facing `hpo.rst` in Session B's doc audit, leaving the in-tree notes redundant.

### Fixed
- `rcut` optimisation path corrected.
- Linking errors when HPO was built with optional features enabled (multiple commits).
- MPI rank-handle leaks resolved (paired with the MLIP `MPI_Type_free` fix).
- `LammpsRunner` API alignment (after a brief revert/redo cycle).
- Hardcoded `/home/marcin` paths removed from `examples/common/run.sh` (`tadah` is now resolved on `PATH` by default).
- `StagnationStop` recovery in `HPO_Host::operator()` no longer overwrites the eval's real loss with `FAIL_SCORE` and no longer flags the result as `was_nan_clamped` — both were corrupting `loss.tadah` for evals that successfully completed before the threshold check. The per-eval flood (one warning per concurrent OMP eval that crossed the threshold) is throttled to one diagnostic per stagnation event.
- `ALGO ANNEAL` whole-proposal rejection in high dimensions — replace "any-dim-OOB → reject" with per-dim mirror reflection. Old code rejected ~98% of proposals on a 38-dim Ta workload; reflection lets every L-step do real work, recovering full chain mobility.
- `pot_header_writer` wrote the `DBFILE` path through `fs::absolute()`, embedding the author's filesystem layout into shipped `best_pot.tadah` headers. It now records the path as written in `config.tadah` (relative).
- Hot variables are now injected into LAMMPS as `equal`-style instead of `string`-style. A hot script that redefines `<name>` (per the README contract) becomes a same-style overwrite — no `variable delete` is needed and the cached LAMMPS instance no longer traps a stale definition into the next iteration; the warmstart cache evolves to the relaxed value instead of freezing at the injected seed.
- `run_reference_suite.sh` return-code carry-over — a single example FAIL no longer marks every later example FAIL in the summary table.

### Internal
- Per-run output artefacts un-tracked from `examples/` (stay reproducible via the driver, no longer cluttering `git status`).
- `tadah_use_local_module` integration — under `TADAH_DEVELOP=1`, HPO consumes the sibling Tadah.MD and Tadah.LAMMPS sources instead of fetching.
- Substantive README rewrite + correction of stale init/run-policy tables.
- Community / governance pointer files added.
- Test build routed through `cmake-tools/cmake/TadahTesting.cmake` helpers (`tadah_declare_testing` / `tadah_add_tests`); external dependency fetches (dlib, Eigen, Ceres, nlopt) wrapped in `tadah_begin/end_external_deps`. **Fixes a latent test-suite-drop bug** where the previous hand-rolled `BUILD_TESTING` capture/restore ran before the cache entry was defined and silently disabled the whole `tests/` subdir on a bare reconfigure. Also sets `NLOPT_TESTS=OFF` explicitly.
- Test fixture `strategy_matrix.manifest` migrated to the block grammar; guarded by a parser-rejection check so future grammar drift trips the test.
- Examples: stale leading-comment numbers fixed after the example set was renumbered.

## [1.2.0] and earlier (condensed)

The `v1.0.0` → `v1.1.1` releases were the "internal playground" era. HPO began life as a small NLOpt wrapper; releases in this period focused on build-system iteration so that HPO could co-build with LAMMPS (CMake + traditional make), `FAIL_SCORE` defaulting, the move to `PC_<type>` performance counters, and a sequence of small fixes. No published changelog covers this era; treat anything before `1.3.0-beta.1` as undocumented in the modern sense.
