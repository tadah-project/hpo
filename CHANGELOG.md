# Changelog

All notable changes to **Tadah.HPO** (hyper-parameter optimisation) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `LIB TADAH ALGO ANNEAL` — classic Kirkpatrick/Metropolis Simulated Annealing driver (refs: Kirkpatrick 1983; Metropolis 1953). Geometric cooling, fixed-shape Gaussian step in transformed-parameter space, Metropolis acceptance, auto-T0 calibration from a small uniform-bounds random walk (overridable via `PARAM T_INIT`). Tunable knobs: `T_MIN`, `RT`, `L`, `STEP_FRAC`, `T0_PROBES`, `T0_PACCEPT`. Honours `MAXEVAL`/`MAXTIME`/`STOPVAL`.
- **`ALGO ANNEAL` Corana adaptive variant** — opt-in via `PARAM CORANA 1` (refs: Corana, Marchesi, Martini, Ridella, *ACM Trans. Math. Soft.* 13:262, 1987; Brommer, Univ. Stuttgart MSc, 2003). Per-dim step vector `V[i]` adapts every `NS`-step cycle from the per-dim accept ratio (grows when rate > 0.6, shrinks when < 0.4); cools after every `NT` step-adjust cycles. Replaces classic's whole-vector Gaussian + fixed σ with one-dim-at-a-time uniform perturbation + adaptive σ — fixes the cold-T freeze that fixed-σ runs hit on continuous multimodal problems. Tunable knobs: `NS` (default 5), `NT` (default 2), `C` (default 2.0). Per-rung journal line shows `V=[V_min, V_max]` so users can watch step-size adaptation. Banner shows `ANNEAL[CORANA]` vs `ANNEAL[CLASSIC]`.
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
