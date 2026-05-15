# Tadah!HPO

A submodule of [Tadah!](https://tadah.readthedocs.io/) that drives
hyperparameter optimisation around a Tadah! training+validation cycle,
with optional LAMMPS-driven physical-property targets.

The user-facing surface is one config file (`config.hpo`) read by
`tadah hpo`. Inside it you choose:

1. **An optimiser** (`OPTIMIZER ... ENDOPTIMIZER`) — `LIB DLIB / NLOPT
   / TADAH` and an `ALGO` (`BFGS`, `GFS`, `LN_SBPLX`, `RANDOM`, ...).
2. **An init strategy** (zero or more `INIT { STRATEGY ... }` lines) — what
   point(s) does the optimiser start from?
3. **A run policy** (any of `STRATEGY { TYPE MULTISTART }`, a `STRATEGY { TYPE PRESTAGE EXPLORE { ... } ENDEXPLORE ... }`
   block, `RESTART, `NOISE.AUTO`) — how is the optimiser invoked?
4. **Loss terms** — `PC_ERMSE` (validation energy-RMSE) and
   `PC_LAMMPS` (per-script `--varloss`).
5. **OPTIM lines** — the search space.

This README is the reference for keywords #1–#3. For full,
worked-through examples see [`examples/`](examples/). For a
beginner-friendly tutorial see [`INIT_STRATEGIES.md`](INIT_STRATEGIES.md).

## Quick start

```
OPTIMIZER
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDOPTIMIZER
```

This is the legacy single-shot form. Bit-identical to releases that
predated the init-strategy / run-policy plumbing.

## Parallelism

Tadah! always links a **serial** LAMMPS library (built with
`BUILD_MPI=OFF`). HPO drives parallelism around LAMMPS, not inside it.

For each parameter evaluation, HPO executes every `PC_LAMMPS --script
<file>` from the HPOTARGET on its own freshly-constructed
`LAMMPS_NS::LAMMPS` instance — no instance is shared between threads.

### Concurrency knobs

- **`OMP_NUM_THREADS`** (env) — controls how many `PC_LAMMPS` scripts
  run concurrently inside a single evaluation. Requires
  `TADAH_ENABLE_OPENMP=ON` at configure time (default in
  `mkz4_develop_tadah.cmake`). With `OMP_NUM_THREADS=N` up to
  `min(N, n_scripts)` LAMMPS instances run in parallel.
- **`THREADS K`** (in `OPTIMIZER` block) — for `DLIB:GFS` only,
  controls dlib's outer thread pool used by `find_min_global`. With
  `K=1` (default) the outer loop is serial. Other optimisers ignore
  this knob.

The two layers compose multiplicatively: a `DLIB:GFS` run with
`THREADS K` and `OMP_NUM_THREADS=N` can have up to `K * N` LAMMPS
instances live at once. Keep the product at or below physical cores
to avoid oversubscription.

### LAMMPS log/screen handling

- HPO injects `-log none -screen none` by default, so individual
  LAMMPS instances stay silent.
- If you supply `-log <file>` or `-screen <file>` explicitly inside a
  `PC_LAMMPS` entry, HPO appends `.t<thread_id>` per OMP thread to the
  filename (so e.g. `-log run.log` becomes `run.log.t0`, `run.log.t1`,
  …). `-log none` / `-screen none` are left untouched.

### Result contract for user scripts

LAMMPS scripts in HPO must communicate their results to Tadah! only
through `--varloss <name>` / `--outvar <name>` (LAMMPS `equal`-style
variables). They should not write files (dumps, restarts,
`write_data`, custom logs) unless the filenames are made unique,
because all OMP threads share the process working directory.

### `--ncpu`

The `--ncpu N` option on a `PC_LAMMPS` line is parsed but only
affects script scheduling order in MPI builds (`TADAH_BUILD_MPI=ON`).
In the default serial build it has no effect on runtime; HPO logs a
warning if `--ncpu > 1` is set in a non-MPI build.

## Performance

### LAMMPS instance reuse

HPO maintains a per-thread, per-script cache of `LAMMPS_NS::LAMMPS *`
instances. The first time a thread executes a given script it
constructs a fresh instance; subsequent evaluations reuse it. The
script's own `clear` directive resets simulation state between runs,
so existing scripts work unmodified.

### Hot variables: `--hotvar`

For LAMMPS scripts whose first action is a `box/relax` from a
hard-coded lattice guess (typical for elastic constants, surface
energies, …), HPO supports a two-phase eval that avoids redoing the
expensive minimisation in every script:

```
PC_LAMMPS --script Ecoh_fast.in --hotvar hot_a0 2.5 4.0 3.302 \
          --varloss Ecoh 8.10 1000
PC_LAMMPS --script C11_fast.in  --varloss C11  263.92 0.2
PC_LAMMPS --script surfaces_fast.in --invar surface 100 \
          --varloss g_su_100 2.7209 800
```

`--hotvar <name> <min> <max> [<initial>]` declares a script as **hot**:

- HPO injects `variable <name> equal <cached-or-initial>` before each
  call. On iter 1 the value is `<initial>` (or the midpoint of
  `[min, max]` if omitted); subsequent iters reuse the previously
  converged value of the same script (cache lives in
  `warmstart_cache.h`).
- The script must redefine `<name>` as an equal-style variable
  (typically `variable <name> equal v_lx/<divisor>`) so HPO can read
  it back. No `variable <name> delete` is needed — HPO uses the same
  `equal` style for injection so the script's redefinition is a
  same-style overwrite, which LAMMPS allows.
- HPO validates the value against `[min, max]`. An out-of-range hot
  value is treated as the unphysical-potential signal: the iter is
  fail-scored and **all cold scripts are skipped**.
- All in-range hot values are then propagated as `${<name>}` strings
  to every cold (no-`--hotvar`) PC_LAMMPS script in the same iter.

Multiple hot scripts are allowed; they run in parallel (OpenMP) in
**phase A** with no cross-references between them. Cold scripts run
in **phase B** after every hot script finishes.

Per-(script, hot-var) values are persisted to `best_pot.tadah` as
`# HOT <script> <name> <value>` lines so a `WARM` restart picks up
exactly where the previous run's best converged. The legacy
`# HOT_A0 <script> <a0>` format still parses, mapped to
`(script, "hot_a0")`.

### Bad-potential early-exit

Each `PC_<METRIC>` line accepts optional `--skip-above <cap>` /
`--skip-rel <ratio>` flags that gate the LAMMPS phase on the
corresponding validation metric:

```
PC_ERMSE 0 0.003 --skip-above 2000          # skip if ERMSE > 2000 meV/at
PC_FRMSE 0 0.5   --skip-rel 5               # skip if FRMSE > 5 × best_seen
```

When any cap fires the LAMMPS phase is skipped entirely and the iter
is fail-scored. `--skip-rel` only triggers once a finite running best
for that metric has been observed.

These flags replace the legacy env vars (`TADAH_HPO_HOT_A0_DEFAULT`,
`TADAH_HPO_A0_MIN`/`MAX`, `TADAH_HPO_BAD_POT_ERMSE_MEV`/`_REL`),
which were ERMSE-only and lattice-only. The new scheme is per-metric
and per-hot-variable, declared once in `config.hpo`.

### Profiling harness

Setting `TADAH_HPO_BENCH=1` writes `hpo_phase_times.csv` and
`hpo_script_times.csv` next to the run, with per-phase wall times
(train / predict / lammps / sims_loss / submit) and per-script
create / run timing. Useful with `perf record` / `valgrind
--tool=callgrind` to verify where time is being spent.

## Init strategies

`INIT { STRATEGY <method> [args...] ... } ENDINIT` — repeatable inside
the `INIT` block. Each `STRATEGY` line appends one entry to the *init
chain*. The chain is resolved into a list of starting points
(`optim_init`) before the optimiser runs.

| Method | Args | Default K | Effect |
|---|---|---|---|
| `CONFIG` | (none) | 1 | Take the value next to each `OPTIM` key. |
| `RANDOM K` | `K` (uint > 0) | `K` | `K` uniform samples in (transformed) bounds. |
| `LHS K` | `K` (uint > 0) | `K` | Latin-hypercube — one sample per stratum per dim. |
| `SOBOL K` | `K` (uint > 0) | `K` | Sobol low-discrepancy sequence. |
| `WARM <file>` | path | 1 | Read previous `pot.tadah`, take values for OPTIM keys. |
| `PRIOR` | (none) | 1 | Per-descriptor `suggested_value`; midpoint fallback. |

### Top-level init knobs

| Keyword | Type | Default | Effect |
|---|---|---|---|
| `INIT { STRATEGY ... }` | repeatable | single `CONFIG` point when no `STRATEGY` line is given | Append one entry to the chain. List `STRATEGY CONFIG` to include the in-config point. |
| `INIT { K_TOTAL ... }` | uint | sum of strategies' `default_k` | Cap on total starting points produced. |
| `INIT { SEED ... }` | uint | random | Seed for stochastic strategies (RANDOM/LHS/SOBOL). |
| `WARM_ON_MISMATCH` | `warn`\|`error`\|`skip` | `warn` | What to do when WARM cannot fill every OPTIM key. |

There is no `INCLUDE_CONFIG` keyword. The in-config point participates
only when `STRATEGY CONFIG` is listed explicitly; the old
`INCLUDE_CONFIG` / `INIT_INCLUDE_CONFIG` keywords are rejected with a
message pointing at `STRATEGY CONFIG`.

### Resolved-chain ordering

The chain is resolved in the exact order the user wrote it — no entry
is reordered. Iter 0 is therefore the first `STRATEGY` line in the
`INIT` block. To make the warm point iter 0, list `STRATEGY WARM`
first. An `INIT` block with no `STRATEGY` line (or no `INIT` block at
all) resolves to a single `CONFIG` point.

### How many seeds the optimiser actually consumes

| Optimiser | Seeds consumed |
|---|---|
| `DLIB:GFS` | All K (forwarded as `initial_function_evals`). |
| `DLIB:BFGS` | First only (`optim_init.front()`). |
| `NLOPT:*` (any algo) | First only. |
| `TADAH:RANDOM`, `TADAH:GRID` | First only — evaluated as eval 0, then sweep. |
| `TADAH:ANNEAL` | First only — used as the SA chain's starting state after T0 calibration. |

`MULTISTART > 1` then runs the inner optimiser K independent times,
each consuming one seed (per-start). For `DLIB:GFS` it collapses to a
single inner call because GFS already consumes K seeds natively.

When the chain produces more seeds than the optimiser will consume,
HPO emits a stderr warning at config time:
`[HPO] WARNING: INIT_STRATEGY chain produced N seeds but ALGO
consumes only the first; (N-1) seed(s) will be discarded.`

## Run policies

Run-policy selection lives inside the mandatory `STRATEGY { ... }
ENDSTRATEGY` block. `TYPE` picks the dispatch shape; `MULTISTART` /
`PRESTAGE` / `BASINS`-specific knobs sit alongside it inside the same
block. `RESTART { ... } ENDRESTART` is a decorator for `TYPE PLAIN`
/ `MULTISTART` / `PRESTAGE`. `NOISE { ... } ENDNOISE` is a sibling
block of `STRATEGY` (one level up, inside `OPTIMIZER`).

| Keyword (location) | Args | Default | Effect |
|---|---|---|---|
| `STRATEGY { TYPE PLAIN \| MULTISTART \| PRESTAGE \| BASINS }` | enum | `PLAIN` | Selects the run policy. `PLAIN` = single-shot; the rest add decorators (see below). |
| `STRATEGY { N_STARTS <int> }` (TYPE MULTISTART) | `K ≥ 1` | `1` | Run the inner optimiser `K` independent times. Validate-rejected for `NLOPT:GN_*`. |
| `STRATEGY { PARALLEL <int> }` (TYPE MULTISTART) | `≥ 1` | `1` | Concurrent starts. Currently parsed and validated; runs serially with a stderr warning if `> 1`. |
| `STRATEGY { BUDGET_PER_START <int> }` (TYPE MULTISTART) | `≥ 1` | derived | Per-start MAXEVAL override. When unset, HPO splits the outer `MAXEVAL` equally across `K` starts (ceil division). |
| `STRATEGY { TYPE PRESTAGE EXPLORE { ... } ENDEXPLORE ... }` | block | (absent) | The `EXPLORE` block runs **first** (one-pass global exploration with its own LIB/ALGO/MAXEVAL/INIT). The outer `STRATEGY` LIB/ALGO then refines its top-K seeds. Replaces the legacy `STAGE_GLOBAL true` keyword. |
| `STRATEGY { TOPK <int> }` (TYPE PRESTAGE) | `K ≥ 1` | `1` | Number of best (loss, params) pairs the PRESTAGE phase forwards to the refinement phase. |
| `STRATEGY { RESTART { AFTER_STAGNATION <int> } }` | `N ≥ 0` | `0` | After this many consecutive evals without a new best, abort and restart with fresh init. `0` disables. |
| `STRATEGY { RESTART { MAX <int> } }` | `M ≥ 0` | `0` | Cap on restart count. |
| `STRATEGY { RESTART { STRATEGY <method> [args] ... } }` | repeatable | (none) | Init chain used to draw the fresh point on each restart (re-uses the `INIT`-block grammar). CONFIG is *not* auto-prepended. |
| `OPTIMIZER { NOISE { AUTO [bool] } ENDNOISE }` | optional bool | `false` | Derive `PARAM fd_step` from the search-space box diagonal: `max(1e-3 × diag / sqrt(dim), 1e-6)`. **`DLIB:BFGS` only**, only when `PARAM fd_step` is not already set explicitly. Bareword `AUTO` parses as `true`. |
| `OPTIMIZER { NOISE { CALIBRATION <int> } ENDNOISE }` | `0` or `≥ 2` | `0` | Repeat the seed evaluation N times to surface the objective's noise floor and emit a suggested `FTOL_REL`. `0` disables; `1` is rejected. |

### `STRATEGY { TYPE BASINS }` — anytime "run and forget" strategy

A self-contained top-level strategy (selected via `STRATEGY { TYPE BASINS }` instead
of a regular `LIB DLIB / LIB NLOPT`). BASINS alternates a global stage
that harvests candidate basins with a cumulative local stage that
refines the persistent basin pool at progressively tighter `ftol_rel`.
Designed for hour/day/week-long invocations on workstations or SLURM
clusters: SIGINT (ctrl-C), SIGTERM, and SIGUSR2 all trigger a clean
checkpoint commit, and re-running the same command resumes the saved
state from `checkpoint.tadah`.

See `examples/15_basins_run_and_forget/` for a worked config and the
`README.md` there for the full mental model.

| Keyword | Args | Default | Effect |
|---|---|---|---|
| `EXPLORE.LIB / EXPLORE.ALGO` | strings | `NLOPT GN_CRS2_LM` | Global-stage optimiser (one per round). |
| `REFINE.LIB  / REFINE.ALGO`  | strings | `NLOPT LN_BOBYQA`  | Local-stage optimiser (one call per due basin per rung). |
| `EXPLORE.FTOL_LADDER`            | list of `>0` | auto from NOISE.CALIBRATION | Per-round `ftol_rel` for the global stage; clamps to last entry past length. |
| `REFINE.FTOL_LADDER`            | list of `>0` | auto                       | Cumulative local-stage `ftol_rel`s; round k visits k rungs. |
| `BASIN_POOL_MAX`            | int ≥ 1     | `32`  | Capacity of the persistent basin pool (worst-loss eviction on overflow). |
| `BASIN_DEDUP_DIST`          | ≥ 0         | `0.05`| Relative L2 in normalised bbox: candidates closer than this merge into an existing basin. |
| `BUDGET_RATE_FACTOR`         | ≥ 0         | `0.5` | Per-stage stuck threshold: `is_stuck` requires `rate_per_eval < factor × tol`. |
| `BUDGET_MIN_EVALS`           | int         | `30`  | Floor on per-stage evals before stuck-detection fires. |
| `BUDGET_MAX_EVALS`           | int         | `5000`| Per-stage / per-basin safety cap for the inner optimiser. |
| `STOP_NO_IMPROVE_ROUNDS` | int       | `5`   | Emergency stop: terminate after this many consecutive rounds with no global best improvement. |
| `CHECKPOINT_EVERY`          | `<N>s` or `<N>` | `0` (off; stage-boundary commits only) | Time- or eval-based checkpoint cadence. |

`MAXTIME` and `MAXEVAL` from the outer OPTIMIZER block remain valid
overall budgets. `BASINS` ignores `STRATEGY { TYPE MULTISTART }`, `STRATEGY { TYPE PRESTAGE }`, and
`RESTART` (it drives global+local internally).

### `LIB TADAH ALGO ANNEAL` — classic Simulated Annealing

In-tree Kirkpatrick/Metropolis SA driver (refs: Kirkpatrick et al.,
*Science* 220, 1983; Metropolis et al., *J. Chem. Phys.* 21, 1953).
Operates entirely in the post-`LOG_HP` transformed parameter space;
the LAMMPS side always sees real-space values.

Stops honoured: `MAXEVAL` (hard cap on host calls, **including** T0
calibration probes), `MAXTIME`, `STOPVAL`, plus the SA-specific
`PARAM T_MIN` floor. `FTOL_REL` / `FTOL_ABS` are **not** honoured by
this driver (no NEPS history; planned for a follow-up). The init
strategy chain's first point is the SA chain's starting state — only
the first seed is consumed (later seeds are dropped, with the usual
warning).

| PARAM         | Default       | Effect |
|---------------|---------------|--------|
| `T_INIT`      | auto          | Initial temperature. When set, T0 calibration is skipped (no probes are spent). |
| `T_MIN`       | `1e-12`       | Cooling floor; the loop stops when `T < T_MIN`. |
| `RT`          | `0.85`        | Geometric cooling factor in `(0,1)`: `T <- T * RT`. |
| `L`           | `50`          | Markov-chain length per temperature. |
| `STEP_FRAC`   | `0.10`        | `sigma_i = STEP_FRAC * (high_i - low_i)` per dim. |
| `T0_PROBES`   | `20`          | Random uniform-bounds probes used to calibrate T0. |
| `T0_PACCEPT`  | `0.80`        | Target acceptance probability for T0: `T0 = -mean|dE| / ln(P)`. |

Top-level `STEP <s>` overrides `STEP_FRAC`, applying `sigma_i = s` for
all dims (uniform sigma). `SEED <int>` seeds the RNG.

When the T0 calibration probes all return the same loss
(`mean|dE| == 0`, e.g. saturated `FAILSCORE` everywhere), T0 falls
back to `1.0` and a warning is logged — pick `PARAM T_INIT` or widen
the bounds in that case.

#### Corana adaptive variant (`PARAM CORANA 1`)

Optional opt-in alternative to the classic Kirkpatrick variant
(refs: Corana, Marchesi, Martini, Ridella, *ACM Trans. Math. Soft.*
13:262, 1987; Brommer, Univ. Stuttgart, 2003). Different inner
structure but same setup, T0 calibration, stops, and journal:

- Perturbs **one dim at a time** (uniform `U(-1,1) * V[i]`) instead of
  the whole vector together.
- Maintains an **adaptive per-dim step vector** `V[i]` (initialised to
  the same `STEP_FRAC * width` as classic σ).
- Every `NS` step cycles, computes per-dim accept rates and adapts
  `V[i]`: grows by `(1 + C·(r_i - 0.6)/0.4)` when `r_i > 0.6`,
  shrinks by `1 / (1 + C·(0.4 - r_i)/0.4)` when `r_i < 0.4`.
- Cools by `RT` after every `NT` step-adjust cycles. Per-rung host
  evals = `NT · NS · D`, replacing classic's `L`.

Why pick Corana over classic on continuous multimodal problems:
classic's fixed σ proposes broad-search jumps even at very low T,
which the cold Metropolis gate rejects en masse — the chain freezes.
Corana's `V[i]` shrinks naturally as the chain settles, so cold-T
proposals stay inside the basin and downhill steps still register.

| PARAM | Default | Effect |
|-------|---------|--------|
| `CORANA` | `0` | Set to `1` to enable the adaptive variant. |
| `NS` | `5` | Step cycles per step-adjust cycle. |
| `NT` | `2` | Step-adjust cycles per T (cooling triggers every `NT`). |
| `C` | `2.0` | Step-adjust gain — controls how fast V[i] reacts to accept-rate deviation from the [0.4, 0.6] deadband. |
| `V_FLOOR_FRAC` | `1e-3` | Lower clamp on `V[i]` as a fraction of `(high_i - low_i)`. Vanilla Corana has no explicit floor; on FAILSCORE-cliff landscapes `V[i]` can collapse by factor `(1+C)^NT` per zero-accept rung, killing the chain. Default `1e-3` keeps the chain mobile. |
| `RESET_TO_BEST` | `0` | Brommer (2003) modification: set to `1` to snap the chain back to the best-so-far point after every cooling step. Gives the adaptive chain "basin memory" — high-T wanders no longer cost progress because the next rung restarts from the deepest known basin. Strongly recommended on FAILSCORE-cliff landscapes or when the chain wanders during the high-T phase. |

`L` is ignored when `CORANA 1` is set. The per-rung journal line
includes `V=[V_min, V_max]` so users can watch how the step vector
adapts. The Corana paper recommends `NS=20`, `NT=max(100, 5D)`, but
those defaults assume cheap function evaluations — for LAMMPS-driven
HPO with ~600 ms/eval, we ship lower defaults (`NS=5`, `NT=2`) that
cover a useful T schedule within practical wall budgets.

#### Journal output

Per run, `ANNEAL` writes the following structured trace to
`hpo_run.log` (timestamps elided):

```
ANNEAL: seed=1234  L=20  RT=0.85  T_MIN=1e-12  STEP_FRAC=0.15
ANNEAL: T0=8.188e+00 from 20 probes,  mean|dE|=1.83e+00  loss range=[3.20e-02, 1.40e+01]  std=2.10e+00  target p_accept=8.000e-01
ANNEAL rung k=0  T=8.188e+00  proposed=20  accepted=14 (70.0%)  oob=2 (9.1%)  f_x=2.31e-01  best=1.84e-01
ANNEAL rung k=1  T=6.960e+00  proposed=20  accepted=11 (55.0%)  oob=0 (0.0%)  f_x=1.97e-01  best=1.84e-01
...
ANNEAL summary: rungs=12  proposed=240  accepted=98 (40.8%)  oob=4 (1.6%)  T0=8.188e+00  T_final=2.46e-01  best=8.71e-02
ANNEAL accept-rate trajectory: min=20.0%  max=70.0%
ANNEAL stopped: MAXEVAL exhausted after 260 host eval(s) (12 rung(s) completed)
```

Reading the trace:

- **Accept rate stays ≈ 100% across all rungs** → `T_INIT` /
  `T0_PACCEPT` too high; chain is doing a random walk. Lower
  `PARAM T0_PACCEPT` (e.g. `0.5`) or set `PARAM T_INIT`.
- **Accept rate collapses in 2–3 rungs** → `RT` too aggressive; use
  `0.90`–`0.95`.
- **`oob` fraction > 30%** → `STEP_FRAC` (or top-level `STEP`) too
  large for the bounds; halve it.
- **Best stagnates while accept rate is > 50%** → chain is escaping
  basins but not converging; raise `PARAM L`.
- **`ANNEAL stopped: MAXTIME exceeded` after only a few rungs** →
  wall-clock cap killed SA before it could cool. Raise `MAXTIME`,
  reduce `L`, or switch to `LIB NLOPT ALGO LN_BOBYQA` for faster
  local refinement.

The standard `BEST` events from `HPO_Host` interleave with the rung
lines; best-tracking is identical to other drivers.

### Composition

- A `STRATEGY { TYPE PRESTAGE EXPLORE { ... } ENDEXPLORE ... }` block triggers staged-global mode by
  itself; do **not** also write `STAGE_GLOBAL true` (that keyword has
  been removed).
- `INNER ... ENDINNER` (formerly `LOCAL ... ENDLOCAL`) is unrelated and
  only carries NLopt's local sub-algorithm (NLopt MLSL / G_MLSL_LDS / AUGLAG).
- `STRATEGY { TYPE PRESTAGE }` + (top-K > 1) → the refinement automatically runs as
  `STRATEGY { TYPE MULTISTART }` over the top-K seeds.
- `RESTART { AFTER_STAGNATION ... }` wraps any inner choice — combine with
  `MULTISTART` or `PRESTAGE` for a global+restart hybrid.
- `STRATEGY { TYPE MULTISTART, N_STARTS > 1 }` is rejected at validate-time for NLopt's `GN_*`
  algorithms (those globals consume one starting point only); use
  `LD_*` / `LN_*` or `DLIB:GFS` / `DLIB:BFGS`.

### Diagnostic output

Each policy emits a one-line stdout marker when it fires:

| Line | Meaning |
|---|---|
| `[HPO] InitStrategy NAME produced N starting points` | After init resolution. |
| `[HPO] PRIOR: N parameter(s) seeded from descriptor priors` | PRIOR success. |
| `[HPO] PRIOR: N parameter(s) used bounds-midpoint fallback` | Some descriptors had no prior. |
| `[HPO] WARM: loaded N parameter(s) from <file>` | WARM read all keys. |
| `[HPO] WARM: primed warmstart cache with N HOT entry(ies)` | `# HOT <script> <name> <value>` (or legacy `# HOT_A0`) lines parsed. |
| `[HPO] WARM: clamped dim[d] from X to Y (bounds […])` | Warm point fell outside narrowed bounds. |
| `[HPO] DLIB:GFS seeding with N starting points` | Multi-seed forwarding. |
| `[HPO] MULTISTART: running K starts (parallel=N)` | Multistart fires. |
| `[HPO] MULTISTART per-start MAXEVAL = N` | Either from BUDGET_PER_START or split. |
| `[HPO] MULTISTART start i/K — running inner optimiser` | Per-start. |
| `[HPO] PRESTAGE: exploration phase LIB:ALGO (MAXEVAL=...)` | PRESTAGE phase starts. |
| `[HPO] PRESTAGE: top-K losses: …` | Snapshot quality after PRESTAGE finishes. |
| `[HPO] REFINEMENT: refining top-K seed(s) via LIB:ALGO` | Outer (refinement) phase starts. |
| `[HPO] RESTART i/M — fresh init from STRATEGY` | Stagnation watchdog fired. |
| `[HPO] NOISE_AUTO: PARAM fd_step set to V (diag=…, dim=…)` | NOISE.AUTO fired and shows the value used. |
| `[HPO] NOISE_CALIBRATION: running N evals at the seed point` | NOISE.CALIBRATION started. |

The `hpo_run.log` file additionally records a structured `Init
strategy:` and `Run policy:` block at run start so the configuration
is preserved alongside results. Either file is the canonical record
of which features fired.

## OPTIM-able context keys: reinit-safety reference

When HPO caches a `Trainer` across iters, every per-iter context
change must be propagated into the cached objects via
`Trainer::reinit()`. A key is **reinit-safe** if changing it between
iters yields the same WEIGHTS that a freshly built Trainer would
produce for the new params. A key consumed in a constructor and
never re-read is **unsafe** — `reinit()` will silently use the value
from when the cached object was built.

(See `INVESTIGATIONS/warm_reproducibility/FINDINGS.md` for the WATOM
regression that motivated this audit.)

This table covers every key consumed in the train-path constructors
audited against their respective `reinit()` overrides.

### Safe to OPTIM (numeric, refreshed on every iter)

| Key | Type | Consumer (file:line) | Refreshed by |
|---|---|---|---|
| `WATOM` | `double[]` | `MODELS/src/d_base.cpp:60-64` | `D_Base::refresh_atom_weights` (called from `DC_Selector::reinit`) |
| `EWEIGHT` | `double` | `MLIP/.../design_matrix.h:104` | `DesignMatrix::refresh_context_scalars` (called from `Trainer::reinit`) |
| `FWEIGHT` | `double` | `MLIP/.../design_matrix.h:105` | same |
| `SWEIGHT` | `double` | `MLIP/.../design_matrix.h:106` | same |
| `ESTDEV` | `double` | `MLIP/.../design_matrix.h:109` | same |
| `FSTDEV` | `double` | `MLIP/.../design_matrix.h:111` | same |
| `SSTDEV` | `double[6]` | `MLIP/.../design_matrix.h:113` | same |
| `RCUT2B` | `double` | `D2_Blip::reinit:105`, `D2_BP::reinit:99`, `D2_MIE::reinit:89`, `D2_ZBL::reinit:128`, `D2_mJoin::reinit:75` | each descriptor's `reinit` re-reads `RCUT2BMAX` (auto-derived from `RCUT2B`) and forwards to `fcut->set_rcut()` |
| `RCUTMB` | `double` | `DM_Blip::reinit:334`, `DM_EAD::reinit:409`, `DM_mEAD<F>::reinit:488`, `DM_Poly<F>::reinit:413`, `DM_mJoin::reinit:130` | similar (re-reads `RCUTMBMAX`) |
| `RCUTENV` | `double` | `F_mEnv::F_mEnv:214` (used by `DM_mBlipEnv` and any `DM_Poly<F_mEnv<...>>`) | `DM_mEAD::reinit:491` and `DM_Poly::reinit:416` reconstruct the embedding function which re-reads `RCUTENV` |
| `CEMBFUNC` | `double[]` | `F_RLR`, `F_SQRT`, `F_Blip`, `F_mEnv` ctors (`MODELS/src/dm_func.cpp`) | parent `DM_mEAD` / `DM_Poly` reinit reconstructs the embedding function |
| `SEMBFUNC` | `double[]` | same | same |
| `CGRID2B` | `double[]` | `D2_Blip::reinit:102`, `D2_BP::reinit:96`, `D2_mJoin::reinit:77-78` | descriptor's `reinit` calls `get_grid(ctx, ...)` |
| `SGRID2B` | `double[]` | same | same |
| `CGRIDMB` | `double[]` | `DM_Blip::reinit:329`, `DM_EAD::reinit:405`, `DM_mEAD::reinit:484`, `DM_Poly::reinit:409`, `DM_mJoin::reinit:127-133` | same |
| `SGRIDMB` | `double[]` | same | same |
| `TYPE2B[1..]` | mixed | `D2_MIE::reinit:87-88` (n,m), `D2_ZBL::reinit:124-127` (a0,s0,p0,p1) | numeric tail of `TYPE2B` re-read by `reinit`; the leading element (descriptor name string) is not OPTIMable. |
| `TYPEMB[1]` | int | `DM_Blip::reinit:332`, `DM_EAD::reinit:407`, `DM_mEAD::reinit:486`, `DM_Poly::reinit:411` (`Lmax`) | each descriptor's `reinit` re-reads. Caveat: changing `TYPEMB[1]` (Lmax) changes descriptor size; reinit returns true and `Trainer` is rebuilt. |

### **Unsafe** to OPTIM (consumed in constructor, NOT refreshed)

| Key | Where consumed | Why unsafe | Workaround |
|---|---|---|---|
| `RCUT2B` *with `TYPE2B D2_EAM`* | `MODELS/src/d2_eam.cpp:20` (validation against the EAM `.setfl` file's intrinsic cutoff) | `D2_EAM` has no `reinit()` override, and the EAM cutoff is a physical property of the tabulated potential. | Don't OPTIM `RCUT2B` when using `D2_EAM`. |
| `RCUTMB` *with `TYPEMB DM_EAM`* | `MODELS/src/dm_eam.cpp:20` | same | Don't OPTIM `RCUTMB` with `DM_EAM`. |
| `TYPE2B[i]` *with `TYPE2B D2_EAM`* (file path / format) | `d2_eam.cpp:17` | string field, not numeric — but the `.setfl` file is loaded once. | Not realistically OPTIMable. |
| `TYPEMB[i]` *with `TYPEMB DM_EAM`* (file path) | `dm_eam.cpp:17` | same | Not realistically OPTIMable. |

### Not OPTIMable by design (structural / non-numeric)

`MODEL`, `ATOM`, `TYPE2B[0]` (descriptor name string), `TYPEMB[0]`,
`RCTYPE2B`, `RCTYPEMB`, `INIT2B`, `INITMB`, `BIAS`, `NORM`, `FORCE`,
`STRESS`, `DBFILE`, `VERBOSE`, `DSIZE` (internal), `RCUT2BMAX`,
`RCUTMBMAX`, `RCUTMAX` (internal, derived from `RCUT2B`/`RCUTMB`).

### Verifying a key

If you want to OPTIM a key not on the table above, the safest test
is the long-run reinit-vs-rebuild equivalence check used to find the
WATOM bug:

```bash
# Same MAXEVAL=200 run twice, identical except for one env var
... tadah hpo --hpotarget config.hpo -c master.tadah  > /dev/null
cp loss.tadah /tmp/A.tadah
TADAH_HPO_FORCE_REBUILD=1 \
... tadah hpo --hpotarget config.hpo -c master.tadah  > /dev/null
diff loss.tadah /tmp/A.tadah    # expect: empty (byte-identical)
```

`TADAH_HPO_FORCE_REBUILD=1` forces a fresh `Trainer` rebuild every
iter instead of using the cached `reinit()` path. Any byte-level diff
indicates the new key is consumed in some constructor and not
refreshed by `reinit()`.
