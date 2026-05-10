# Tadah!HPO — initial guesses and run policies, in plain English

This guide is a tutorial for someone who has never used HPO before.
It explains what the new keywords do, when each one helps, and how to
combine them. The cross-references all point to a runnable example in
[`examples/`](examples/) so you can see the keyword in action.

This tutorial covers only behaviour that has been exercised end-to-end
by the example bundle. Anything that is parsed but does not have a
verified runtime effect is called out as such.

For the keyword reference (every flag, every default), see
[`README.md`](README.md). For the table of OPTIM-able context keys, see
the same file's *OPTIM-able context keys* section.

## The two-layer mental model

HPO splits two questions:

1. **How should the search be seeded?** Init strategies produce one or
   more starting points in real (untransformed) parameter space.
2. **How should the search be driven?** Strategy types decorate the
   per-optimiser dispatch — single-shot (`TYPE PLAIN`), multi-start
   (`TYPE MULTISTART`), two-stage global→local (`TYPE PRESTAGE`),
   or anytime/interruptible (`TYPE BASINS`).

Init knobs live in the `INIT { ... } ENDINIT` block; strategy
selection lives in the `STRATEGY { TYPE ... }` block. Both sit at the
top of the `OPTIMIZER ... ENDOPTIMIZER` scope:

```
OPTIMIZER
  INIT       # seeding
    ...
  ENDINIT
  NOISE      # objective noise (auto fd_step / calibration)
    ...
  ENDNOISE
  STRATEGY   # mandatory; TYPE picks the policy
    TYPE PLAIN
    LIB DLIB
    ALGO BFGS
    MAXEVAL 50
  ENDSTRATEGY
ENDOPTIMIZER
```

With no `INIT` block and `STRATEGY { TYPE PLAIN }`, behaviour collapses
to the legacy single-point CONFIG seed plus single-shot dispatch.

## Vocabulary

- **Init chain** — the ordered list of `STRATEGY` lines inside the
  `INIT { ... }` block. After resolution, this becomes `optim_init`,
  the list of starting points the optimiser will draw from. The chain
  is *resolved* into the list once, before the optimiser runs.
- **Iter 0** — the first evaluation HPO performs. With `INIT { STRATEGY
  CONFIG }` this is the value next to your `OPTIM` lines.
- **Implicit CONFIG** — when you do not write any `INIT { STRATEGY }`
  line, HPO behaves as if you wrote `INIT { STRATEGY CONFIG }`.
- **Bound transform** — `LOG_HP <ratio>` (or environment-driven defaults)
  log-transforms dims whose `high/low` ratio exceeds `<ratio>`. From
  this point onward, internal arithmetic happens in transformed space;
  only the output is exponentiated back when reported.

## Init strategies, one by one

### CONFIG

```
INIT
  STRATEGY CONFIG
ENDINIT
```

Reads the value next to each `OPTIM` line. Always available; produces
exactly one starting point. Implicit by default — so a config without
any `INIT { STRATEGY }` line behaves as if you wrote this one.

**When you write a chain explicitly, the implicit CONFIG is dropped.**
The diagnostic on stdout makes this visible:

```
[HPO] InitStrategy: explicit chain set, implicit CONFIG suppressed.
Add `STRATEGY CONFIG` explicitly inside INIT if you want the in-config
point too.
```

If you want CONFIG mixed into your explicit chain, write it.

See [`examples/01_baseline_bfgs/`](examples/01_baseline_bfgs/) for the
implicit-only case.

### RANDOM K

```
INIT
  STRATEGY RANDOM 4
  SEED 42
ENDINIT
```

`K` uniform-random samples inside the bounds (in transformed space).
Cheap; useful when CONFIG is far from a basin and you have no priors.
Reproducible if you set `INIT { SEED }`. Without it, draws come from
`std::random_device`.

See [`examples/02_random_seeded/`](examples/02_random_seeded/).

### LHS K (Latin hypercube)

```
INIT
  STRATEGY LHS 6
  SEED 11
ENDINIT
```

For each dim, the box is split into `K` equal strata; one sample lands
in each stratum (with per-stratum jitter). For small `K`, this fills
the box more evenly than RANDOM at the same `K`. Reproducible via
`INIT { SEED }`.

See [`examples/03_lhs_multistart/`](examples/03_lhs_multistart/).

### SOBOL K

```
INIT
  STRATEGY SOBOL 32
  SEED 0
ENDINIT
```

A quasi-random Sobol sequence — deterministic by construction and
better-distributed than LHS or RANDOM at large `K`. `SEED N`
discards the first `(N & 0xFFFF) * dim` Sobol points so different
seeds pick different sub-sequences (the engine is not RNG-driven).

See [`examples/04_sobol_seeded_gfs/`](examples/04_sobol_seeded_gfs/).

### WARM <file>

```
INIT
  STRATEGY WARM ../prev_run/best_pot.tadah
  INCLUDE_CONFIG false
  WARM_ON_MISMATCH warn
ENDINIT
```

Reads a previously written `*.tadah` (typically the previous run's
`best_pot.tadah`). For each OPTIM key in the current config, WARM
pulls the matching value from the file; missing keys fall through.

Two implementation details that affect reproducibility:

1. **Hot-var priming.** When the writer dumped `# HOT <script> <name>
   <value>` lines (one per `--hotvar` declaration; see the *Hot
   variables* section in HPO/README.md), WARM reads them and primes
   the per-(script, name) warmstart cache. Without this, every hot
   PC_LAMMPS script would start its box/relax from the user's
   `<initial>` (or the midpoint of `[min, max]`) rather than the
   converged value the previous run's best landed on, and might fall
   into a different basin. The diagnostic
   `[HPO] WARM: primed warmstart cache with N HOT entry(ies)`
   confirms how many lines were absorbed. Legacy
   `# HOT_A0 <script> <a0>` lines from older best_pot files still
   parse and map to `(script, "hot_a0")`.
2. **Bounds clamping.** If the bounds have shrunk between runs and the
   warm point now sits outside, each clamped dim is logged so the
   silent narrowing does not get lost in noise.

`WARM_ON_MISMATCH` controls behaviour on partial coverage:

- `warn` (default) — log to stderr and fall through to the next chain
  entry for missing keys.
- `error` — throw; treat partial coverage as a config bug.
- `skip` — return no point; let the next chain entry fully cover.

See [`examples/05_warm_refinement/`](examples/05_warm_refinement/).

#### WARM is reordered to the chain front

The resolver promotes the first WARM entry to position 0 in the chain
regardless of where you wrote it. Iter 0 is then the warm point even
when you also wrote LHS / RANDOM / SOBOL / PRIOR after it. The
diagnostic `[HPO] InitStrategy: WARM moved to chain front` fires when
the move actually changes the order.

See [`examples/11_chained_init_warm_lhs/`](examples/11_chained_init_warm_lhs/).

### PRIOR

```
INIT
  STRATEGY PRIOR
  INCLUDE_CONFIG false
ENDINIT
```

For each OPTIM-flagged dim, asks the descriptor for its
`suggested_value(key, arg_pos)` (registered per descriptor inside
MODELS). For dims whose descriptor has nothing registered (or whose
key is not descriptor-owned), the resolver falls back to the bounds
midpoint (geometric for log-scaled, arithmetic for linear).

Use as a *first-time-use* seed when you have no `pot.tadah` to WARM
from and no informed CONFIG.

See [`examples/06_prior_seeded/`](examples/06_prior_seeded/).

## INIT block top-level knobs

All inside the `INIT { ... } ENDINIT` block:

```
K_TOTAL 8                # cap on total points produced (default = sum of strategies' default_k)
SEED 12345               # seed for RANDOM/LHS/SOBOL determinism
INCLUDE_CONFIG false     # see below
WARM_ON_MISMATCH warn   # warn|error|skip — used by WARM
```

**`INCLUDE_CONFIG` and the implicit CONFIG.** Counter-intuitive:
the value of this flag matters ONLY when your chain is empty.
- empty chain + `true` (default) → CONFIG is auto-prepended.
- empty chain + `false` → no init points; you'd typically combine this
  with one explicit strategy.
- non-empty chain → implicit CONFIG is *suppressed regardless of the
  flag*. Add `STRATEGY CONFIG` yourself if you want it in.

This is the most common surprise. The diagnostic line at run start
exists to make the suppression visible.

## How many seeds the optimiser actually consumes

A multi-seed chain is only useful if the optimiser knows what to do
with extra seeds.

| Optimiser | Consumes |
|---|---|
| `DLIB:GFS` | All K (forwarded as `find_min_global` `initial_function_evals`). Each seed counts towards `MAXEVAL`. |
| `DLIB:BFGS` | First only (`optim_init.front()`). |
| `NLOPT:*` | First only. |
| `TADAH:RANDOM`, `TADAH:GRID` | First only — evaluated as eval 0, then sweep. |

`MULTISTART > 1` then runs the inner optimiser `K` independent times.
Each run consumes one seed (per-start). For `DLIB:GFS` the multistart
collapses into a single inner call because GFS already takes K seeds
natively.

When the chain produces more seeds than the optimiser will use, HPO
emits a warning at config time:

```
[HPO] WARNING: INIT chain produced 4 seeds but DLIB:BFGS consumes
only the first; 3 seed(s) will be discarded. Set TYPE MULTISTART,
add a TYPE PRESTAGE EXPLORE block, or use DLIB:GFS to use them.
```

## Run policies (STRATEGY block TYPE), one by one

### TYPE PLAIN

```
STRATEGY
  TYPE PLAIN
  LIB DLIB
  ALGO BFGS
  MAXEVAL 50
ENDSTRATEGY
```

Single-shot dispatch. The optimiser runs once from `optim_init.front()`
(or its consumer-specific seed handling — see "How many seeds the
optimiser actually consumes" above).

### TYPE MULTISTART

```
STRATEGY
  TYPE MULTISTART
  LIB DLIB
  ALGO BFGS
  MAXEVAL 100
  N_STARTS 4
  BUDGET_PER_START 25
ENDSTRATEGY
```

Runs the inner optimiser `N_STARTS` independent times, each from a
different seed in the chain. The host's best-tracker keeps the global
minimum across all runs.

`BUDGET_PER_START` overrides per-start MAXEVAL. When unset, HPO splits
the outer `MAXEVAL` across `N_STARTS` (ceil division), so `MAXEVAL 100,
N_STARTS 4` runs 25 evals per start. The diagnostic

```
[HPO] MULTISTART per-start MAXEVAL = 25 (split from outer MAXEVAL)
```

makes the split visible. With BUDGET_PER_START explicit, the suffix
changes to `(from BUDGET_PER_START)`.

`STRATEGY { TYPE MULTISTART, N_STARTS > 1 }` is rejected at
validate-time for `NLOPT:GN_*` algorithms — those globals consume one
starting point only. Pick `LD_*` / `LN_*` or `DLIB:GFS` / `DLIB:BFGS`
instead, or set `N_STARTS 1` (which is just `TYPE PLAIN`).

`PARALLEL > 1` is parsed but currently runs starts
sequentially — HPO emits a warning if you set it. Inner
`OMP_NUM_THREADS` still parallelises LAMMPS within an evaluation.

See [`examples/03_lhs_multistart/`](examples/03_lhs_multistart/).

### TYPE PRESTAGE (staged-global mode)

```
OPTIMIZER
  STRATEGY
    TYPE PRESTAGE
    LIB DLIB
    ALGO BFGS              # local refinement, runs SECOND
    MAXEVAL 60
    TOPK 3
    EXPLORE
      LIB DLIB
      ALGO GFS             # global exploration, runs FIRST
      MAXEVAL 200
    ENDEXPLORE
  ENDSTRATEGY
ENDOPTIMIZER
```

`STRATEGY { TYPE PRESTAGE }` enables staged-global mode. The
`EXPLORE { ... }` sub-block is the self-contained spec for the
**first phase**; the outer STRATEGY's `LIB`/`ALGO`/`MAXEVAL` is the
**second phase**, refining the top-`K` (lowest-loss) seeds.

The semantics:

1. HPO turns on top-K tracking, runs the EXPLORE block. While it runs,
   every eval is checked against the running top-K snapshot; the
   `K` lowest-loss `(loss, params)` pairs are kept.
2. EXPLORE finishes (MAXEVAL or stagnation). HPO snapshots and
   prints the top-K losses:
   `[HPO] PRESTAGE: top-3 losses: 0.0152 0.0186 0.0204`.
3. The outer (refinement) optimiser is invoked as a multistart over the
   K seeds (or as a single-shot run when K=1). Per-start MAXEVAL is
   then split from the outer MAXEVAL by the same rule MULTISTART uses.

The EXPLORE block's bounds are inherited from the outer spec (post
log-transform), so you do not have to re-declare `OPTIM` lines inside
`EXPLORE`.

`INNER { ... } ENDINNER` is unrelated to PRESTAGE — it carries NLopt's
MLSL/AUGLAG inner sub-algorithm. You can nest INNER inside EXPLORE
when the EXPLORE phase uses MLSL.

The legacy `STAGE_GLOBAL true` keyword has been removed; HPO will
parse-error and tell you to migrate to `STRATEGY { TYPE PRESTAGE }`.

### TYPE BASINS (anytime/interruptible)

```
OPTIMIZER
  STRATEGY
    TYPE BASINS
    EXPLORE
      LIB NLOPT
      ALGO GN_CRS2_LM
      FTOL_LADDER 1e-3 1e-4 1e-5 1e-6
      BUDGET_MAX_EVALS 5000
    ENDEXPLORE
    REFINE
      LIB NLOPT
      ALGO LN_BOBYQA
      FTOL_LADDER 1e-3 1e-4 1e-5 1e-6 1e-7
      BUDGET_MAX_EVALS 5000
    ENDREFINE
    BASIN_POOL_MAX 32
    STOP_NO_IMPROVE_ROUNDS 5
    CHECKPOINT_EVERY 60s
    MAXTIME 86400
  ENDSTRATEGY
ENDOPTIMIZER
```

Round k = 1, 2, ... alternates an EXPLORE stage at
`EXPLORE.FTOL_LADDER[k]` (broad exploration, harvests basin
candidates) with a REFINE stage that visits the cumulative
`REFINE.FTOL_LADDER[1..k]` and refines every basin in the persistent
pool. Designed for hour/day/week-long runs with SIGINT/SIGTERM
tolerance — checkpoint commits at every stage boundary plus every
`CHECKPOINT_EVERY`.

See [`examples/15_basins_run_and_forget/`](examples/15_basins_run_and_forget/)
and [`examples/07_stage_global/`](examples/07_stage_global/).

## RESTART decorator

```
OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB DLIB
    ALGO BFGS
    MAXEVAL 1000
    RESTART
      AFTER_STAGNATION 80
      MAX 5
      STRATEGY SOBOL 1
    ENDRESTART
  ENDSTRATEGY
ENDOPTIMIZER
```

`RestartPolicy` watches `evals_since_last_best`. When it reaches
`RESTART { AFTER_STAGNATION }`, HPO throws `StagnationStop` from inside
the next eval; the policy catches it, draws a fresh init from
`RESTART { STRATEGY ... }` (uses the same INIT-block grammar inside),
resets the counter, and re-enters the inner optimiser.

The cycle terminates when:
- the inner policy returns naturally (MAXEVAL exhausted, ftol etc.), or
- `MAX` attempts have been used.

Each restart gets its own stagnation window; the windows do not bleed
into each other. The fresh-init chain follows the usual chain rules,
EXCEPT that implicit CONFIG is not auto-prepended (a stagnated run by
definition has CONFIG in its history already).

RESTART composes with `TYPE PLAIN`, `TYPE MULTISTART`, and
`TYPE PRESTAGE`. It does NOT compose with `TYPE BASINS` (BASINS
drives its own global+local cycle and supersedes the decorator stack).

See [`examples/08_restart_on_stagnation/`](examples/08_restart_on_stagnation/).

## NOISE block

```
OPTIMIZER
  NOISE
    AUTO
    CALIBRATION 0
  ENDNOISE
  STRATEGY
    TYPE PLAIN
    LIB DLIB
    ALGO BFGS
    MAXEVAL 200
  ENDSTRATEGY
ENDOPTIMIZER
```

`DLIB:BFGS` uses a finite-difference gradient. The default `PARAM
fd_step = 1e-6` is far smaller than the noise floor of a typical
LAMMPS-driven evaluation — the computed gradient is then dominated by
noise rather than signal.

`NOISE.AUTO` derives `fd_step` from the search-space geometry:

```
fd_step = max(1e-3 × diag / sqrt(dim), 1e-6)
```

where `diag` is the L2 norm of `(high - low)` in the post-transform
space.

Restrictions:
- Only `DLIB:BFGS`. For other engines NOISE.AUTO is a no-op with a
  stderr note (`[HPO] NOISE.AUTO is implemented for DLIB:BFGS only ...
  ignored`).
- An explicit `PARAM fd_step <value>` always wins; the heuristic is
  suppressed and HPO logs that.

The fired-value diagnostic:

```
[HPO] NOISE.AUTO: PARAM fd_step set to 1.43e-03 (diag=2.022, dim=2)
```

See [`examples/09_noise_auto_bfgs/`](examples/09_noise_auto_bfgs/).

## Worked scenarios

### Scenario 1 — first-time use on a new system

You have a new material and bounds you trust, but no good seed.

```
OPTIMIZER
  INIT
    STRATEGY PRIOR
    STRATEGY LHS 4
    INCLUDE_CONFIG false
  ENDINIT
  STRATEGY
    TYPE MULTISTART
    LIB DLIB
    ALGO BFGS
    MAXEVAL 200
    N_STARTS 5
    BUDGET_PER_START 40
  ENDSTRATEGY
ENDOPTIMIZER
```

What happens:

- `PRIOR` produces 1 point.
- `LHS 4` adds 4 stratified points.
- `INCLUDE_CONFIG false` keeps the chain to those 5 (no CONFIG).
- `N_STARTS 5` runs BFGS five times, one per seed.
- `BUDGET_PER_START 40` caps each BFGS pass at 40 evals.

Trade-off: `N_STARTS K` multiplies wall time by ≈ K. Always set
`BUDGET_PER_START` (or rely on the auto split from `MAXEVAL`) —
otherwise each start runs to its own MAXEVAL and total wall time
blows up.

### Scenario 2 — iterative tuning of a known-good baseline

You already trained once and got a `pot.tadah` you liked. You widened
a bound or relaxed a constraint. Refine.

```
OPTIMIZER
  INIT
    STRATEGY WARM ../prev/best_pot.tadah
    INCLUDE_CONFIG false
    WARM_ON_MISMATCH warn
  ENDINIT
  STRATEGY
    TYPE PLAIN
    LIB DLIB
    ALGO BFGS
    MAXEVAL 80
  ENDSTRATEGY
ENDOPTIMIZER
```

What happens:

- WARM reads the previous `best_pot.tadah` and pulls every OPTIM key.
- For any key WARM cannot supply, it falls through. Without other
  chain entries, the fallback is the in-config value next to the OPTIM
  line — even though `INCLUDE_CONFIG false` is set. (This is by
  design: WARM emits *one point* with all dims filled. Falling through
  for missing dims uses the in-config dim value, not a chain re-roll.)
- BFGS refines from this single combined seed.

### Scenario 3 — suspected multimodal landscape

```
OPTIMIZER
  STRATEGY
    TYPE PRESTAGE
    LIB DLIB
    ALGO BFGS
    MAXEVAL 60
    TOPK 3
    EXPLORE
      LIB DLIB
      ALGO GFS
      MAXEVAL 200
    ENDEXPLORE
  ENDSTRATEGY
ENDOPTIMIZER
```

GFS canvases the box for 200 evals; the top 3 (loss, params) pairs
seed three BFGS refinements (≈ 20 evals each). HPO's best-tracker
spans both phases.

### Scenario 4 — long unattended run with stagnation watchdog

```
OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB DLIB
    ALGO BFGS
    MAXEVAL 1000
    RESTART
      AFTER_STAGNATION 80
      MAX 5
      STRATEGY SOBOL 1
    ENDRESTART
  ENDSTRATEGY
ENDOPTIMIZER
```

After 80 evals without a new best the policy restarts from a fresh
Sobol point; up to 5 restarts. A good window starting point is a few
times the dimensionality (e.g. 50–150 for 10–30-dim search).

For week-long unattended runs prefer `STRATEGY { TYPE BASINS }` —
it commits checkpoints at every stage boundary and resumes cleanly
after SIGINT/SIGTERM (see example 15).

### Scenario 5 — noisy objective; derivative-free local

```
OPTIMIZER
  INIT
    STRATEGY PRIOR
  ENDINIT
  STRATEGY
    TYPE PLAIN
    LIB NLOPT
    ALGO LN_SBPLX
    MAXEVAL 200
  ENDSTRATEGY
ENDOPTIMIZER
```

`LN_SBPLX` (Subplex), `LN_BOBYQA`, and `LN_COBYLA` are
derivative-free. They never differentiate the loss, so small-amplitude
noise does not crash the search.

If you want to keep BFGS, use `NOISE { AUTO }` to widen the
finite-difference step (see the NOISE section above).

## What to leave alone

- `PARALLEL` — parsed but currently runs sequentially. The inner
  `OMP_NUM_THREADS` already parallelises LAMMPS within an evaluation,
  so the loss is mostly cosmetic for now.
- `INIT { K_TOTAL }` — a hard cap on starting points. You rarely need
  it; each strategy has a sensible `default_k`.
- `INIT { SEED }` only matters if you want bit-reproducible runs. The
  inherent noise of LAMMPS-driven evaluations dwarfs init-seed effects
  after a handful of evals.

## When in doubt

- `STRATEGY { TYPE PLAIN, LIB DLIB, ALGO BFGS }` first
  ([`examples/01_baseline_bfgs/`](examples/01_baseline_bfgs/)).
  Faster to debug; the loss it reaches is a sane reference.
- If it stalls, add a `RESTART` block before adding multistart — it
  is cheaper.
- Add `STRATEGY { TYPE MULTISTART }` only when you suspect multimodality
  and you have a budget for `K × per-start_evals`.
- Add `STRATEGY { TYPE PRESTAGE EXPLORE { ... } }` when you can afford
  a generous global exploration budget (it has to find the basins).
- For hour/day-long unattended runs, jump straight to
  `STRATEGY { TYPE BASINS }` (example 15).
