# Tadah!HPO — load-time dataset transforms

This guide covers the load-time transforms that reshape a training set
before any neighbour finding, descriptor calculation, or training. They
exist to bridge two gaps:

1. **DFT vs experiment** — DFT lattice constants typically miss
   experiment by a few %, and DFT total energies have a per-element
   offset that depends on pseudopotential / functional. A potential fit
   on raw DFT data inherits both. Tadah! lets you absorb either at load
   time so the trained model "lives" at the experimental scale and at
   a sensible energy baseline.
2. **Mixed-quality / mixed-size datasets** — when you train on a few
   critical structures alongside thousands of routine MD frames, the
   routine frames numerically dominate the loss. Tadah! gives you knobs
   (per-dataset weights, automatic size balancing, Boltzmann reweighting,
   outlier filters) to rebalance the loss without touching the input
   files.

All transforms run inside `apply_load_transforms()` immediately after
`StructureDB::add(...)` and before any neighbour finding. They are
shared across `tadah train`, `tadah predict`, and `tadah hpo`. The
two transforms that are also used at predict time (`LSCALE`, `ESHIFT`)
are stamped into `pot.tadah` so prediction round-trips them
automatically.

For per-key reference (defaults, examples, exact CLI form) see
`tadah explain <key>` (e.g. `tadah explain train.lscale`). This guide
is the conceptual layer.

## Mental model

A freshly-loaded `StructureDB` is just a list of structures with
positions, energies, forces, stresses, and per-structure weights. The
load-time transforms are functions of `(StructureDB, Context) →
StructureDB` plus a `LoadTransformReport` that summarises what was
done. The orchestrator runs them in this order:

```
1. Outlier filter (EFILTER, FFILTER)        — drop bad configs first
2. Length rescale (LSCALE)                  — geometric only
3. ESHIFT determination + apply              — explicit > ATOM > DBATOM
4. Boltzmann eweight (EWEIGHT_TEMP)         — uses post-shift energies
5. COM force zeroing (ZERO_COM_FORCE)
6. Per-dataset weights (WDBFILE * WDBFILE_AUTO)
7. Equilibrium-volume diagnostic
```

Order matters. EFILTER runs first so an extreme outlier doesn't poison
ESHIFT_DBATOM's least-squares fit; ESHIFT_DBATOM runs before
EWEIGHT_TEMP so the Boltzmann factor sees user-meaningful (shifted)
energies. The diagnostic at the end records the lowest-energy
configuration after all transforms so you can sanity-check the
reference state.

## The transforms, one by one

### LSCALE — uniform length rescale

```
LSCALE 1.005          # in config.hpo / config.tadah
--lscale 1.005        # CLI
```

Multiplies positions and cell vectors by `s = LSCALE`. To keep the
total energy invariant under a change of length units, forces are
divided by `s` (chain rule on `E(r)`). Stresses are stored as virial
(`V·σ_Cauchy`, energy units) so they are invariant under uniform length
rescaling and are not touched.

The intended workflow:

1. Your DFT dataset has `a_DFT` (e.g. 3.30 Å for bcc Ta).
2. You want the model to reproduce `a_exp` (e.g. 3.31 Å) in LAMMPS.
3. Set `LSCALE = a_exp / a_DFT` (e.g. 1.003).
4. Train. The model is fit on the rescaled dataset, so its
   equilibrium volume is at the experimental scale.
5. Run LAMMPS with starting positions at `a_exp`. The model produces
   correct forces and energies for that scale; equilibrium relaxes
   to `a_exp`. ✓

**Pot.tadah → predict round-trip is closed.** `tadah predict` reads
LSCALE from pot.tadah and re-applies it to whatever dataset you pass.
This means you can `tadah train --lscale 1.005 …` and then
`tadah predict -p pot.tadah -d original_DFT_data.tadah` and the
predicted energies will match training references.

**Pot.tadah → LAMMPS is one-way by design.** The LAMMPS pair_style does
NOT re-apply LSCALE to its input positions. The user is expected to
supply LAMMPS with positions at the trained scale (typically the
experimental scale). pot.tadah carries LSCALE for documentation and so
`tadah predict` can round-trip; LAMMPS silently ignores the key.

**Cancellation at predict time.** `--no-lscale` makes `tadah predict`
ignore the LSCALE recorded in pot.tadah. Use this when the dataset
you're predicting on is already at the trained scale.

### ESHIFT — runtime per-element energy shift

```
# Explicit, one value per element (ordered by Z)
ESHIFT -3.0 -7.0       # ; or
--eshift -3.0 -7.0

# Auto-derive from isolated-atom configs in the dataset
ESHIFT_ATOM true       # ; or
--eshift-atom

# Auto-derive by least-squares fit over the whole database
ESHIFT_DBATOM true     # ; or
--eshift-dbatom
```

The applied transformation is, for every configuration:

```
E_after = E_before − Σ_Z N_Z(config) · ESHIFT[Z]
```

The unified Tadah! convention is *subtraction*: `ESHIFT[Z]` is "the
per-atom reference energy of species Z, to be removed". So setting
`ESHIFT[Ti] = -3.0` (the isolated-atom Ti energy) on a 4-atom Ti config
with `E = -11.0` eV gives `-11.0 − 4·(-3.0) = +1.0` eV — the cohesive
energy. The same convention is used by `tadah data balance --eshift`,
which differs from older releases (≤ pre-2026 where `data.balance`
*added* the shift); the changelog has details.

**Three ways to determine ESHIFT.** Mutually exclusive — if you set
more than one, the engine warns and applies precedence: explicit >
ATOM > DBATOM.

- **Explicit `ESHIFT a b c…`** — you provide the values directly. One
  per unique species, ordered by Z.
- **`ESHIFT_ATOM`** — Tadah! scans the dataset for single-atom
  configurations (`natoms == 1`), groups them by Z, and uses the mean
  energy per group. If a species has no isolated-atom config, ESHIFT
  for that Z is left at 0 (no shift) and a WARNING is logged. If
  multiple configs of the same Z disagree by > 1 meV, an INFO line
  records the spread.
- **`ESHIFT_DBATOM`** — Tadah! solves `min ‖y − Mβ‖²` where `y[i]` is
  the total energy of config `i` and `M[i,k]` is the count of species
  `k` in config `i`. Fitted `β_k` becomes `ESHIFT[Z(k)]`. More robust
  than ATOM when the dataset has compositional diversity but no
  isolated-atom configs (NEP/MACE/GAP all do something similar).

**Pot.tadah round-trip.** Whichever path produced the values, the
*resolved* per-element shifts are stamped into pot.tadah as plain
ESHIFT entries. The ATOM/DBATOM toggles are not stamped — once
derivation is done, only the values matter.

**Cancellation.** `tadah predict --no-eshift` ignores the ESHIFT
recorded in pot.tadah. Use this when the dataset you're predicting on
is already at the shifted baseline, or when you want raw model output
without any reference subtraction.

### EFILTER / FFILTER — outlier dropping

```
EFILTER -12.0 -2.0        # drop configs with E/N outside [-12, -2] eV
FFILTER 20.0              # drop configs with any |F_i| > 20 eV/Å
```

`EFILTER` thresholds are compared against **per-atom** energy `E/N`
(this is the more physical interpretation — a 100-atom config and a
1-atom config are scored on the same scale).

`FFILTER` drops the configuration if any single atom has force
magnitude exceeding the threshold. Useful for catching unconverged SCF
or genuinely broken DFT runs.

Both filters run *first* in the pipeline so dropped outliers don't
poison ESHIFT_ATOM / ESHIFT_DBATOM / EWEIGHT_TEMP / equilibrium-volume
diagnostics. The INFO log shows the counts:

```
Outlier filter: dropped 12/582 by E, 3/582 by F
```

Filters are **training-time only** — they are not stamped into
pot.tadah and not re-applied at predict time.

### WDBFILE — explicit per-dataset weight multipliers

```
DBFILE  defects.tadahdb
DBFILE  md_300K.tadahdb
DBFILE  surfaces.tadahdb
WDBFILE  5.0  1.0  1.0       # one per DBFILE
```

Multiplies `eweight`, `fweight`, and `sweight` of every configuration
in dataset `i` by `WDBFILE[i]`. Use to bias training toward (or away
from) particular datasets. `WDBFILE.size()` must equal `DBFILE.size()`.

### WDBFILE_AUTO — automatic size balancing

```
WDBFILE_AUTO 0.5         # alpha = 0.5 (recommended starting point)
```

Multiplies dataset `i`'s per-config weight by `1 / N_i^α`, where `N_i`
is the post-EFILTER/FFILTER count of configurations in `DBFILE[i]`.
The single parameter `α ∈ [0, 1]` controls how aggressively to
rebalance:

| α    | Per-dataset factor `N^(−α)` | Effect                                                     |
|------|------------------------------|-------------------------------------------------------------|
| 0.0  | 1.0 (disabled, default)      | No size-balancing.                                          |
| 0.5  | 1/√N                         | **Soft balance.** Recommended starting point.               |
| 1.0  | 1/N                          | **Full balance** — each dataset contributes equal aggregate loss. |

Numbers are the per-config weight multiplier — a small dataset with
N=2 gets a ×0.71 (α=0.5) or ×0.5 (α=1.0) factor; a large dataset with
N=1000 gets ×0.032 or ×0.001 respectively. The smaller dataset's
*aggregate* contribution to the loss is `N · factor = N^(1−α)`, which
goes to 1 as α → 1.

Composes multiplicatively with WDBFILE. The classic case — 2 critical
defect configs vs 1000 routine MD frames, with the user wanting
defects 5× more important than the auto-balanced baseline:

```
DBFILE  defects.tadahdb     # N=2
DBFILE  md_300K.tadahdb     # N=1000
WDBFILE       5.0   1.0
WDBFILE_AUTO  0.5
```

Composed weights:

- defects: 5.0 · (1/√2)    ≈ 3.5355
- md:      1.0 · (1/√1000) ≈ 0.0316

**Validation.** `α < 0` or `α > 1` is rejected. Dataset counts and
factors are logged at INFO.

Training-time only — not stamped into pot.tadah.

### EWEIGHT_TEMP — Boltzmann reweighting

```
EWEIGHT_TEMP 300         # Kelvin
```

After ESHIFT is applied, multiplies each configuration's `eweight`
by `exp(-(E/N - E_min)/(kB·T))` where `E_min` is the minimum per-atom
energy in the dataset and `kB = 8.617333262 × 10⁻⁵ eV/K`. This biases
the loss toward low-energy configurations, which is often what you
want for property fits (the GS energy and a few low-T MD frames matter
much more than high-T or distorted snapshots).

Composes multiplicatively with the per-structure `eweight` already in
the dataset file. INFO log records the resulting `[eweight_min,
eweight_max]` range. `EWEIGHT_TEMP <= 0` is a validation error — omit
the key to disable.

### ZERO_COM_FORCE — neutralise net force

```
ZERO_COM_FORCE true
```

Per configuration, subtracts the mean force so that `Σ F_i = 0`
exactly. Useful when DFT runs leave residual translational forces from
incomplete relaxation/SCF. INFO log records the count of touched
configs and the largest residual seen.

### Equilibrium-volume diagnostic

No key. After every other transform, the orchestrator finds the
configuration with the lowest *post-shift* per-atom energy and logs
its label, energy/atom, and volume/atom:

```
Reference state (min-energy config): label='Ta_bcc_eq' E/N=-12.34 eV V/N=15.62 Å³
```

Goes into both `hpo_run.log` (LOAD TRANSFORMS section) and the header
of best_pot.tadah — handy for sanity-checking the trained model
against the dataset's own equilibrium reference.

## Where the transforms live in HPO

The HPO host loads training and validation StructureDBs once at
construction. `apply_load_transforms` is called on the training set
with the master Context; the resulting `LoadTransformReport` is held
on the host as `report_`. The validation set is then transformed
with a **stripped** context that contains:

* `LSCALE` from the master config (so geometric scale matches),
* the **resolved** `ESHIFT` from the training run (so ESHIFT_ATOM /
  ESHIFT_DBATOM derivations propagate; raw `ESHIFT_ATOM` /
  `ESHIFT_DBATOM` keys are intentionally *not* re-derived on
  validation),
* `EFILTER` and `FFILTER` **only if they appear in the --validation
  file itself**.

Training-only knobs (`EFILTER` / `FFILTER` set in `master_ctx_`,
`WDBFILE`, `WDBFILE_AUTO`, `EWEIGHT_TEMP`, `ZERO_COM_FORCE`) are
never forwarded to the validation set: they would silently change
what the validation metric is computed on without the user opting in,
breaking apples-to-apples comparison across HPO trials.

To filter the validation set, declare the threshold in the file you
pass to `--validation` directly:

```
# validation.in
DBFILE   val_eos.tadahdb
EFILTER  -12.0  -2.0       # honoured for validation only
FFILTER   20.0
```

The validation-side report is held on the host as `val_report_` and
the host emits a single `INFO` log line at startup of the form

```
Validation set: 532 configs loaded; dropped 4 by EFILTER, 1 by FFILTER; 527 remain.
```

so it's clear which configs the validation metric is being computed
on. The line is emitted unconditionally — when no filter is active
the dropped counts are zero and the line still confirms the loaded
size.

The report is consumed in three places:

1. **`hpo_run.log` `LOAD TRANSFORMS` section** — written once at host
   startup, summarises every applied transform.
2. **`best_pot.tadah` header** — embedded as a comment block alongside
   the existing DATA / OPTIMISER / LOSS sections.
3. **`best_pot.tadah` body** — `LSCALE` and resolved `ESHIFT` keys are
   stamped into the per-eval pot context so the file round-trips
   through `tadah predict`.

The trainer itself doesn't know about load-time transforms; the host
injects them immediately after `model->get_param_file()` returns its
parameter context.

## Worked example: bcc Ta with experimental lattice match

Say you trained a Ta potential on DFT data with `a_DFT = 3.30 Å`, and
the experimental lattice is `a_exp = 3.31 Å`. You also want to bias
the training toward low-temperature configs and rebalance two
datasets that differ by ~500× in size.

`config.tadah`:

```
DBFILE   ta_eos.tadahdb            # N ≈ 20
DBFILE   ta_md_500K.tadahdb        # N ≈ 10000

LSCALE         1.003                # 3.31 / 3.30
ESHIFT_ATOM    true                 # baseline from your isolated-atom configs

WDBFILE_AUTO   0.5                  # soft size balance

EWEIGHT_TEMP   500                  # de-emphasise hot configs

EFILTER       -12.0  -2.0           # drop garbage E/N
FFILTER        20.0                 # drop unconverged SCF

# … rest of MODEL/TYPE2B/RCUT2B unchanged …
```

After training, inspect `pot.tadah`:

```
LSCALE 1.003
ESHIFT -8.4567        # whatever ESHIFT_ATOM derived from your single-Ta config
```

Predict on the original DFT dataset:

```
tadah predict -p pot.tadah -d ta_eos.tadahdb --analytics
# energies match training references — round-trip closed
```

Predict on the same dataset but ignore the stored transforms:

```
tadah predict -p pot.tadah -d ta_eos.tadahdb --analytics --no-lscale --no-eshift
# energies differ from training references by Σ_Z N_Z·ESHIFT[Z] per config
```

Run LAMMPS with positions at `a_exp = 3.31 Å`:

```
pair_style tadah
pair_coeff * * pot.tadah Ta
# … LAMMPS produces forces/energies in the trained scale ✓
```

LAMMPS sees `LSCALE 1.003` in pot.tadah but does nothing with it.

## What's not covered here

- **Anisotropic length rescale** (separate per-axis or per-element)
  is intentionally deferred — needs more plumbing for marginal benefit
  over uniform LSCALE.
- **Position / cell augmentation** (jitter on inputs to grow effective
  dataset size) is deferred — non-trivial validation surface.
- **MPI training** does not yet wire LSCALE / explicit ESHIFT through
  `TrainerWorker`. `ESHIFT_ATOM` / `ESHIFT_DBATOM` / `EWEIGHT_TEMP` /
  `WDBFILE_AUTO` need a global-view pass over the whole database that
  is not currently available on workers; the engine fails fast if you
  set any of these under MPI. Run without MPI, set explicit ESHIFT
  values, or wait for the MPI plumbing to land.
